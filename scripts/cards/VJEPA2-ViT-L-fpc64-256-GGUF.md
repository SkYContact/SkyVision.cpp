---
license: mit
base_model:
- facebook/vjepa2-vitl-fpc64-256
base_model_relation: quantized
library_name: gguf
pipeline_tag: feature-extraction
language:
- en
- zh
tags:
- vision
- video
- gguf
- v-jepa2
- world-model
- jepa
- 3d-rope
- masked-latent-prediction
- skyvision.cpp
---

# V-JEPA 2 ViT-L (fpc64-256) GGUF

[facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) 的 GGUF 版本，由自研推理引擎
[SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp)（纯 ggml，无 llama.cpp 依赖）转换并逐级对拍验证。
权重原样保留（encoder F16），没有任何训练。

> **这是世界模型的编码器，不是对话模型。** 一次前向把 64 帧视频压成一个 1024 维向量，
> 没有 tokenizer、没有 lm head、不生成文本。
>
> **仅供 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 使用。** GGUF 里
> `general.architecture = vjepa2`，且三维分解 RoPE 复现了上游一个"频率平铺"的历史 bug
> （[vjepa2 PR #15](https://github.com/facebookresearch/vjepa2/pull/15)，修正会破坏预训练权重兼容性），
> 权重里的 q/k 行按该布局重排过。**llama.cpp / ollama / LM Studio 等通用 GGUF 运行时无法加载本文件。**

## 文件

| 文件 | 大小 | 内容 |
|---|---|---|
| `vjepa2-vitl-fpc64-256-f16.gguf` | 0.61 GiB (653 MB) | encoder（326M 参数，F16）+ masked-latent predictor（12 层 d=384，F16） |

一个文件装全部权重：24 层 encoder + predictor，predictor 用 `--mode predict` 驱动。

## 模型做了什么

| 输入 | 输出 |
|---|---|
| 64 帧视频（预处理：短边缩到 292 → 中心裁 256） | 1024 维 mean-pooled 特征 |
| 同上 + mask 规格（`ctx=0:4096,tgt=4096:8192`） | 被遮住的那一半 token 的预测潜变量（1024 维 × token 数） |

结构（与官方 transformers `VJEPA2Model` 逐 op 对齐）：

- Conv3d tubelet patch 嵌入（kernel = stride = (2, 16, 16)）折叠成一次 matmul，
  64 帧 × 256² → **8192 个 token**
- **没有位置表**：每层内部对 q/k 做三维分解 RoPE，head_dim 64 切成
  20（时间）+ 20（高）+ 20（宽）+ 4（不旋转），位置取自 token 的 `(t', h', w')` 网格
- 24 层全时空注意力（16 头，head_dim 64），q/k/v 与输出投影**全部带 bias**，MLP 4096 宽 + GELU(erf)，LN eps 1e-6
- encoder 末尾 LayerNorm → 对 token 轴求均值 → 1024 维
- predictor：特征 1024→384 投影 → 拼 mask token → **按位置排序** → 12 层窄栈（自己的 RoPE 表）
  → LN → 反排序取 target token → 384→1024 投影

## 使用

```bash
git clone https://github.com/SkYContact/SkyVision.cpp
cd SkyVision.cpp && git clone --depth 1 https://github.com/ggml-org/ggml
scripts\build-cuda.bat            # 或 scripts/build-cpu.sh

hf download SkyContact/VJEPA2-ViT-L-fpc64-256-GGUF --local-dir models/VJEPA2

# 1024 维特征（JSON）
vqa-cli --model models/VJEPA2/vjepa2-vitl-fpc64-256-f16.gguf \
        --video clip.mp4 --ask x --mode embed

# 同一特征，L2 归一化后输出（适合做检索/聚类）
vqa-cli --model models/VJEPA2/vjepa2-vitl-fpc64-256-f16.gguf \
        --video clip.mp4 --embed ""

# 掩码潜变量预测：前一半 token 当上下文，预测后一半（区间 a:b 半开）
vqa-cli --model models/VJEPA2/vjepa2-vitl-fpc64-256-f16.gguf \
        --video clip.mp4 --mode predict --ask "ctx=0:4096,tgt=4096:8192"
```

图像也能直接喂（当成单帧 clip，不需要 ffmpeg）：`--image photo.jpg`。

## 精度验证（vs 官方 transformers 5.x 实现）

| 对照 | 结果 |
|---|---|
| **同一份 patch 输入**（把引擎的输入张量灌回官方实现）24 层全长 | 逐 token cos **1.000000**，max 0.065 —— 图实现本身没有偏差 |
| 端到端（各自预处理）patch 嵌入 / 第 0 层 | 相对差 6.9e-4 / 7.1e-4 |
| 端到端 mean-pooled 1024 维特征 | cos **0.9994 ~ 0.9998** |
| masked-latent predictor | cos **0.998** |
| 线性层权重 | F16（`--outtype f32` 可导出 1.2 GB 版本，用于把"权重精度"和"预处理差异"分离） |

端到端残差全部来自重采样：本引擎走 PIL 语义，官方 video processor 走 torch 抗锯齿，
像素级 mean 差 1.7e-4（max 0.0175 归一化，约半个灰阶）。8192 token 的全注意力对
亚灰阶输入差极其敏感，所以逐 token 看会放大、均值池化后又收敛。

## 已知限制

- 只出特征 / 潜变量，不能生成文本；`--mode embed` 之外没有别的用法
- 单 clip 形状固定（64 帧 256²），采样帧数由引擎按 clip 长度均匀取；CPU 上 19 线程约 **25-30 s/clip**
- 权重是 F16：需要的显存/内存约为文件大小；GPU 上建议直接跑 f16
- 官方另有 ViT-g（fpc64-384）与 V-JEPA 2-AC（带动作条件），本仓库未包含

## 许可

上游权重 [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) 为 **MIT**，
本 GGUF 沿用同一许可。转换脚本、推理引擎为 MIT（见 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp)）。
