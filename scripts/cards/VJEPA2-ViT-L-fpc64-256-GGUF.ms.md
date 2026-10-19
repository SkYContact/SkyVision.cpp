---
license: mit
base_model: facebook/vjepa2-vitl-fpc64-256
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

[facebook/vjepa2-vitl-fpc64-256](https://modelscope.cn/models/facebook/vjepa2-vitl-fpc64-256) 的 GGUF 版本，由自研推理引擎
[SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp)（纯 ggml，无 llama.cpp 依赖）转换并逐级对拍验证。
权重原样保留（encoder F16），没有任何训练。

> **这是世界模型的编码器，不是对话模型。** 一次前向把 64 帧视频压成一个 1024 维向量，
> 没有 tokenizer、没有 lm head、不生成文本。
>
> **仅供 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 使用。** GGUF 里
> `general.architecture = vjepa2`，且三维分解 RoPE 复现了上游一个"频率平铺"的历史 bug
> （vjepa2 PR #15，修正会破坏预训练权重兼容性），权重里的 q/k 行按该布局重排过。
> **llama.cpp / ollama / LM Studio 等通用 GGUF 运行时无法加载本文件。**

## 文件

| 文件 | 大小 | 内容 |
|---|---|---|
| `vjepa2-vitl-fpc64-256-f16.gguf` | 0.61 GiB (653 MB) | encoder（326M 参数，F16）+ masked-latent predictor（12 层 d=384，F16） |

## 模型做了什么

| 输入 | 输出 |
|---|---|
| 64 帧视频（预处理：短边缩到 292 → 中心裁 256） | 1024 维 mean-pooled 特征 |
| 同上 + mask 规格（`ctx=0:4096,tgt=4096:8192`） | 被遮住的那一半 token 的预测潜变量 |

结构（与官方 transformers `VJEPA2Model` 逐 op 对齐）：Conv3d tubelet patch 嵌入折叠成一次 matmul
（64 帧 × 256² → 8192 token）；**没有位置表**，每层内部做三维分解 RoPE（head_dim 64 = 20 时间 +
20 高 + 20 宽 + 4 不旋转，位置取自 `(t', h', w')` 网格）；24 层全时空注意力（16 头），
q/k/v 与输出投影全带 bias，MLP 4096 + GELU(erf)，LN eps 1e-6；末尾 LayerNorm 后对 token 求均值。

## 使用

```bash
git clone https://github.com/SkYContact/SkyVision.cpp
cd SkyVision.cpp && git clone --depth 1 https://github.com/ggml-org/ggml
scripts\build-cuda.bat            # 或 scripts/build-cpu.sh

modelscope download SkyContact/VJEPA2-ViT-L-fpc64-256-GGUF --local_dir models/VJEPA2

vqa-cli --model models/VJEPA2/vjepa2-vitl-fpc64-256-f16.gguf \
        --video clip.mp4 --ask x --mode embed                       # 1024 维特征 JSON
vqa-cli --model models/VJEPA2/vjepa2-vitl-fpc64-256-f16.gguf \
        --video clip.mp4 --embed ""                                 # L2 归一化
vqa-cli --model models/VJEPA2/vjepa2-vitl-fpc64-256-f16.gguf \
        --video clip.mp4 --mode predict --ask "ctx=0:4096,tgt=4096:8192"
```

图像也能直接喂（当成单帧 clip）：`--image photo.jpg`。

## 精度验证（vs 官方 transformers 5.x 实现）

| 对照 | 结果 |
|---|---|
| **同一份 patch 输入** 24 层全长 | 逐 token cos **1.000000**，max 0.065 |
| 端到端 patch 嵌入 / 第 0 层 | 相对差 6.9e-4 / 7.1e-4 |
| 端到端 mean-pooled 特征 | cos **0.9994 ~ 0.9998** |
| masked-latent predictor | cos **0.998** |

端到端残差来自重采样（本引擎 PIL 语义 vs 官方 torch 抗锯齿，约半个灰阶），
8192 token 全注意力会把亚灰阶差异放大，均值池化后收敛。

## 已知限制

- 只出特征 / 潜变量，不能生成文本
- 单 clip 形状固定（64 帧 256²）；CPU 19 线程约 25-30 s/clip
- 官方另有 ViT-g（fpc64-384）与 V-JEPA 2-AC（动作条件），本仓库未包含

## 许可

上游权重为 **MIT**，本 GGUF 沿用同一许可；转换脚本与推理引擎为 MIT。
