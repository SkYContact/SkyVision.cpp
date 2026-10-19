---
license: mit
base_model: facebook/vjepa2-vitl-fpc16-256-ssv2
base_model_relation: quantized
library_name: gguf
pipeline_tag: video-classification
language:
- en
tags:
- vision
- video
- gguf
- v-jepa-2
- v-jepa2
- video-classification
- ssv2
- 3d-rope
- attentive-pooler
- skyvision.cpp
---

# V-JEPA 2 ViT-L (fpc16-256, SSv2) GGUF

[facebook/vjepa2-vitl-fpc16-256-ssv2](https://modelscope.cn/models/facebook/vjepa2-vitl-fpc16-256-ssv2) 的 GGUF 版本：
V-JEPA 2 ViT-L 在 **Something-Something v2** 上微调得到的 174 类视频动作分类模型，
由自研推理引擎 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp)（纯 ggml）转换并逐级对拍验证。

> **仅供 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 使用。** GGUF 里
> `general.architecture = vjepa2`，三维分解 RoPE 复现了上游"频率平铺"的历史 bug，权重里的 q/k 行
> 已按该布局重排；llama.cpp / ollama / LM Studio 无法加载本文件。

## 文件

| 文件 | 大小 | 内容 |
|---|---|---|
| `vjepa2-vitl-fpc16-256-ssv2-f16.gguf` | 0.70 GiB (717 MB) | encoder（24 层，F16）+ attentive pooler（3 层）+ 174 类线性头 + masked-latent predictor |

174 个 SSv2 标签随权重写进 GGUF（`vjepa2.labels`），`--mode N` 直接打印标签名，不需要外挂 id2label。

## 结构差异（相对 fpc64-256 预训练版）

- **16 帧**输入 → 16/2 × 16 × 16 = **2048 个 token**（预训练版是 64 帧 / 8192 token）
- **attentive pooler**：3 层普通 pre-norm 自注意力（无 RoPE、无 mask）跑完整 token 流，
  再用一个学出来的 query token 对全序列做 cross attention（**该 cross attention 没有输出投影**，
  上游如此），query 残差进 MLP；输出 1024 维
- 线性分类头 1024 → 174；有 pooler 时 `--mode embed` 返回的也是 pooler 输出（与官方
  `get_vision_features` 一致），不再走 mean pooling
- 其余与预训练版相同：三维分解 RoPE（20+20+20+4）、24 层全时空注意力、q/k/v 与输出投影带 bias、
  MLP 4096 + GELU(erf)、LN eps 1e-6

## 使用

```bash
git clone https://github.com/SkYContact/SkyVision.cpp
cd SkyVision.cpp && git clone --depth 1 https://github.com/ggml-org/ggml
scripts\build-cuda.bat

modelscope download SkyContact/VJEPA2-ViT-L-fpc16-256-SSv2-GGUF --local_dir models/VJEPA2-SSv2

vqa-cli --model models/VJEPA2-SSv2/vjepa2-vitl-fpc16-256-ssv2-f16.gguf \
        --video clip.mp4 --ask x --mode 5           # top-5 标签
vqa-cli --model models/VJEPA2-SSv2/vjepa2-vitl-fpc16-256-ssv2-f16.gguf \
        --video clip.mp4 --ask x --mode embed       # 池化后的 1024 维特征
```

SSv2 的类别是"动作 + 两个物体的交互"（如 `Moving [something] and [something] so they pass each other`），
对日常随手拍片段不一定有合适类别，属正常现象。

## 精度验证（vs 官方 transformers 5.x 实现）

| 对照 | 结果 |
|---|---|
| **同一份 patch 输入** 24 层全长 | 逐 token cos **1.000000**，max 0.065 |
| 端到端 patch 嵌入 / 第 0 层 | 相对差 6.9e-4 / 7.1e-4 |
| attentive pooler 输出 | cos **0.9994** |
| 174 类概率 | 最大差 **4.5e-3**，**top-1 标签一致**，top-4 集合一致 |

端到端残差来自重采样（PIL 语义 vs 官方 torch 抗锯齿，约半个灰阶），不是图实现差异。

## 已知限制

- 分类头只覆盖 SSv2 的 174 类；换数据集需微调后重新转换（转换器读 `classifier.*` 与 `id2label`）
- 单 clip 形状固定（16 帧 256²）；CPU 19 线程约 5 s/clip
- 只输出标签与特征，不能生成文本

## 许可

上游权重为 **MIT**，本 GGUF 沿用同一许可；转换脚本与推理引擎为 MIT。
