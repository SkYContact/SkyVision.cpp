---
license: apache-2.0
base_model: openbmb/MiniCPM-V-4.7-1B
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- vision
- video
- gguf
- minicpm
- siglip
- canvas-mrope
- hybrid-attention
- gated-delta-net
- skyvision.cpp
---

# MiniCPM-V 4.7 1B GGUF

[openbmb/MiniCPM-V-4.7-1B](https://modelscope.cn/models/openbmb/MiniCPM-V-4.7-1B) 的社区 GGUF 量化版，
由自研推理引擎 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp)（纯 ggml，无 llama.cpp
依赖）转换，并与官方 transformers 实现**逐级对拍**验证。

> **仅供 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 使用。** GGUF 里
> `general.architecture = minicpmv47`，文本主干是 Qwen3.5 混合线性注意力（gated-delta 状态 + 全注意力
> 交错），视觉侧是 SigLIP canvas 塔 + 窗口注意力 merger，位置编码是 canvas M-RoPE。
> **llama.cpp / ollama / LM Studio 等通用 GGUF 运行时无法加载本仓库文件。**

## 模型结构（与官方一致）

| 部分 | 规格 |
|---|---|
| 文本主干 | `qwen3_5_text`：24 层（每 4 层 = 3 层 gated-delta 线性注意力 + 1 层全注意力），d=1024，8 头 / 2 KV 头，head_dim 256，IMROPE 11/11/10（64/256 维部分旋转），q\|gate 交织投影，untied lm_head，词表 248094 |
| 视觉塔 | SigLIP：27 层，d=1152，16 头，patch 14，GELU-tanh，**无 rope**（learned 70×70 位置表按最近桶取） |
| 合并 | 第 6 层后插入窗口注意力 merger（2×2 窗口内注意力 + 4×d 展平 MLP），再过 2×2 投影 MLP → **16× 压缩**；`4x` 模式跳过前者 |
| 位置 | **canvas M-RoPE**：视觉 token 铺 2-D 画布，`<image>`/`</image>`/`<slice>`/`</slice>`/换行钉在画布角点，文本计数从 `max(h,w)+1` 续 |
| 切片 | 官方 slice_mode：长边 >448 起切，最多 9 块明细 crop + 一张缩略图 |
| 视频 | 逐帧独立过塔，帧级 canvas 按时间顺序推进 |

## 量化档位

| 文件 | 大小 | 说明 |
|---|---|---|
| `minicpm-v-4.7-1b-f16.gguf` | 3.12 GB | 视觉塔 + 两个 merger 全 F16，文本主干 F16 |
| `tokenizer.json` | 约 18 MB | 官方检查点分词器（BPE，Qwen2 正则），本仓库一并附带 |
| `config.json` / `preprocessor_config.json` | 几 KB | 上游配置，便于核对字段（`insert_layer_id`、`downsample_mode`、`mrope_mode` 等） |

## 对拍结果（对比 transformers 5.20 官方实现）

单图（512×384 → 63 视觉 token）：

| 阶段 | corr | maxdiff |
|---|---|---|
| 预处理 patch（含 0.5/0.5 归一化） | 0.999999 | 0.09 |
| patch embed（conv+bias+位置表） | 0.999997 | 0.13 |
| 窗口注意力 merger | 0.999908 | 0.31 |
| 视觉塔输出（27 层 + post-LN） | 0.999981 | 0.28 |
| 投影特征（送 LLM） | 0.999839 | 0.034 |
| 末 token logits | 0.999820 | 0.33 |

**贪婪 24 token 与官方逐字一致**：`The image displays three simple geometric shapes on a light gray background: - A **red circle** on the left.`

切片（1400×900 → 缩略图 + 6 切片，444 视觉 token）：canvas M-RoPE 位置 id **逐元素相同**，
7 个 crop 的 patch ≤2 灰阶，合并特征 corr 0.999458，prefill logits corr 0.999518（top-8 相同）。

视频（6 帧，420 token）：帧级 canvas 位置 **逐元素相同**，按帧顺序叙述正确。

## 速度（本机实测）

| 设备 | 视觉（7 crop / 444 token） | prefill 484 token | decode |
|---|---|---|---|
| CPU 19 线程 | 7704 ms | 168 tok/s | 27 tok/s |
| RTX 4070 SUPER 12GB | **244 ms** | **15.3k tok/s** | **185 tok/s** |

## 使用

```bash
git clone https://github.com/SkYContact/SkyVision.cpp
cd SkyVision.cpp && git clone --depth 1 https://github.com/ggml-org/ggml
scripts/build-cuda.bat            # 或 scripts/build-vqa.bat（CPU）

modelscope download --model SkyContact/MiniCPM-V-4.7-1B-GGUF \
    --local_dir models/MiniCPM-V-4.7-1B-GGUF

# 图像（自动按官方 slice_mode 切片）
vqa-cli --gpu --model models/MiniCPM-V-4.7-1B-GGUF/minicpm-v-4.7-1b-f16.gguf \
        --tokenizer models/MiniCPM-V-4.7-1B-GGUF --image photo.jpg \
        --ask "Read all text in the image verbatim."

# 视频（帧目录或视频文件）
vqa-cli --gpu --model ... --tokenizer ... --frames frames/ --ask "What moves, and in which direction?"
vqa-cli --gpu --model ... --tokenizer ... --video clip.mp4 --fps 2 --ask "..."
```

运行时开关：

| 变量 | 作用 |
|---|---|
| `MINICPM_DOWNSAMPLE=4x` | 4× 保留视觉 token（跳过窗口 merger），适合 OCR/小目标 |
| `MINICPM_NO_SLICES=1` | 关闭切片，只发缩略图（快，但小字会丢） |
| `MINICPM_SCALE_RES=<n>` | 缩略图/切片长边目标，默认 448 |

## 自己转换

```bash
python convert/convert_minicpmv47.py --src models/MiniCPM-V-4.7-1B --outtype f16
# 量化：--outtype q8_0 / q6_k
```

转换脚本把 HF 权重映射到 GGUF 的 `blk.N.*`（文本，沿用 SkyVision 的 qwen3_5 命名）、
`v.*`（SigLIP 塔）、`vm.*`（窗口 merger）、`mm.*`（投影 MLP）、`minicpm.*`（canvas 常量）几组键；
`output.weight` 是 untied lm_head。

## 致谢与许可

模型权重来自 openbmb（Apache-2.0），本仓库只做格式转换与量化，能力/局限与原模型一致。
