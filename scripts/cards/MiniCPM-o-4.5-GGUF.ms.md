---
license: other
license_name: minicpm-model-license
base_model: openbmb/MiniCPM-o-4_5
base_model_relation: quantized
library_name: ggml
pipeline_tag: image-text-to-text
language:
- en
- zh
tags:
- vision
- audio
- speech
- gguf
- minicpm
- omni
- qwen3
- siglip
- cosyvoice
- skyvision.cpp
---

# MiniCPM-o 4.5 GGUF

[openbmb/MiniCPM-o-4_5](https://modelscope.cn/models/openbmb/MiniCPM-o-4_5) 的社区 GGUF 量化版，
由自研推理引擎 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp)（纯 ggml，无 llama.cpp
依赖）转换，并与官方 transformers 4.51 实现**逐级对拍**验证。

> **仅供 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 使用。** GGUF 里
> `general.architecture = minicpmo`（声码器兄弟文件 `general.architecture = token2wav`）：
> Qwen3-8B 稠密主干 + SigLIP NaFiT 视觉塔 + Whisper-medium 音频塔 + Llama-768 TTS 副塔。
> **llama.cpp / ollama / LM Studio 等通用 GGUF 运行时无法加载本仓库文件。**

## 模型结构（与官方一致）

| 部分 | 规格 |
|---|---|
| 文本主干 | Qwen3-8B 稠密：36 层 d=4096、GQA 32:8、head_dim 128、theta 1e6、q/k 逐 head RMSNorm、untied lm_head、词表 151748、ctx 40960 |
| 视觉塔 | SigLIP-SO400M NaFiT：27 层 d=1152 @448px patch14，学出的 70×70 位置表按图**动态多瓦片切片**（slice_image） |
| 连接器 | 64-query Perceiver 重采样器（视觉 → LLM 投影） |
| 音频塔 | Whisper-medium：24 层 d=1024、80 mel、1s 块对角注意力掩码、AvgPool1d(5)；whisper 的 k_proj 无 bias（q/v/out 有） |
| TTS 副塔 | Llama-768×20L，6562 码本 speech token，weight_norm head 重组合；**必须官方采样参数**（temp 0.8 / top_p 0.85 / top_k 25 / rep_pen 1.05，贪心会码塌缩） |
| 声码器 | CosyVoice2 token2wav（兄弟 GGUF `minicpmo-vocoder.gguf`，arch `token2wav`，引擎在主模型同目录自动发现，`MINICPMO_VOCODER` 可改路径） |
| 词表 / 上下文 | 151748（Qwen3 底座；152064 是 TTS 副塔文本表，勿混淆）/ 40960 |

## 文件说明

| 文件 | 大小 | 说明 |
|---|---|---|
| `minicpmo45-q8_0.gguf` | 10.09 GB | 全兼容基准档：686 个 %32 宽 matmul 操作数 Q8_0；视觉 fc2（4304 宽，连 %32 都不满足）×27 张保 F16 |
| `minicpmo45-q6k_mixed.gguf` | 8.19 GB | 混合档：550 个 %256 宽 Q6_K；1152 系×136 + fc2×27 保 F16 |
| `minicpmo-vocoder.gguf` | 377 MB | CosyVoice2 token2wav 声码器（F32 conv 权重），`--speak` 必需 |
| `tokenizer.json` / `vocab.json` / `merges.txt` 等 | ~26 MB | 官方检查点分词器 + generation_config（GGUF 不含 tokenizer，引擎直读） |

量化档自检（`convert/check_minicpmo45.py`）：1412 张量键数/差集/byte-exact 抽样全过
（1413 消费 + 1 记录在案跳过 = 1414 HF 张量）。norms/biases/位置表/查询向量恒 F32。

## 对拍结果（oracle = transformers 4.51 官方栈 BF16，numpy 独立重算判卷）

| 门 | 结果 |
|---|---|
| 文本（"What is the capital of France?"） | 贪心输出 **"Paris"** 与官方 chat() 一致（prefill 末位 top1 p=0.9999） |
| 视觉（3 瓦片） | 重采样器输出 cos_mean **0.99994**，语义正确 |
| 音频（jsut01 wav） | 投影 cos_mean **0.9999980**（max_abs_diff 0.057） |
| TTS（seed 0 官方采样） | 102 个语音码 → 声码器 wav 4.08s @24kHz |
| 声码器同噪声门 | mel cos **0.999999986** / wav SNR **12.6dB**（AR 采样 RNG 对齐不可行，parity 口径=同噪声门） |

## 量化档引擎验证（vs F16 基线，同 prompt 贪心）

| 门 | q8_0 | q6k_mixed |
|---|---|---|
| 文本 prefill argmax | 25/26（=F16 基线），gen "Paris"+im_end 全对 | 23/26，gen 仍全对 |
| 视觉重采样器 cos | 0.999877 | 0.999961 |
| 音频投影 cos | 0.999654 | 0.998163 |
| TTS 语音码 | AR 采样从码 1 起分叉（逐 step logit cos 0.985-0.998），语音正常产出 | 同左 |

## 性能参考（Release 构建；CPU 16 线程 / CUDA 5070 Ti Laptop 12GB 全卡）

| | q8_0 | q6k_mixed |
|---|---|---|
| CPU decode | 7.25 tok/s | 9.17 tok/s |
| CUDA decode（全卡） | 62.5 tok/s | 52.6 tok/s |

## 使用

```bash
modelscope download --model SkyContact/MiniCPM-o-4.5-GGUF --local-dir models/MiniCPM-o-GGUF
vqa-cli --model models/MiniCPM-o-GGUF/minicpmo45-q8_0.gguf \
        --tokenizer models/MiniCPM-o-GGUF --image photo.jpg --ask "What is in this picture?"
vqa-cli --model ... --tokenizer ... --audio clip.wav --ask "Transcribe this audio."
vqa-cli --model ... --tokenizer ... --ask "用一句话介绍你自己" --speak reply.wav
```

声码器 GGUF 必须**与主模型同目录**（引擎自动发现；`MINICPMO_VOCODER` 可改路径）。

## 校验

下载后建议核对 sha256（见仓库 `SHA256SUMS`）：

```bash
sha256sum -c SHA256SUMS
```

## 致谢与许可

模型权重来自 OpenBMB（MiniCPM 自研协议，商用需登记），本仓库只做格式转换与量化，
能力/局限与原模型一致；转换与推理由 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 完成。
