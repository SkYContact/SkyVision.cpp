---
license: nvidia-open-model-license
base_model:
- nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
language:
- en
tags:
- vision
- audio
- gguf
- nemotron
- mamba2
- hybrid
- moe
- radio
- parakeet
- skyvision.cpp
---

# Nemotron-3-Nano-Omni-30B-A3B GGUF

[nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)
的社区 GGUF 量化版，由自研推理引擎
[SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp)（纯 ggml，无 llama.cpp 依赖）转换，
并与官方 transformers 5.21 实现**逐级对拍**验证（文本贪心 argmax 100% 一致、音频转写逐字一致）。

> **仅供 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 使用。** GGUF 里
> `general.architecture = nemotronomni`：52 层 Mamba2/Attention/MoE 混合主干（NoPE attention）+
> C-RADIOv4-H 视觉塔 + Parakeet 音频塔。**llama.cpp / ollama / LM Studio 等通用 GGUF 运行时
> 无法加载本仓库文件。**

## 模型结构（与官方一致）

| 部分 | 规格 |
|---|---|
| 文本主干 | `nemotronomni`：52 层混合（pattern `MEMEM*…`）= 23 Mamba2 + 6 attention + 23 MoE；d=2688、32q/2kv GQA、head 128、**NoPE**、RMSNorm 1e-5、词表 131072（untied） |
| Mamba2 | 64 头 × 64 维、n_groups 8、d_state 128、conv k=4、chunk 128；dt 走 inverse-softplus 预补偿把 HF `clamp(min=1e-3)` 语义精确搬进 `ggml_ssm_scan` |
| MoE | 128 专家选 6：sigmoid ×2.5 路由（e_score_correction_bias F32，选择分带 bias、专家权重取无 bias sigmoid）、relu² 非门控专家（中间 1856）+ 1 个 3712 共享专家 |
| 视觉塔 | C-RADIOv4-H：32 层 ViT d=1280、16 头、10 个 register token 前置后整块丢弃、学出的 128×128 位置表按内容区插值、T=2 tubelet 视频嵌入（1280×1536） |
| 连接器 | pixel_shuffle 0.5 + mlp1：RMSNorm(5120) → 20480 → relu² → 2688 |
| 音频塔 | Parakeet/FastConformer：128 mel@16k、3 级 stride-2 卷积 8× 下采样、24 层 Conformer（d=1024、相对位置注意力、BatchNorm 推理语义在引擎侧折叠进卷积权重）→ RMSNorm + 2 层 MLP → 2688 |
| 上下文 | 256k（eos=11 `<\|im_end\|>`，generation_config 还有 2；bos=1、pad=0） |
| 输出 | **纯文本**（无 talker，不生成语音）；英文 only；thinking 默认开 |

## 量化档位

本模型行宽 2688/1856/3712 都不是 256 的倍数，k-quant（%256）只能覆盖 ~5% 的字节，
**没有"纯 Q4_K_M / Q6_K"档**；两档都是逐张量混合：

| 文件 | 大小 | 逐张量类型 |
|---|---|---|
| `nemotronomni-q8_0.gguf` | 35.17 GB | 全部 %32 matmul 操作数 Q8_0（全兼容基准档）；1 维 norm/bias/router 全程 F32 |
| `nemotronomni-q6k_mixed.gguf` | 23.29 GB | token_embd/lm_head Q8_0 + 专家堆叠（55.6GB 主体）Q5_0 + %256 宽（视觉/音频塔、ssm_out、attn_o 等）Q6_K + 其余 %32 宽 Q5_0；4-tap ssm conv、4D 音频子采样卷积等宿主直读尾巴保持 F16 |
| `tokenizer.json` 等 | ~19 MB | 官方检查点分词器（byte-level BPE，pixtral split 正则），本仓库一并附带 |

Q6_K 编码器是 ggml 参考算法（quantize_row_q6_K_ref）的 numpy 移植，与引擎反量化**位级一致**；
Q5_0/Q8_0 走 gguf-py。转换后自检（`convert/check_nemotronomni.py`）：1481 张量键集/形状/dtype/
offset 链全对，采样张量对 safetensors **byte-exact**。

## 对拍结果（oracle = transformers 5.21 官方栈，BF16）

| 项 | 结果 |
|---|---|
| 文本 prompt A/B 贪心 argmax | 30/30 + 3/3 全对（F16 基线） |
| 音频转写 | 与官方逐字一致 |
| RoPE 判决 | NoPE（partial84 媒体路径 step0 即 cos 0.85、丢失转写内容） |
| EC 专家驻留 XCHK | logits maxdiff 0.00000、104/104 层输出位级一致 |

量化档验证（贪心流 vs F16 基线、`NEMO_XCHK` 漂移）：

| 档 | prompt A argmax | prompt B argmax | XCHK maxdiff |
|---|---|---|---|
| q8_0 | 3/3 | 30/30 | 0.00000 |
| q6k_mixed | 3/3 | 30/30 | 0.00000 |

量化档注意：文本贪心与 F16 基线逐 token 一致；**音频 ASR 在量化档下会漂移**
（首几个 token 后分叉，Q8_0/Q6K_mixed 同）——逐字转写请用 F16 档或官方栈。

## 性能参考（Release 构建；F16 66GB 对 31GB 内存全靠 mmap 缺页，墙 ~350MB/s）

| | F16 | q8_0 | q6k_mixed |
|---|---|---|---|
| CPU 16 线程 decode | 0.33 tok/s | 0.95 tok/s | 3.39 tok/s |
| CUDA EC split（5070 Ti Laptop 12GB） | 0.14-0.19 tok/s | 2.3 tok/s | 3.1 tok/s |

CUDA 形态 = EC split（`NEMOTRONOMNI_GPU=1 NEMOTRONOMNI_EXPERTCACHE=1`）：稠密主干+塔+KV 上卡，
专家按步从 host mmap 换入有界 VRAM slot——瓶颈是每步 ~0.7-1GB 的专家换入带宽，量化由此从
"能跑"变成"可用"。

## 使用

```bash
modelscope download --model SkyContact/Nemotron3-Nano-Omni-30B-A3B-GGUF --local-dir models/Nemotron3-Omni-GGUF
vqa-cli --model models/Nemotron3-Omni-GGUF/nemotronomni-q8_0.gguf \
        --tokenizer models/Nemotron3-Omni-GGUF \
        --image photo.jpg --ask "What is in this picture?"
vqa-cli --model ... --tokenizer ... --audio clip.wav --ask "Transcribe this audio."
```

| 文件 | 大小 | sha256（前 8…后 6） |
|---|---|---|
| `nemotronomni-q8_0.gguf` | 35.17 GB | `916044dc…eeb141` |
| `nemotronomni-q6k_mixed.gguf` | 23.29 GB | `8d7ea1e9…ccfa8` |

## 校验

下载后建议核对 sha256（见仓库 `SHA256SUMS`）：

```bash
sha256sum -c SHA256SUMS
```

## 致谢与许可

模型权重来自 NVIDIA（NVIDIA Open Model License），本仓库只做格式转换与量化，
能力/局限与原模型一致；转换与推理由 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 完成。
