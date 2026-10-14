# moondream.cpp

用独立的 [ggml](https://github.com/ggml-org/ggml) 推理引擎在本地运行 **moondream3.1-9B-A2B**
—— 不需要 Python、不需要 CUDA Toolkit、不依赖任何厂商运行时。

[English](README.en.md)

moondream3.1 是 Moondream 团队推出的 90 亿参数混合专家（MoE）视觉语言模型（每 token 激活 2B）。
目前本地运行它只有一个途径：Moondream 官方的 [Photon](https://moondream.ai/p/photon) 引擎
（Python + 专有 CUDA 内核，需要 24GB+ 显存）。llama.cpp 生态对 moondream 的支持从 2025 年起
就只有一份[挂起的 feature request](https://github.com/ggml-org/llama.cpp/issues/13332)——
所以我们自己动手，按 [whisper.cpp](https://github.com/ggml-org/whisper.cpp) 和
[audio.cpp](https://github.com/amusecule/audio.cpp) 的同款模式：独立仓库 + vendored ggml + 自有模型层。

## 当前状态

> **开发中。** 文本推理已端到端跑通：生成质量与官方实现对齐（层 0 逐张量数值校验 ~1e-3），
> CUDA 后端 61.6 tok/s（5070 Ti Laptop，贪心）。SigLIP 视觉编码器已跑通（parity < 0.04）。

| | |
|---|---|
| 文本 decode（24 层） | ✅ 已跑通，质量对齐（vs 官方 torch 前向逐张量 diff ~1e-3） |
| FP8 专家权重 → Q8_0 / Q4_K GGUF | ✅ 转换器已就绪 |
| tau attention、MoE（64 选 8） | ✅ 已实现 |
| 对照官方 Photon 的质量对齐 | ✅ 完成（2026-10-14） |
| GPU 后端（CUDA via ggml） | ✅ 完成：CUDA 61.6 tok/s / CPU 12.8 tok/s（flash-attn + F16 KV） |
| SigLIP 视觉编码器 | ✅ 已跑通（27 层 + 投影，与 torch parity < 0.04；`moondream3.1-vision.gguf` + `moondream-vision-test`） |
| detect / point 头 | ✅ 完成：`--detect` / `--point` 输出框/坐标 |
| segment 头 | ➖ 官方模型不提供（region 只有 coordinate/size 头） |

## 为什么做独立引擎？

moondream3.1 的架构和 llama.cpp 支持的任何模型都不一样：

- **tau attention** —— 逐 head、逐 token 的可学习 Q/V 缩放：
  `q *= tanh(Wq·gelu(qkv)) + 1 + (sigmoid(alpha·log(pos+1)) − 0.5)`
- **并行残差 block** —— `x = x + attn(LN(x)) + mlp(LN(x))`，GPT-J 风格
- **处处带 bias 的 LayerNorm**（attention 前、FFN 前、最后的 norm），lm_head 也带 bias
- **GeGLU 变体** —— MoE 专家内是 `gelu(h) · (g + 1)`
- **dense 前缀 + MoE 主体** —— 第 0~3 层是 dense MLP，第 4~23 层才是 MoE
- **官方检查点直接发 FP8 (e4m3) 专家权重**，带逐通道 scale
- SigLIP 视觉塔：588 列 patch embedding + 多 crop 切图

与其把这些硬塞进 llama.cpp 的模型注册表，不如整条栈自己掌控：权重转换、计算图、KV cache、
采样全部自研，只为 kernel 复用 ggml——就像 whisper.cpp 对音频做的那样。

## 权重转换

转换器（`convert/`）读取官方 FP8 检查点
[`moondream/moondream3.1-9B-A2B`](https://huggingface.co/moondream/moondream3.1-9B-A2B)，
产出标准 GGUF：

- FP8 专家张量逐位精确解码（与 PyTorch 的 e4m3 解码 bit 级一致）后重量化为 Q8_0，
  `--q4` 可选 Q4_K
- BF16 张量转 F16
- FP8 的逐通道 scale 直接折叠进量化块

```bash
pip install gguf safetensors torch
huggingface-cli download moondream/moondream3.1-9B-A2B
python convert/convert_hf_to_gguf.py            # → moondream3.1-q8_0.gguf（约 10GB）
```

模型权重遵循单独的
[Moondream Model License 1.0](https://moondream.ai/licenses/model/1.0)，
**不在本仓库分发**。该许可明确允许量化、转换、再分发、自托管与商用集成
（官方 [license matrix](https://moondream.ai/licenses/model/1.0)）；
仅对公众开放的通用 moondream API 转售需要单独授权——那是部署者的义务，
与本引擎无关。

## 构建

目前以 Windows（MSVC）为主；Linux 走 CMake，脚本稍作调整即可。

```bash
git clone https://github.com/<you>/moondream.cpp
cd moondream.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp   # 提供 ggml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release --target moondream-cli
```

## 运行

```bash
# tokenizer（byte-level BPE，来自 moondream/starmie-v1）
curl -L https://huggingface.co/moondream/starmie-v1/resolve/main/tokenizer.json -o tokenizer.json

build/Release/moondream-cli.exe -m moondream3.1-q8_0.gguf -t tokenizer.json -p "hi" -n 16
```

## 架构速览

（逆向自官方检查点与开源实现 [vikhyat/moondream](https://github.com/vikhyat/moondream)）


| | |
|---|---|
| 层数 | 24（0–3 dense，4–23 MoE） |
| dim / heads | 2048 / 32 heads，无 GQA，head_dim 64 |
| RoPE | 半旋转，仅前 32 维，theta 1.5e6 |
| MoE | 64 专家选 8，专家 inter dim 1024，router 带 bias |
| 词表 | 51200（`moondream/starmie-v1`） |
| 上下文 | 32768 |

## License

MIT，见 [LICENSE](LICENSE)。ggml 为 MIT（ggml-org）。模型权重与 tokenizer 遵循其各自条款，单独下载。
