# SkyVision.cpp

用 [ggml](https://github.com/ggml-org/ggml) 在本地跑**图像/视频 VQA** —— 一个多模型推理引擎：
每个视觉语言模型是一个后端（实现 `vqa::Model`），统一前端按 GGUF 里的
`general.architecture` 自动选择后端，图像与视频帧序列共用同一套输入抽象。
运行时不需要 Python、不需要 CUDA Toolkit、不依赖任何厂商运行时。

[English](README.en.md)

## 支持的后端

| 后端 | 架构标签 | 输入 | 说明 |
|---|---|---|---|
| **moondream3.1-9B-A2B** | `moondream3` | 图像 | MoE 文本 + SigLIP 视觉塔；五个官方技能（caption / query / detect / point / gaze）、Web UI、批量推理 |
| **VideoChat3-4B** | `videochat3` | 图像 + 视频 | Qwen3 文本主干（36 层 GQA + per-head qk-norm）+ VideoChat3 自研 27 层 ViT（2D RoPE + 2×2 merge）、2×2→4608 merger、multimodal projector |

## 当前状态

> moondream3.1 全部功能已就绪：文本 + 视觉 + 五个官方技能 + 嵌入式 Web UI + 批量推理
> + 流式输出，CPU / CUDA 双后端。VideoChat3-4B 后端可用的部分见下方标注。

| | |
|---|---|
| 文本 decode（24 层） | ✅ 质量对齐官方（逐张量 diff ~1e-3），贪心输出无 reserved token |
| GPU 后端（CUDA via ggml） | ✅ 单序列 ~106 tok/s，8 路批量 231 tok/s（5070 Ti Laptop，Q6_K；flash-attn + F16 KV + CUDA graph 回放） |
| CPU 后端 | ✅ 12.8 tok/s（16 线程，与 CUDA 数值一致） |
| SigLIP 视觉编码器 | ✅ 官方多 crop Lanczos 预处理 + 批量编码，~0.56s/图（10 crop），parity < 0.04 |
| 五个官方技能 | ✅ caption（short/normal/long）/ query / detect / point / gaze，模板与官方逐 token 一致 |
| Web UI（`--serve`） | ✅ 图像优先界面：拖拽/粘贴图片、三档描述、提问、检测绿框、定位十字、注视点预测；同图缓存命中 ~66ms |
| FP8 专家权重 → Q8_0 / Q6_K / Q4_K | ✅ 转换器已就绪（Q6_K ≈ 7.6GB，12GB 显卡首选） |
| 批量推理（`--batch`）/ 流式输出 | ✅ |
| VideoChat3-4B 文本 / 图像问答 | ✅ 端到端可用（`vqa-cli`，CPU 与 CUDA 均验证）；**与 PyTorch 参考实现的逐级 parity 尚未做** |
| VideoChat3-4B 视频输入 | 🔜 帧目录解码与时间戳提示词已设计，尚未接线 |

## 快速开始

### 1. 构建

唯一第三方依赖是 ggml 本体（它同时提供 gguf）：

```bash
git clone https://github.com/<you>/SkyVision.cpp
cd SkyVision.cpp
git clone --depth 1 https://github.com/ggml-org/ggml

# Windows（MSVC + Ninja）：一次构建 vqa-cli 与 moondream-cli
scripts\build-cuda.bat        # CUDA 后端（默认 sm_89=Ada，可用 CUDA_ARCH 覆盖）
scripts\build-vqa.bat         # 仅 CPU
scripts\build.bat             # 仅 CPU，改用 Visual Studio 生成器（产物在 build\Release\）
# 手工 CMake：
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release --target vqa-cli moondream-cli
```

`vqa-cli` 是按架构分派的统一前端；`moondream-cli` 是 moondream3.1 的独立二进制
（Web UI、批量推理等专属开关都在它上面）。两者共用同一套引擎代码。
CUDA 构建下引擎默认优先使用 GPU，`--cpu`（或 moondream 的 `MD_CPU=1`）可强制 CPU。

### 2. 取权重

```bash
pip install -r convert/requirements.txt

# moondream3.1-9B-A2B（官方 FP8 检查点）
hf download moondream/moondream3.1-9B-A2B --local-dir models/moondream3.1-9B-A2B

# VideoChat3-4B
hf download qqceqqq/VideoChat3-4B --local-dir models/VideoChat3-4B
```

转换器会自己找检查点：`--src <目录|文件>` → 该模型的 `$MOONDREAM_SRC` / `$VIDEOCHAT3_SRC`
→ HF 缓存（`$HF_HOME` 或 `~/.cache/huggingface/hub`）→ 仓库下的 `models/`；
找不到时会打印下载命令。详见 [模型转换](#模型转换)。

### 3. 运行

```bash
# 统一前端：后端按 GGUF 的 general.architecture 自动选择
vqa-cli --model models/VideoChat3-4B/videochat3-4b-q8_0.gguf \
        --tokenizer models/VideoChat3-4B \
        --image photo.jpg --ask "What is in this image?" --max-tokens 256

vqa-cli --model moondream3.1-q6k.gguf --tokenizer tokenizer.json \
        --vision moondream3.1-vision.gguf --region moondream3.1-region.gguf \
        --image photo.jpg --ask "What is in this image?" --n-ctx 1024

# moondream3.1 独立二进制：技能、Web UI、批量
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json \
    -v moondream3.1-vision.gguf --region moondream3.1-region.gguf \
    -i photo.jpg -p "What is in this image?"
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json \
    -v moondream3.1-vision.gguf --region moondream3.1-region.gguf --serve   # http://127.0.0.1:8080
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json --batch prompts.txt
```

`vqa-cli` 常用开关：`--video <file> --fps 2 --max-frames 32`（内置视频：ffmpeg 管道解码 + qwen_vl_utils 式采样 n = min(duration×fps, max_frames)，超限自动全片均匀重采样）、`--frames <dir>`（预抽帧目录）、`--temp/--top-k/--top-p`（采样，默认贪心）、
`--max-tokens`、`--n-ctx`、`--threads`、`--cpu`、`--dump-ids`（只跑分词并打印 token id）、
`--help` 看全部。

moondream Web UI 覆盖官方全部五个技能：图像描述（简短/标准/详细）、提问（VQA）、
目标检测（绿框）、定位（黄色十字）、注视点预测（点选起点→预测注视点）。
对同一张图连续提问时自动复用视觉编码与 KV 前缀（~1.2s → ~66ms）。
HTTP API：`POST /api/vision`，`mode=ask|caption|detect|point|gaze`
（caption 带 `len=short|normal|long`，gaze 带 `x=&y=` 起点坐标），body 为图片二进制。

## 模型转换

所有转换器都在 `convert/`，共用 `hf_source.py` 定位检查点、`_lib.py` 提供 GGUF/量化工具，
日志统一 `[convert]` 前缀，写出后会重新打开文件校验张量表。

| 脚本 | 产出 |
|---|---|
| `convert_hf_to_gguf.py` | `moondream3.1-q8_0.gguf` / `-q6k` / `-q4k`（文本主干，`--q6` / `--q4` / `--q8-dense`） |
| `convert_vision.py` | `moondream3.1-vision.gguf`（SigLIP 塔，~0.84GB） |
| `convert_region.py` | `moondream3.1-region.gguf`（detect/point/gaze 头，~30MB） |
| `convert_videochat3.py` | `videochat3-4b-<outtype>.gguf`（Qwen3 + VideoChat3 ViT，单文件，`--outtype f16\|q8_0\|q6_k\|q4_0`） |

### moondream3.1

读取官方 FP8 检查点 [`moondream/moondream3.1-9B-A2B`](https://huggingface.co/moondream/moondream3.1-9B-A2B)：

- FP8 专家张量逐位精确解码（与 PyTorch 的 e4m3 解码 bit 级一致）后重量化为 Q8_0，
  `--q6` 可选 Q6_K（约 7.6GB，12GB 显存首选）、`--q4` 可选 Q4_K
- BF16 张量转 F16
- FP8 的逐通道 scale 直接折叠进量化块

```bash
python convert/convert_hf_to_gguf.py          # → moondream3.1-q8_0.gguf（约 10GB）
python convert/convert_hf_to_gguf.py --q6     # → moondream3.1-q6k.gguf（约 7.6GB）
python convert/convert_vision.py              # → moondream3.1-vision.gguf
python convert/convert_region.py              # → moondream3.1-region.gguf
```

输出默认写当前目录（`-o` 可改，moondream 侧的 `-o` 里 `%s` 会替换成量化标签）。
Q6_K 量化器是本仓库自带的（gguf-py 至今只实现了 Q6_K 的**解码**，`quants.quantize` 会抛
`NotImplementedError`），`python scripts/test_q6k.py` 可自检块布局与量化误差。

### VideoChat3-4B

单个 GGUF 装下全部张量：文本解码器（`--outtype` 量化，默认 `q8_0`）、
27 层 ViT 与 multimodal projector（保持 F16）、学习式位置编码表（F32，供双三次插值）。
分词器不打包进 GGUF，运行时从检查点目录的 `tokenizer.json` 读取（`--tokenizer <dir>`）。

```bash
python convert/convert_videochat3.py --outtype q8_0   # → videochat3-4b-q8_0.gguf（4.83GB）
python convert/convert_videochat3.py --outtype q6_k   # → 3.92GB
python convert/convert_videochat3.py --outtype q4_0   # → 2.95GB
```

`q4_k` / `q5_k` / `q2_k` 目前不可用（上游 gguf-py 只实现了它们的解码），
`q6_k` 走本仓库的 `convert/q6k.py`。Q8_0 及以上精度更高，Q6_K/Q4_0 更省显存与带宽。

模型权重遵循各自的许可证（moondream 为
[Moondream Model License 1.0](https://moondream.ai/licenses/model/1.0)），**不在本仓库分发**。
该许可明确允许量化、转换、再分发、自托管与商用集成；仅对公众开放的通用 moondream API
转售需要单独授权——那是部署者的义务，与本引擎无关。

## 为什么做独立引擎？

moondream3.1 是 Moondream 团队推出的 90 亿参数混合专家（MoE）视觉语言模型（每 token 激活 2B）。
目前本地运行它只有一个途径：Moondream 官方的 [Photon](https://moondream.ai/p/photon) 引擎
（Python + 专有 CUDA 内核，需要 24GB+ 显存）。llama.cpp 生态对 moondream 的支持从 2025 年起
就只有一份[挂起的 feature request](https://github.com/ggml-org/llama.cpp/issues/13332)——它的架构
和 llama.cpp 支持的任何模型都不一样：

- **tau attention** —— 逐 head、逐 token 的可学习 Q/V 缩放：
  `q *= tanh(Wq·gelu(qkv)) + 1 + (sigmoid(alpha·log(pos+1)) − 0.5)`
- **并行残差 block** —— `x = x + attn(LN(x)) + mlp(LN(x))`，GPT-J 风格
- **处处带 bias 的 LayerNorm**（attention 前、FFN 前、最后的 norm），lm_head 也带 bias
- **GeGLU 变体** —— MoE 专家内是 `gelu(h) · (g + 1)`
- **dense 前缀 + MoE 主体** —— 第 0~3 层是 dense MLP，第 4~23 层才是 MoE
- **官方检查点直接发 FP8 (e4m3) 专家权重**，带逐通道 scale
- SigLIP 视觉塔：588 列 patch embedding + 多 crop 切图

与其把这些硬塞进 llama.cpp 的模型注册表，不如整条栈自己掌控：权重转换、计算图、KV cache、
采样全部自研，只为 kernel 复用 ggml——就像 [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
和 [audio.cpp](https://github.com/0xShug0/audio.cpp) 那样。VideoChat3-4B 是同一套框架下的第二个后端，
它的视觉塔、位置编码与多模态投影同样不在 ggml 的既有模型里。

## 架构速览

### moondream3.1（逆向自官方检查点与 [vikhyat/moondream](https://github.com/vikhyat/moondream)）

| | |
|---|---|
| 层数 | 24（0–3 dense，4–23 MoE） |
| dim / heads | 2048 / 32 heads，无 GQA，head_dim 64 |
| RoPE | 半旋转，仅前 32 维，theta 1.5e6 |
| MoE | 64 专家选 8，专家 inter dim 1024，router 带 bias |
| 词表 | 51200（`moondream/starmie-v1`） |
| 上下文 | 32768 |

### VideoChat3-4B

| | |
|---|---|
| 文本主干 | Qwen3：36 层，dim 2560，32 heads / 8 KV heads，head_dim 128，per-head qk-norm，RoPE theta 5e6，tied lm_head |
| 视觉塔 | 27 层 ViT，dim 1152，16 heads，patch 14，2D RoPE，2×2 merge，GELU(tanh) MLP，学习式 64×64 位置表 |
| 融合 | merger 把 4 个 patch 拼成 4608 维 → LayerNorm → 4608→4608 GELU(erf) → 2560 |
| 词表 / 上下文 | 151936（Qwen2 BPE）；文本上下文 262144，引擎 KV 默认 8192 |
| 提示词 | `<|im_start|>user\n<|vision_start|>` + N×`<|image_pad|>` + `<|vision_end|>` + 问题 |

## 性能

CPU 上 Q8_0 的 VideoChat3-4B 受内存带宽限制；换成更小的量化或 CUDA 构建提升明显
（下表为实测，37 token prompt、贪心解码，具体数值随硬件与负载浮动）：

| 模型 / 量化 | 后端 | 硬件 | prefill | decode |
|---|---|---|---|---|
| VideoChat3-4B Q8_0（4.83GB） | CPU 14 线程 | i7-14700K | ~71 tok/s | ~10.4 tok/s |
| VideoChat3-4B Q8_0（4.83GB） | CUDA | RTX 4070 SUPER | ~401 tok/s | ~49.8 tok/s |

每次推理都会在 stderr 打印一行 `[perf] prefill ... | decode ...`，便于自行核对。
moondream3.1 的数字见[当前状态](#当前状态)（官方标注来自 5070 Ti Laptop）。

## 仓库结构

```
src/
  core/                共享层：vqa::Model 接口、mmap/UTF-8/JSON 工具、字节级 BPE 分词器、
                       图像与视频帧解码（stb_image 实现所在）、按架构发现后端
  models/moondream/    moondream3.1 后端（引擎 + 视觉塔 + 多 crop 预处理 + region 头 + 服务端/批量）
  models/videochat3/   VideoChat3-4B 后端（视觉预处理 + 计算图）
  cli/                 统一前端 vqa-cli
convert/               HF 检查点 → GGUF 转换器
scripts/               构建脚本、parity 与自检脚本
vendor/stb/            自带的单头图像解码器
ggml/                  唯一第三方依赖（clone 得到，不入库）
```

## 开发

- **parity 脚本**：`scripts/ref_videochat3.py` 用 transformers 跑官方实现并导出中间张量，
  `scripts/compare_vc3.py` 与 C++ 侧的 dump 逐级比对（预处理 / patch 嵌入 / 每层输出 / logits）。
- **调试开关**：`VC3_DUMP_DIR=<dir>` 导出中间张量，`VC3_MAX_LAYERS=N` 截断层数做分层对照，
  `VC3_NO_FUSION=1` 退回未融合的 SwiGLU，`--dump-ids` 只跑分词。
- **改分词器数据**：`scripts/gen_unicode_ranges.py` 重新生成 `src/core/unicode_ranges.inc`。

## License

MIT，见 [LICENSE](LICENSE)。ggml 为 MIT（ggml-org）。模型权重与 tokenizer 遵循其各自条款，单独下载。
