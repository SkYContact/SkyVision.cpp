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
| **Qwen3.5 系（含 Marlin-2B）** | `marlin` | 图像 + 视频 | Qwen3.5 混合主干（18 层 gated-delta 线性注意力 + 6 层 full attention，interleaved MRoPE）+ Qwen3-VL 式视觉塔（学习式绝对位置嵌入 + 2D RoPE + 逐时间步注意力）。官方 `Qwen/Qwen3.5-*` 检查点直接转换即用（结构同构、逐级 parity 已验证）；Marlin-2B 为其视频字幕特化版，额外带官方训练技能 caption（带时间戳事件）/ find（时序定位）
| **MOSS-VL-Realtime（11B）** | `moss_vl` | 图像 + 视频 | OpenMOSS cross-attention 架构：Qwen3 文本主干（36 self 层 + 12 纯 cross-attn 层，tanh 双门控）+ SigLIP 塔（deepstack 8/16/24 + merger）+ 每帧可学习 separator。视觉 KV 走独立 cross 注意力，interleaved MRoPE（视觉三轴平面位置）。支持多轮会话（`?session=`） |
| **VideoMAE（base，video tower）** | `videomae` | 视频（图像=静帧） | 纯视觉塔，没有文本侧：Conv3d tubelet patch 嵌入 + 固定 sin-cos 位置表（**无 cls token**）+ 12 层全时空注意力，一次前向出分类标签（`*-finetuned-*` 检查点，如 Kinetics-400）或 pooled 嵌入（`MCG-NJU/videomae-base` 是预训练版，无分类头）。单 clip 1568 token，单图；`--mode embed` 直接吐 768 维特征 |

## 当前状态

> moondream3.1 全部功能已就绪：文本 + 视觉 + 五个官方技能 + 嵌入式 Web UI + 批量推理
> + 流式输出，CPU / CUDA 双后端。VideoChat3-4B 后端可用的部分见下方标注。

| | |
|---|---|
| 文本 decode（24 层） | ✅ 质量对齐官方（逐张量 diff ~1e-3），贪心输出无 reserved token |
| GPU 后端（CUDA via ggml） | ✅ 单序列 ~106 tok/s，8 路批量 ~241 tok/s（5070 Ti Laptop，Q6_K；flash-attn + F16 KV + CUDA graph 回放） |
| CPU 后端 | ✅ 12.8 tok/s（16 线程，与 CUDA 数值一致） |
| SigLIP 视觉编码器 | ✅ 官方多 crop Lanczos 预处理 + 批量编码，~0.56s/图（10 crop），parity < 0.04 |
| 五个官方技能 | ✅ caption（short/normal/long）/ query / detect / point / gaze，模板与官方逐 token 一致 |
| Web UI（`--serve`，可绑 0.0.0.0 部署局域网） | ✅ 移动端友好：拖拽/选图、NDJSON 流式逐字输出（UTF-8 边界安全）、视觉编码逐 clip 进度、detect 绿框 / point 十字 / gaze；同图同视频追问走两级缓存（上传解码 + 视觉特征），二次提问 ~26ms |
| moondream 思考模式 | ✅ `mode=ask-think`：官方 reasoning 协议（thinking_id → grounded 坐标 token → answer），推理段与回答流式分段输出 |
| FP8 专家权重 → Q8_0 / Q6_K / Q4_K | ✅ 转换器已就绪（Q6_K ≈ 7.6GB，12GB 显卡首选） |
| 批量推理（`--batch`）/ 流式输出 | ✅ prefill 注意力按 prompt 长度开窗（不再扫满 n_ctx）；8 路批量 decode ~241 tok/s |
| VideoChat3-4B 图像问答 | ✅ 端到端 + **逐级 parity 已验证**：patch 嵌入 rel 4e-4、27 层 ViT + merger + projector rel 0.026（F16 权重噪声级），贪心输出与官方 bf16 逐字一致 |
| VideoChat3-4B 视频输入 | ✅ 内置 `--video`（ffmpeg 管道，`FFMPEG=<path>` 可指定）：抽帧 + 时间戳提示词 + 4 帧时序 clip 联合注意力；视觉塔逐 clip 分批执行（激活峰值降至单 clip），同上下文预算内自动钳制超大图/视频 |
| Marlin-2B 图像/视频问答 | ✅ 端到端 + **逐级 parity 已验证**（官方 transformers 5.20 做 oracle）：pixel_values 与引擎 patchify 逐位一致、pos 表 2.7e-8、patch 卷积 2.6e-3、block0 5.3e-3；整塔漂移 6.6e-2 **优于官方 bf16 自身的 1.1e-1**；末 token logits 1.5e-2 < 官方 bf16 2.8e-2；文本纯输入 gated-delta 层 1.4e-2 |
| Marlin-2B 视频结构化技能 | ✅ `--mode caption`（Scene + 带秒级时间戳的 Events 列表）/ `--mode find`（自然语言事件 → `From X.X to Y.X.`），训练对齐的官方提示词 |
| 视觉预处理 token 预算 | ✅ 图片/视频的视觉 token 上限 = n_ctx − max_tokens − 文本开销，编码前钳制分辨率或提前报错（不再白等几秒后 context overflow） |

| MOSS-VL-Realtime 图像/视频问答 | ✅ 端到端 + 官方栈对拍：图像 prefill token 流 24/24、视频 78/78 逐位一致；layer2 cross-attn 中间量 corr ≥ 0.999；KV cache 回读 vs 官方 roped K corr 0.9997。CUDA（5070 Ti Laptop，Q4_K）：视觉 213ms/图、decode 67 tok/s；CPU 16 线程 decode 8.9 tok/s |
| MOSS-VL 多轮会话 | ✅ `--serve` 的 `/ask?session=<id>`（`reset=1` 清空）：server 维护最近 12 轮文本历史，引擎 prompt 重建（媒体缓存命中时视觉塔免重跑） |
| MOSS-VL 实时流会话 | ✅ `/ask?session=<id>&rt=1`：官方 real_time_generate 的引擎侧实现——新帧/新问题增量追加到活跌 KV（文本 KV 续写 + cross-KV 行追加，不重建全上下文），位置按官方增量 MRoPE 公式；`<|silence|>` 作为闲置信号终止本轮。已验证：跨轮记忆（两张图分别在后续追问中正确引用） |
| MOSS-VL 常驻推理流 | ✅ `rt/start?session=&src=&interval=`（src=文件/URL/`video=`摄像头）：ffmpeg 采集线程 + 常驻推理循环，模型按轮观察新帧并**主动发言**（`<|silence|>`=本轮无话可说）；`rt/say?q=` 流内提问（独立文本轮）、`rt/events?since=` 轮询 NDJSON 事件、`rt/stop` 停止 |
| VideoMAE 视频分类 / 特征 | ✅ 一个 GGUF 装全部权重（94M 参数，f16 约 180MB）：16 帧均匀采样 → 中心裁剪 224 → 一次全时空注意力前向 → top-k 标签；无 tokenizer、无 KV cache。**逐级对拍已验证**（numpy 从 GGUF 权重复算整条前向）：12 层全长相对差 ~1e-3、patch 嵌入 3e-3、block0 注意力 3e-4、Kinetics-400 的 400 个类别概率差 <1e-4。预训练版（无分类头）输出 pooled 嵌入，`--mode embed` 直出 768 维 JSON。CPU 单 clip（1568 token×12 层）约 0.85 s（q8_0 权重 0.59 s）|
## 快速开始

### 1. 构建

唯一第三方依赖是 ggml 本体（它同时提供 gguf）：

```bash
git clone https://github.com/SkYContact/SkyVision.cpp
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

预转好的 GGUF 免转换直接下：[魔搭 ModelScope](https://modelscope.cn/collections/SkyContact/SkyVisioncpp) · [Hugging Face](https://huggingface.co/collections/SkyContact/skyvisioncpp)

想自己转，或者要别的量化档位：

```bash
pip install -r convert/requirements.txt

# moondream3.1-9B-A2B（官方 FP8 检查点）
hf download moondream/moondream3.1-9B-A2B --local-dir models/moondream3.1-9B-A2B

# VideoChat3-4B
hf download MCG-NJU/VideoChat3-4B --local-dir models/VideoChat3-4B

# MOSS-VL-Realtime（图像/视频 + 多轮会话）
vqa-cli --model Moss-VL-Realtime-q4k.gguf --image photo.jpg --ask "What is in this image?"
vqa-cli --model Moss-VL-Realtime-q4k.gguf --video clip.mp4 --max-frames 8 --ask "What happens?"   # 视频：逐帧时间戳提示词
# 服务端多轮会话：POST /api/model/Moss-VL-Realtime-q4k/ask?q=...&session=abc（reset=1 清空）

# Marlin-2B（魔搭镜像，无需登录）
modelscope download NemoStation/Marlin-2B --local_dir models/Marlin-2B

# Qwen3.5-2B（官方，通用图文/视频问答）
modelscope download Qwen/Qwen3.5-2B --local_dir models/Qwen3.5-2B

# VideoMAE：预训练版（只有嵌入）或带 Kinetics-400 分类头的 fine-tune
hf download MCG-NJU/videomae-base --local-dir models/videomae-base
hf download MCG-NJU/videomae-base-finetuned-kinetics --local-dir models/videomae-base-ft
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

# Marlin-2B：视频结构化信息提取
vqa-cli --model marlin-2b-q8_0.gguf --tokenizer models/Marlin-2B         --video clip.mp4 --fps 2 --max-frames 16         --ask x --mode caption --max-tokens 512          # Scene: + 带时间戳的事件列表
vqa-cli --model marlin-2b-q8_0.gguf --tokenizer models/Marlin-2B         --video clip.mp4 --fps 2         --ask "a person enters the room" --mode find     # From X.X to Y.Y.

vqa-cli --model moondream3.1-q6k.gguf --tokenizer tokenizer.json \
        --vision moondream3.1-vision.gguf --region moondream3.1-region.gguf \
        --image photo.jpg --ask "What is in this image?" --n-ctx 1024

# VideoMAE：视频 → 动作标签（分类头版）/ 768 维嵌入（预训练版）
vqa-cli --model videomae-base-ft.gguf --video clip.mp4 --ask x --mode 5
vqa-cli --model videomae-base.gguf    --video clip.mp4 --ask x --mode embed

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
| `convert_marlin.py` | `marlin-2b-<outtype>.gguf`（Qwen3.5 混合主干 + Qwen3-VL 视觉塔，单文件，`--outtype f16\|q8_0\|q6_k\|q4_0`） |
| `convert_videomae.py` | `videomae-<outtype>.gguf`（VideoMAE 视频塔，单文件，`--outtype f16\|q8_0\|q4_0`；分类头与 `id2label` 一起写进 GGUF） |

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
python convert/convert_marlin.py --src models/Qwen3.5-2B   # → qwen3.5-2b-q8_0.gguf（2.67GB，Qwen3.5/Marlin 通用）
python convert/convert_videochat3.py --outtype q6_k   # → 3.92GB
python convert/convert_videochat3.py --outtype q4_0   # → 2.95GB
```

`q4_k` / `q5_k` / `q2_k` 目前不可用（上游 gguf-py 只实现了它们的解码），
`q6_k` 走本仓库的 `convert/q6k.py`。Q8_0 及以上精度更高，Q6_K/Q4_0 更省显存与带宽。

### VideoMAE

读取任意 transformers `videomae` 检查点（`MCG-NJU/videomae-base`、`MCG-NJU/videomae-base-finetuned-kinetics` 等）：

```bash
python convert/convert_videomae.py --src models/videomae-base-ft   # → videomae-f16.gguf（约 190MB）
```

- Conv3d patch 嵌入折叠成一次 matmul（权重 `[dim, C*tubelet*patch*patch]`），
  token 顺序 `(t', h', w')`、元素顺序 `(c, dt, dh, dw)`，与卷积权重展平顺序一致
- **位置表不是权重**：VideoMAE 用的是原始实现里的固定 sin-cos 表（`get_sinusoid_encoding_table`，
  一维序号 0..1567），转换器现场生成后以 F32 存入 GGUF
- **没有 cls token**：`pooled` 取 token 0（`use_mean_pooling` 的 fine-tune 才做均值 + `fc_norm`）
- q/k/v 三块权重在转换期拼成 `v.blk.{i}.wqkv.weight`，bias 是 `q_bias | 0 | v_bias`
  （key 无 bias，这是 VideoMAE 特有的写法）
- 池化分两种，跟着检查点走：预训练版 `use_mean_pooling=false` → 取 token 0 且**有**末尾
  LayerNorm（`v.norm`）；`*-finetuned-kinetics` 这类 `use_mean_pooling=true` 的微调版 →
  对全部 token 求均值、**没有**末尾 LayerNorm，改用 `fc_norm`（eps 1e-5）。转换器按张量
  实际存在与否写 `videomae.has_final_norm` / `videomae.use_mean_pooling`
- 分类头的 `id2label` 写成 `videomae.labels` 字符串数组，引擎按它打印标签名；
  没有 `classifier.*` 的预训练检查点转换后只出嵌入
- `--mode N` 打印 top-N（默认 5），`--mode embed` 输出 768 维 pooled 嵌入（JSON 数组）；
  `--ask` 对这类模型没有意义，会被忽略
- **图只建一次**：clip 形状固定（16 帧 → 1568 token），`load()` 里建好静态图后每次只回填输入
  张量。这不只是省掉 per-clip 的建图/分配（实测 <0.5 ms，可忽略），更是 CUDA graph capture
  的前提——ggml 的 capture key 是 `nodes[0]`，图对象必须跨调用存活
- CPU 上想更快就量化权重：`--outtype q8_0`（~100 MB，i7-14700K 上 850 ms → 590 ms，
  Kinetics-400 的 top-1 概率只差 4e-4）。f16 更省事、GPU 更快，所以仍是默认
- 实测（i7-14700K，19 物理核，单 clip 1568 token × 12 层）：f16 约 0.85 s、q8_0 约 0.59 s

对拍：`python scripts/ref_videomae.py --model <hf_dir> --video clip.mp4 --out ref_videomae`
导出官方实现的 `v_patch_out / v_blk0_out / v_final / v_pooled / v_logits / v_probs`，
再用 `vqa-cli --dump-dir ref_videomae` 跑同一段视频，逐级比 `cpp_*.bin`。
本仓库提交前已用另一条独立通道（numpy 从 GGUF 权重复算整条前向，preprocess 用引擎自己的
`patches` 导出）验证：12 层全长 maxdiff 8e-3（值域 ~13），patch 嵌入 3e-3，block0 注意力
3e-4，Kinetics-400 分类头 400 个概率与引擎最大差 8.7e-6——都在 F16 权重噪声量级。

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
| Marlin-2B Q8_0（2.67GB） | CUDA | RTX 5070 Ti Laptop | ~9.7k tok/s | ~121 tok/s（视觉编码 112ms/300tok） |

每次推理都会在 stderr 打印一行 `[perf] prefill ... | decode ...`，便于自行核对。
moondream3.1 的数字见[当前状态](#当前状态)（官方标注来自 5070 Ti Laptop）。

## 仓库结构

```
src/
  core/                共享层：vqa::Model 接口、mmap/UTF-8/JSON 工具、字节级 BPE 分词器、
                       图像与视频帧解码（stb_image 实现所在）、按架构发现后端
  models/moondream/    moondream3.1 后端（引擎 + 视觉塔 + 多 crop 预处理 + region 头 + 服务端/批量）
  models/videochat3/   VideoChat3-4B 后端（视觉预处理 + 计算图）
  models/videomae/     VideoMAE 视频塔后端（clip 预处理 + 全时空注意力前向 + 分类/嵌入）
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
  `VC3_NO_FUSION=1` 退回未融合的 SwiGLU，  `--dump-ids` 只跑分词；VideoMAE 用
  `VIDEOMAE_DUMP_DIR=<dir>`（或 `--dump-dir`）导出 `patches / v_patch_out / v_blk0_qkv /
  v_blk0_attn_out / v_blk0_out / v_final / v_pooled / v_logits / v_probs`，
  `VIDEOMAE_MAX_LAYERS=N` 只跑前 N 层做逐层漂移对照（本机实测 12 层全长相对差 ~1e-3，
  约为 F16 权重噪声 + 线程归约顺序噪声）。
- **共享层（`src/core`）**：后端之间不放重复实现——`vqa.hpp` 接口、`backend.hpp`（CUDA 优先 +
  CPU 回退 + 线程数默认值，按**物理核**而不是逻辑核，SMT 对这类模型的 GEMV 是负收益）、
  `gguf_util.hpp`（元数据读取 / 张量转 F32）、`weight_load.hpp`（mmap 绑定、1D 张量转
  host F32、CPU 零拷贝/GPU 上传的 buffer 归属）、`load_util.hpp`（LoadOptions 落地 +
  各后端 `<PREFIX>_DUMP_DIR` 覆盖 + 特殊 token id 解析）、`vit_block.hpp`（所有 ViT 塔共用的 block 体，
  moondream / VideoChat3 / VideoMAE 三家都用它，差异走 hooks）、`resample.hpp`（smart_resize、
  PIL 语义 antialias bicubic 缩放、位置表双线性插值——原来寄居在 videochat3 目录里被另外三个
  模型反向 include）、`attention.hpp`（KV cache + mask）、`stage_buf.hpp`（device 侧暂存）、
  `ggml_dump.hpp`（对拍导出）、`sampler.hpp`、`tokenizer.cpp`、`media_io.cpp`。
- **改分词器数据**：`scripts/gen_unicode_ranges.py` 重新生成 `src/core/unicode_ranges.inc`。

## License

MIT，见 [LICENSE](LICENSE)。ggml 为 MIT（ggml-org）。模型权重与 tokenizer 遵循其各自条款，单独下载。
