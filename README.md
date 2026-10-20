# SkyVision.cpp

[![docker-mcp](https://github.com/SkYContact/SkyVision.cpp/actions/workflows/docker-mcp.yml/badge.svg)](https://github.com/SkYContact/SkyVision.cpp/actions/workflows/docker-mcp.yml)

用 [ggml](https://github.com/ggml-org/ggml) 在本地跑**图像/视频 VQA** —— 一个多模型推理引擎：
每个视觉语言模型是一个后端（实现 `vqa::Model`），统一前端按 GGUF 里的
`general.architecture` 自动选择后端，图像与视频帧序列共用同一套输入抽象。
运行时不需要 Python、不需要 CUDA Toolkit、不依赖任何厂商运行时。

[English](README.en.md)

也可以用 **MCP（Model Context Protocol）服务器**的方式跑：`vqa-cli --mcp` 把同一套引擎包成
stdio JSON-RPC 服务，暴露 9 个工具（看图问答 / 图像描述 / 目标检测 / 定位 / 注视点 / 视频问答 /
视频分类 / 模型管理），仓库自带 `Dockerfile`，镜像内置 403MB 的 SmolVLM，**不挂载任何权重也能
开箱回答"这张图里有什么"**。见 [MCP 服务器](#4-mcp-服务器给-agent-用) 一节。

## 支持的后端

| 后端 | 架构标签 | 输入 | 说明 |
|---|---|---|---|
| **moondream3.1-9B-A2B** | `moondream3` | 图像 | MoE 文本 + SigLIP 视觉塔；五个官方技能（caption / query / detect / point / gaze）、Web UI、批量推理 |
| **VideoChat3-4B** | `videochat3` | 图像 + 视频 | Qwen3 文本主干（36 层 GQA + per-head qk-norm）+ VideoChat3 自研 27 层 ViT（2D RoPE + 2×2 merge）、2×2→4608 merger、multimodal projector |
| **Qwen3.5 系（含 Marlin-2B）** | `qwen3_5`（旧 GGUF 标签 `marlin` 兼容） | 图像 + 视频 | Qwen3.5 混合主干（18 层 gated-delta 线性注意力 + 6 层 full attention，interleaved MRoPE）+ Qwen3-VL 式视觉塔（学习式绝对位置嵌入 + 2D RoPE + 逐时间步注意力）。官方 `Qwen/Qwen3.5-*` 检查点直接转换即用（结构同构、逐级 parity 已验证）；Marlin-2B 为其视频字幕特化版，额外带官方训练技能 caption（带时间戳事件）/ find（时序定位）
| **VideoJev-4B（打分式）** | `videojev` | 视频 | Qwen3.5-4B 主干（32 层：24 线性 + 8 full，**线性注意力 16 键 / 32 值头 GQA**）+ 同款视觉塔，但**末层被裁成 attention-only**（无 MLP/post-norm），并用训练好的 cosine 决策头替代 lm head：一次前向对候选列表打分（时序定位=逐时间片 sigmoid，单选=VQA softmax），**不生成文本**。走 `--score <qs.json>`，见"候选打分" |
| **Mage-VL-4B** | `mage_vl` | 图像 + 视频 | 微软亚研院 codec 原生 VLM：Mage-ViT Codec-ViT（24 层、16×16 patch、逐维 4:6:6 T/H/W RoPE、按帧分块注意力、2×2 block 布局直折 merger）+ Qwen3-4B-Instruct-2507 文本主干（36 层 GQA + qk-norm，与 VideoChat3 同构）。官方 `microsoft/Mage-VL` 检查点转换即用；图像单块、视频逐帧 `<X.X seconds>` 时间戳标签（帧采样后端）。社区首个 ggml 实现
| **MOSS-VL-Realtime（11B）** | `moss_vl` | 图像 + 视频 | OpenMOSS cross-attention 架构：Qwen3 文本主干（36 self 层 + 12 纯 cross-attn 层，tanh 双门控）+ SigLIP 塔（deepstack 8/16/24 + merger）+ 每帧可学习 separator。视觉 KV 走独立 cross 注意力，interleaved MRoPE（视觉三轴平面位置）。支持多轮会话（`?session=`） |
| **VideoMAE（base，video tower）** | `videomae` | 视频（图像=静帧） | 纯视觉塔，没有文本侧：Conv3d tubelet patch 嵌入 + 固定 sin-cos 位置表（**无 cls token**）+ 12 层全时空注意力，一次前向出分类标签（`*-finetuned-*` 检查点，如 Kinetics-400）或 pooled 嵌入（`MCG-NJU/videomae-base` 是预训练版，无分类头）。单 clip 1568 token，单图；`--mode embed` 直接吐 768 维特征 |
| **ViViT（b-16x2，video tower）** | `vivit` | 视频（图像=静帧） | 同一条视频塔（复用 `videomae` 后端）：差别只是**有 cls token**、位置表是**学出来的参数**、`gelu_fast`（tanh 形式）、LN eps 1e-6、qkv 三路都有 bias、预处理短边 256→裁 224 且**双线性**重采样 + 零中心化。32 帧 → 3136+1 token，一次前向出 Kinetics-400 标签 |
| **V-JEPA 2（ViT-L/16，video tower）** | `vjepa2` | 视频（图像=静帧） | Meta FAIR 的世界模型编码器：**没有位置表**，改用**三维分解 RoPE**（head_dim 64 = 20 时间 + 20 高 + 20 宽 + 4 不动，位置取自 `(t', h', w')` 网格）；24 层全时空注意力，q/k/v 三路都带 bias，GELU(erf)。64 帧 × 256² → **8192 token**。三个头都支持：**attentive pooler**（fine-tune 用它池化）+ **线性分类器**（`--mode N` 出 top-N 标签，如 SSv2/Diving48）+ **masked-latent predictor**（`--mode predict` 做掩码潜变量预测）。发布版只有编码器 + predictor |
| **MiniCPM-V 4.7（1B）** | `minicpmv47` | 图像 + 视频 | openbmb 端侧 VLM，**文本主干就是 Qwen3.5**（`text_config.model_type = qwen3_5_text`，24 层 3:1 gated-delta/全注意力、IMROPE 11/11/10、q\|gate 交织、untied lm_head），因此直接跑在 `qwen3_5` 后端的 minicpm 模式下，不重复一份文本图。新增部分只有：**SigLIP canvas 视觉塔**（27 层、d=1152、patch14、GELU-tanh、无 rope——learned 70×70 位置表按最近桶取值，整塔即共享 `vqa::vit::block`）+ **16× 两级合并**（第 6 层后插入窗口注意力 merger：2×2 窗口内注意力 + 4×d 展平后 LN/Linear/GELU/Linear，再 2×2 投影 MLP 到文本宽度）+ **canvas M-RoPE**（视觉 token 铺 2-D 画布，`<image>`/`</image>` 钉在画布光晕/远端角，文本计数从 `max(h,w)+1` 续；图像块后面那个模板换行也钉在 canvas 原点，官方行为）。`slice_mode` 走官方切片规则（长边 >448 起切，最多 9 块 `<slice>` 明细 + 缩略图，每块独立过一次塔），视频按帧推进同一套 canvas。转换 `convert/convert_minicpmv47.py`。开关：`MINICPM_DOWNSAMPLE=4x`（4x 保 token）、`MINICPM_NO_SLICES=1`、`MINICPM_SCALE_RES=<n>`（缩略图长边，默认 448）。预置权重：`hf download SkyContact/MiniCPM-V-4.7-1B-GGUF`（含 tokenizer，魔搭同名 `modelscope download --model SkyContact/MiniCPM-V-4.7-1B-GGUF`） |
| **Qwen2.5-VL（3B）** | `qwen2vl` | 图像（视频待接） | Qwen2 文本主干（36 层 GQA 16:2、qkv 带 bias、section MRoPE 16/24/24、SwiGLU）+ Qwen2.5 ViT（32 层、patch14、temporal2、2×2 merge、**窗口注意力**：每 8 层一次全局）+ merger。动态分辨率走 `smart_resize`（28 的倍数），预处理是 **CLIP mean/std**、bicubic 缩放、patch 按 **2×2 分组序**打包 |
| **Qwen2.5-Omni（3B）** | `qwen25omni` | 图像 + 音频输入，文本 + 语音输出 | 全功能 thinker-talker：Qwen2.5 文本主干（untied lm_head）+ Qwen2.5-VL 视觉塔（与 `qwen2vl` 同款窗口注意力图）+ Whisper 系音频塔（128-bin log-mel、**n_window=100 分块卷积+分块注意力**、25 token/s）→ **Talker**（24 层双轨 LM：文本轨吃 thinker 隐状态，输出 25Hz codec token）→ **Token2Wav**（flow-matching DiT，RK4×10 步 + CFG + sway 采样；BigVGAN v2 抗锯齿激活，纯 C++ 宿主实现）。音频输入走 `--audio`（ffmpeg 解码 16k mono）。参考实现：llama.cpp PR #13784（thinker）+ HF transformers 5.20 官方源码（全部语义基准），mel/音频塔与官方逐位对齐 |
| **SmolVLM（256M，Idefics3）** | `smolvlm` | 图像 | 目前最小的可用后端：SigLIP-base 视觉塔（12 层、patch16、512²、learned pos、LN/注意力/MLP 全带 bias、`gelu_tanh`）+ **4×4 pixel shuffle** 连接器（12288→576）+ SmolLM2-135M 文本主干（30 层、无 bias、GQA 9:3、rope 1e5）。预处理按官方 Idefics3：长边缩到 **2048** → 补齐到 512 的倍数 → 切 **512² 瓦片**（行优先）+ 一张全局视图，每块 64 token。f16 515 MB / q8 ≈ 270 MB |
| **EmbeddingGemma-2（740M）** | `embgemma2` | 图像 + 视频 + 音频 + 文本（嵌入） | 多模态嵌入模型（不生成文本）：Gemma3 文本塔（24 层 d=512、GQA 4:1、SWA/full 交错、**逐层 PLE**）+ Gemma4 ViT（16 层 patch16、轴向 2D RoPE、3×3 位置池化）+ Gemma4 音频塔（USM conformer：双 3×3 stride-2 卷积子采样 + 12 层 macaron FFN×2 / LightConv1d / **分块局部注意力 + Transformer-XL 相对位置偏置**，ClippableLinear 全程带 clamp）。`--embed` 输出 768 维（Matryoshka 可截 512/256/128/64）、`--embed-preset` 官方任务前缀、`--embed-file` 批量 JSONL。四模态对 fp32 HF 参考 **0.99999+** |

## 当前状态

> moondream3.1 全部功能已就绪：文本 + 视觉 + 五个官方技能 + Web UI + 批量推理 + 流式输出，
> CPU / CUDA 双后端，且和其它后端一样只通过统一的 `vqa-cli` 驱动（没有专属二进制/接口）。
> VideoChat3-4B 后端可用的部分见下方标注。

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
| VideoJev-4B 候选打分 | ✅ 全链路打通 + **对拍官方 transformers 5.14**（`scripts/parity_videojev.py`，prompt token 序列逐位一致）。以 CPU float32 参考为真值：引擎 f16 的最大概率偏差 **6.8e-3**（q8_0 5.9e-3），而官方仓库默认的 bf16 跑法偏差 5.7e-2 —— 引擎比官方参考更贴近 fp32。choice 端偏差 6.6e-6（官方 bf16 为 5.1e-5）。见 `scripts/parity_videojev.py` / `scripts/gen_fake_videojev.py` |
| Marlin-2B 视频结构化技能 | ✅ `--mode caption`（Scene + 带秒级时间戳的 Events 列表）/ `--mode find`（自然语言事件 → `From X.X to Y.X.`），训练对齐的官方提示词 |
| 视觉预处理 token 预算 | ✅ 图片/视频的视觉 token 上限 = n_ctx − max_tokens − 文本开销，编码前钳制分辨率或提前报错（不再白等几秒后 context overflow） |
| MOSS-VL-Realtime 图像/视频问答 | ✅ 端到端 + 官方栈对拍：图像 prefill token 流 24/24、视频 78/78 逐位一致；layer2 cross-attn 中间量 corr ≥ 0.999；KV cache 回读 vs 官方 roped K corr 0.9997。CUDA（5070 Ti Laptop，Q4_K）：视觉 213ms/图、decode 67 tok/s；CPU 16 线程 decode 8.9 tok/s |
| MOSS-VL 多轮会话 | ✅ `--serve` 的 `/ask?session=<id>`（`reset=1` 清空）：server 维护最近 12 轮文本历史，引擎 prompt 重建（媒体缓存命中时视觉塔免重跑） |
| MOSS-VL 实时流会话 | ✅ `/ask?session=<id>&rt=1`：官方 real_time_generate 的引擎侧实现——新帧/新问题增量追加到活跌 KV（文本 KV 续写 + cross-KV 行追加，不重建全上下文），位置按官方增量 MRoPE 公式；`<|silence|>` 作为闲置信号终止本轮。已验证：跨轮记忆（两张图分别在后续追问中正确引用） |
| MiniCPM-V 4.7 图像/切片/视频 | ✅ 端到端 + **逐级数值消融**（`scripts/ref_minicpmv47.py` 出 oracle，`scripts/parity_minicpmv47.py` 逐级对拍）：patches corr 0.999999、patch embed 0.999997、窗口 merger 0.999908、塔输出 0.999981、投影特征 0.999839、末 token logits 0.999820（top-8 完全相同），**贪婪 24 token 与官方逐字一致**（"The image displays three simple geometric shapes…"）；`MINICPM_DOWNSAMPLE=4x` 同套消融 corr 0.9997~0.99999；**切片路径**（1400×900 → 缩略图 + 6 切片，444 vision token）：canvas M-RoPE 位置 id 与官方逐元素相同（484 token）、7 个 crop 的 patch ≤2 灰阶、合并特征 corr 0.999458、prefill logits corr 0.999518（top-8 相同）；**视频**（6 帧 420 token）帧级 canvas 位置与官方逐元素相同（delta -343），按帧顺序叙述正确（"Frame 1: red square on the left, frame 2: slightly moved right…"）；切片开/关 OCR 消融：`MINCPM SLICE TEST12345` vs 幻觉 `MARGIN ACT SETTING` |
| CPU vs CUDA（RTX 4070 SUPER 12GB，7 crop 切片图 444 vision token） | 视觉 7704 ms → **244 ms**；prefill 484 tok 1680 ms → **32 ms**（15.3k tok/s）；decode 27 → **185 tok/s**；视频 6 帧视觉 7765 ms → **187 ms**。两条路径答案逐字相同 |
| Mage-VL 图像问答 | ✅ 端到端 + 视觉塔逐级 parity（`scripts/parity_mage_vision.py`，1GB 视觉塔-only oracle）：patches 5e-3、v_patch_out 1.5e-2、block0 qkv 7e-3；e2e 与官方 transformers（bf16 CPU/GPU oracle）贪心输出语义一致（llamas 测试图：数量/着装/颜色三问全对）。CUDA（3060 6GB，Q6_K）：视觉 162ms/图、prefill 2557 tok/s、decode 63 tok/s；q6_k 权重 4.28GB + 0.6GB KV 正好放进 6GB 显存
| Mage-VL System 1(主动流式门控) | ✅ `streammind_gate` 认知门控接入:PreNet→Mamba-1 逐窗口递推→PostNet→vocab-2 Qwen3 分类头(纯 C++ host 实现,与 torch 参考逐级对拍 rel≈0)。`--mode gate` 逐帧输出 p(speak);`--mode gated[:τ]` 门控生成;rt 流(`rt/start`)的纯观察轮自动走门控,`MAGE_GATE_TAU` 可调阈值(默认 0.5)。门控权重:`convert/convert_mage_gate.py` → `mage-gate-f16.gguf`(放到模型同目录自动加载)
| Mage-VL 视频问答 | ✅ `--video`（ffmpeg 帧采样后端，逐帧 `<X.X seconds>` 标签 + 每帧独立注意力块）：双图案拼接测试视频的前/后半段区分描述正确；1800 视觉 token prefill 2416 tok/s
| MOSS-VL 常驻推理流 | ✅ `rt/start?session=&src=&interval=`（src=文件/URL/`video=`摄像头）：ffmpeg 采集线程 + 常驻推理循环，模型按轮观察新帧并**主动发言**（`<|silence|>`=本轮无话可说）；`rt/say?q=` 流内提问（独立文本轮）、`rt/events?since=` 轮询 NDJSON 事件、`rt/stop` 停止 |
| VideoMAE 视频分类 / 特征 | ✅ 一个 GGUF 装全部权重（94M 参数，f16 约 180MB）：16 帧均匀采样 → 中心裁剪 224 → 一次全时空注意力前向 → top-k 标签；无 tokenizer、无 KV cache。**逐级对拍已验证**（numpy 从 GGUF 权重复算整条前向）：12 层全长相对差 ~1e-3、patch 嵌入 3e-3、block0 注意力 3e-4、Kinetics-400 的 400 个类别概率差 <1e-4。预训练版（无分类头）输出 pooled 嵌入，`--mode embed` 直出 768 维 JSON。CPU 单 clip（1568 token×12 层）约 0.85 s（q8_0 权重 0.59 s）|
| EmbeddingGemma-2 四模态嵌入 | ✅ 文本 0.9999999 / 图像 0.99999 / 音频 0.99999 / 视频 0.9999998（fp32 HF 参考，f16 权重）。音频塔逐层 hook 对拍修复 5 个图实现 bug（漏乘 attention 输出投影、content 项与 rel-bias 布局互为转置、rel-bias 行视图跨度、头合并 cont_2d 乱序、GLU 切分/残差/clamp）；视频逐帧跑 B=1 视觉图（注意力不跨帧，视频预处理用 video 预算 140 soft/帧） |
| V-JEPA 2 特征 / 分类 / 掩码预测 | ✅ ViT-L（326M + predictor 25M，f16 653MB）：64 帧 → 短边 292（双线性）→ 裁 256 → **8192 token 全时空注意力 ×24 层** → 1024 维特征；SSv2 微调版（16 帧）再走 **3 层 attentive pooler + 174 类线性头**，`--mode 5` 直出标签；`--mode predict` 跑 12 层 masked-latent predictor。**对拍已验证**（transformers 5.x）：同输入下 24 层全长 cos **1.000000**（max 0.065）；端到端（各自预处理）pooled cos **0.9994**、SSv2 top-1 标签一致（0.916 vs 0.921，概率最大差 4.5e-3）、predictor cos **0.998**。CPU 单 clip（8192 token）约 25 s，SSv2（2048 token）约 5 s |

## 快速开始

### 1. 构建

唯一第三方依赖是 ggml 本体（它同时提供 gguf）：

```bash
git clone https://github.com/SkYContact/SkyVision.cpp
cd SkyVision.cpp
git clone --depth 1 https://github.com/ggml-org/ggml

# Windows（MSVC + Ninja）
scripts\build-cuda.bat        # CUDA 后端（默认 sm_89=Ada，可用 CUDA_ARCH 覆盖）
scripts\build-vqa.bat         # 仅 CPU
scripts\build.bat             # 仅 CPU，改用 Visual Studio 生成器（产物在 build\Release\）
# 手工 CMake：
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release --target vqa-cli
```

`vqa-cli` 是按架构分派的统一前端——**包括 moondream3.1**：它没有独立二进制，也没有自带
HTTP server，全部能力（看图问答、caption/detect/point/gaze/ask-think、批量、Web UI）都通过
`vqa::Model` 接口由同一个 `vqa-cli` 驱动。
CUDA 构建下引擎默认优先使用 GPU，`--cpu`（或 moondream 的 `MD_CPU=1`）可强制 CPU。

可选：把 JPEG 解码换成 libjpeg-turbo（不 clone 也能编，走 stb_image，代价是与
PIL/libjpeg 的 IDCT 差 1-3 灰阶；换了之后与 PIL 逐位一致，omni 视觉塔 parity 闭环）：

```bash
git clone --depth 1 https://github.com/libjpeg-turbo/libjpeg-turbo vendor/libjpeg-turbo
# 上游禁止未修改的树被 add_subdirectory：把其自保护守卫改成自比较（该补丁已进 Dockerfile）
sed -i 's/CMAKE_SOURCE_DIR/CMAKE_CURRENT_SOURCE_DIR/g' vendor/libjpeg-turbo/CMakeLists.txt
# 有 nasm 自动开 SIMD（输出与不开逐位一致，只是快 2-4 倍）；MSVC 下自动对齐 /MD 运行时
```

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

# VideoJev-4B（打分式，不生成文本；用 --score 驱动）
hf download ZhuYuhan/VideoJev-4B --local-dir models/VideoJev-4B
python convert/convert_videojev.py --src models/VideoJev-4B -o videojev-4b-q8_0.gguf

# VideoMAE：预训练版（只有嵌入）或带 Kinetics-400 分类头的 fine-tune
hf download MCG-NJU/videomae-base --local-dir models/videomae-base
hf download MCG-NJU/videomae-base-finetuned-kinetics --local-dir models/videomae-base-ft

# V-JEPA 2：ViT-L 视频编码器（64 帧 256²，只出特征）
hf download facebook/vjepa2-vitl-fpc64-256 --local-dir models/vjepa2-vitl-fpc64-256
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

# VideoJev-4B：候选打分（不生成文本）。grounding 逐时间片 sigmoid → 区间；
# choice 对选项 softmax → answer_index。--threshold 覆盖模型自带阈值（默认 0.25）
cat > qs.json <<'JSON'
{"questions": [{"id": "q1", "task": "grounding", "query": "when does the person sit down"},
               {"id": "q2", "task": "choice", "query": "where does it happen",
                "options": ["A. indoors", "B. outdoors"]}]}
JSON
vqa-cli --model videojev-4b-q8_0.gguf --video clip.mp4 --fps 2 --score qs.json

# VideoMAE：视频 → 动作标签（分类头版）/ 768 维嵌入（预训练版）
vqa-cli --model videomae-base-ft.gguf --video clip.mp4 --ask x --mode 5
vqa-cli --model videomae-base.gguf    --video clip.mp4 --ask x --mode embed

# V-JEPA 2：视频 → 1024 维特征 / SSv2 标签 / 掩码潜变量预测
vqa-cli --model vjepa2-f16.gguf --video clip.mp4 --ask x --mode embed
vqa-cli --model vjepa2-f16.gguf --video clip.mp4 --embed ""     # L2 归一化向量
vqa-cli --model vjepa2-ssv2-f16.gguf --video clip.mp4 --ask x --mode 5
vqa-cli --model vjepa2-f16.gguf --video clip.mp4 --mode predict \
        --ask "ctx=0:4096,tgt=4096:8192"                        # 预测后一半 token 的潜变量

# EmbeddingGemma-2：多模态嵌入（图像/视频/音频/文本 → 768 维，不生成文本）
vqa-cli --model embeddinggemma2-f16.gguf --tokenizer models/embeddinggemma-2 \
        --image photo.jpg --embed "describe" --embed-preset query          # JSON 输出嵌入向量
vqa-cli --model embeddinggemma2-f16.gguf --tokenizer models/embeddinggemma-2 \
        --video clip.mp4 --embed "summarize"                               # 1 fps 均匀采样 ≤32 帧
vqa-cli --model embeddinggemma2-f16.gguf --tokenizer models/embeddinggemma-2 \
        --audio clip.wav --embed "what is said" --embed-dim 256            # Matryoshka 截断
vqa-cli --model embeddinggemma2-f16.gguf --tokenizer models/embeddinggemma-2 \
        --embed-file docs.txt                                              # 批量：一行一条 → JSONL

# moondream3.1 就是一个普通后端：技能、批量、Web UI 全在 vqa-cli 上
vqa-cli --model models/moondream3.1-q6k.gguf --image photo.jpg --ask "What is in this image?"
vqa-cli --model models/moondream3.1-q6k.gguf --image photo.jpg --ask person  --mode detect
vqa-cli --model models/moondream3.1-q6k.gguf --image photo.jpg --ask normal  --mode caption
vqa-cli --model models/moondream3.1-q6k.gguf --image photo.jpg --ask "0.5,0.5" --mode gaze
vqa-cli --model models/moondream3.1-q6k.gguf --batch prompts.txt      # 纯文本批量
vqa-cli --serve 8080                                                  # 统一 Web UI
# --vision/--region 省略时按模型所在目录找 moondream3.1-vision/-region.gguf
```

`vqa-cli` 常用开关：`--video <file> --fps 2 --max-frames 32`（内置视频：ffmpeg 管道解码 + qwen_vl_utils 式采样 n = min(duration×fps, max_frames)，超限自动全片均匀重采样）、`--frames <dir>`（预抽帧目录）、`--temp/--top-k/--top-p`（采样，默认贪心）、
`--max-tokens`、`--n-ctx`、`--threads`、`--cpu`、`--dump-ids`（只跑分词并打印 token id）、
`--help` 看全部。

moondream Web UI 覆盖官方全部五个技能：图像描述（简短/标准/详细）、提问（VQA）、
目标检测（绿框）、定位（黄色十字）、注视点预测（点选起点→预测注视点）。
对同一张图连续提问时自动复用视觉编码与 KV 前缀（~1.2s → ~66ms）。
HTTP API：`POST /api/vision`，`mode=ask|caption|detect|point|gaze`
（caption 带 `len=short|normal|long`，gaze 带 `x=&y=` 起点坐标），body 为图片二进制。

**双模型同驻**：单槽 registry 一次只驻留一个模型，换载已异步化——`POST /api/model/<id>/load`
立即返回 `{"loading":true}`，轮询 `GET /api/model/<id>/status`（loading 进度字节/ready/error），
换载期间 `/ask` 返回 409，加载前有显存守卫（预估 权重×1.1+KV，不足时给行动建议；
`SKYVISION_FORCE_LOAD=1` 可越过）。两个大模型要同时在线时用双进程：
`scripts\serve-dual.bat`——GPU 实例 `--serve 8080` 跑 MOSS-VL，第二实例 `--serve 8081 --cpu`
跑 Qwen2.5-Omni（CPU 全功能含 `--speak`），端口/绑定地址可用 `SKYVISION_*_PORT/_HOST` 环境变量改。

### 4. MCP 服务器（给 agent 用）

`vqa-cli --mcp` 把同一个引擎包装成 **MCP（Model Context Protocol）** 服务器，走标准
stdio + JSON-RPC（换行分隔），任何支持 MCP 的客户端（CodeBuddy / Claude Desktop / 自研 agent）
都能直接让模型"看图/看视频"，不需要写粘合代码：

```bash
# 从仓库根目录启动；--models-dir 省略时用 ./models（不存在则用 .）
vqa-cli --mcp --models-dir models
# 也可以固定一个默认模型（工具调用里不写 model 时用它）
vqa-cli --mcp --models-dir models --model moondream3.1-q6k.gguf
```

客户端配置示例（`mcpServers` 条目）——用仓库自带的启动脚本最省事，它会切到仓库根目录、
挑一个真正支持 `--mcp` 的构建目录、把 `ggml*.dll` 所在目录加进 `PATH` 再启动：

```json
{
  "mcpServers": {
    "skyvision": {
      "command": "cmd",
      "args": ["/c", "C:\\moondream\\scripts\\mcp-server.bat"],
      "env": { "FFMPEG": "ffmpeg" }
    }
  }
}
```

也可以直接指向二进制（这时要自己保证 DLL 能找到）：

```json
{
  "mcpServers": {
    "skyvision": {
      "command": "C:/moondream/build-ninja/vqa-cli.exe",
      "args": ["--mcp", "--models-dir", "C:/moondream/models"],
      "env": { "PATH": "C:/moondream/build-ninja/bin;%PATH%" }
    }
  }
}
```

> 共享库版构建把 `ggml*.dll` 放在 `<build>/bin/`，客户端从任意工作目录启动时不会自动找到，
> 所以要么走启动脚本，要么把该目录加进 `PATH`／把三个 DLL 拷到 `vqa-cli.exe` 旁边。
> `scripts/mcp-server.bat`（Windows）与 `scripts/mcp-server.sh`（Linux/macOS/WSL）都支持
> `SKYVISION_BUILD` / `SKYVISION_MODELS` 覆盖。视频工具另外需要 `ffmpeg`（`FFMPEG` 可指定
> 路径），纯图片工具不需要。

#### Docker（MCP 目录/沙箱跑的就是这个）

仓库自带 `Dockerfile`：多阶段构建出 CPU 版二进制，并**内置 403MB 的 SmolVLM-256M q8**，
所以镜像开箱就是个能真答话的 MCP 服务器：

```bash
docker build -t skyvision-mcp .                       # 内置 403MB smolvlm
docker build --build-arg MODEL_REPO= -t skyvision-mcp:bare .   # 不带权重（CI 用）
docker run --rm -i skyvision-mcp                      # 内置 smolvlm，直接可用
docker run --rm -i -v ~/models:/models skyvision-mcp   # 换成自己的 GGUF
```

入口是 `vqa-cli --mcp --models-dir /models`；内置模型就是 `describe_image` / `ask_image`
的兜底，挂进去的每个 `*.gguf` 都会出现在 `list_models` 里。镜像预设
`SMOLVLM_MAX_EDGE=1024`（官方预处理会放大到 2048，见 SmolVLM 那节：同样答案但慢 2.3 倍）。

**完全没有权重时**它也不会崩：`initialize` / `tools/list` / `list_models`（返回空列表）
都正常应答 —— 这正是目录爬虫、CI 冒烟测试和 MCP 平台审核所做的事。

内置权重默认从 Hugging Face 的 `SkyContact/SmolVLM-256M-GGUF` 拉取（魔搭有同款镜像），
换源或固定版本用 `--build-arg MODEL_BASE=... --build-arg MODEL_REV=...`；
`--build-arg MODEL_REPO=` 则完全不下载（得到一个只有二进制、靠 `-v` 挂权重的镜像）。

[`.github/workflows/docker-mcp.yml`](.github/workflows/docker-mcp.yml) 在 Linux 上真的构建
这个镜像并跑冒烟，所以 README 顶部的徽章是"真构建过"，不是宣称。构建成本按三层摊开：

| 层 | 何时失效 | 说明 |
|---|---|---|
| `toolchain` | 只改 apt 包列表 | apt lists 走 cache mount |
| `ggml` | 只改 `GGML_REF` | ggml 独立成 stage 单独编译（`-DGGML_PREBUILT_DIR` 复用），**改 `src/` 不会碰它** |
| `build` | 改源码 | `COPY .` + ccache（本地重复构建有效；CI 跨 runner 不复用 ccache，靠上面两层的层缓存） |

CI 的推送构建默认**不带权重**（`MODEL_REPO=`）：BuildKit 的 `--mount=type=cache` 在新 runner 上
是空的，带权重等于每次多传 400MB。完整镜像（含 `describe_image` 冒烟）走每周定时 +
手动 `workflow_dispatch`，缓存用 ghcr registry（`mode=max`，无 10GB 配额限制）。

| 工具 | 说明 |
|---|---|
| `list_models` | 列出可运行的 GGUF（架构/量化/体积/是否常驻）+ 已加载模型的能力位 |
| `load_model` | 显式加载（同一时刻只常驻一个模型；换模型会卸载旧的），可设 `n_ctx` / `threads` / `gpu` |
| `describe_image` | 图像描述，`length=short\|normal\|long` |
| `ask_image` | 自由提问（`mode=ask-think` 可开 moondream 的 grounded 推理），支持 `session` 多轮 |
| `detect_objects` | 目标检测，返回 JSON（归一化 0..1 + 像素坐标），需 moondream3 + `*-region.gguf` |
| `point_objects` | 目标中心点，同上 |
| `gaze_point` | 由起点预测注视点，同上 |
| `analyze_video` | 视频问答（ffmpeg 抽帧，`fps` / `max_frames`），Marlin 支持 `mode=caption\|find` |
| `classify_video` | 纯视频塔（VideoMAE / ViViT / V-JEPA 2）→ top-k 标签或 pooled 嵌入 |

约定：媒体参数既可以是本地路径，也可以是 `data:image/...;base64,<data>` / `base64:<data>`；
静止图片可以直接喂给 `classify_video` / `analyze_video`（当作单帧 clip，不需要 ffmpeg）；
坐标类工具返回两层坐标（`x1/y1…` 归一化、`x1_px/…` 像素），方便直接画框。
工具失败以 `isError: true` + 可读原因返回（不占用 JSON-RPC 错误码），协议层错误才用 JSON-RPC error。

## 模型转换

所有转换器都在 `convert/`，共用 `hf_source.py` 定位检查点、`_lib.py` 提供 GGUF/量化工具，
日志统一 `[convert]` 前缀，写出后会重新打开文件校验张量表。

| 脚本 | 产出 |
|---|---|
| `convert_hf_to_gguf.py` | `moondream3.1-q8_0.gguf` / `-q6k` / `-q4k`（文本主干，`--q6` / `--q4` / `--q8-dense`） |
| `convert_vision.py` | `moondream3.1-vision.gguf`（SigLIP 塔，~0.84GB） |
| `convert_region.py` | `moondream3.1-region.gguf`（detect/point/gaze 头，~30MB） |
| `convert_videochat3.py` | `videochat3-4b-<outtype>.gguf`（Qwen3 + VideoChat3 ViT，单文件，`--outtype f16\|q8_0\|q6_k\|q4_0`） |
| `convert_qwen3_5.py` | Qwen3.5 / Marlin-2B 通用：`<检查点>-<outtype>.gguf`（混合主干 + Qwen3-VL 视觉塔，单文件，`--outtype f16\|q8_0\|q6_k\|q4_0`，默认 q8_0；Marlin 检查点自动写入 `marlin.skills` 技能位） |
| `convert_videojev.py` | `videojev-4b-<outtype>.gguf`（cosine 打分头与特殊 token 一起写进 GGUF，`--outtype` 默认 q8_0） |
| `convert_mage_vl.py` | `mage-vl-4b-<outtype>.gguf`（Codec-ViT + Qwen3 文本主干，单文件，`--outtype f16\|q8_0\|q6_k\|q4_0`） |
| `convert_minicpmv47.py` | `minicpm-v-4.7-1b-<outtype>.gguf`（SigLIP canvas 塔 + 窗口/投影 merger + Qwen3.5 文本主干，单文件，`--outtype f16\|q8_0\|q6_k`） |
| `convert_moss_vl.py` | `Moss-VL-Realtime-<nf4-out>.gguf`（官方 NF4 self 层权重反量化后重量化，默认 q4_k；`--nf4-out` / `--bf16-text-out` / `--vision-out` 分层选档） |
| `convert_videomae.py` | `videomae-<outtype>.gguf`（VideoMAE 视频塔，单文件，`--outtype f16\|q8_0\|q4_0`；分类头与 `id2label` 一起写进 GGUF） |
| `convert_vjepa2.py` | `vjepa2-<outtype>.gguf`（V-JEPA 2 编码器 + 可选 attentive pooler/分类器/predictor，单文件，`--outtype f16\|f32\|q8_0\|q4_0`，f16 约 653MB；`--no-predictor` / `--no-pooler` 裁头；q/k 行按 RoPE 布局重排） |
| `convert_qwen2vl.py` | `qwen2vl-<outtype>.gguf`（单文件，`--outtype f16\|q8_0\|q4_0`，默认 f16） |
| `convert_qwen25omni.py` | `qwen25omni-<outtype>.gguf`（thinker）+ `qwen25omni-talker-<outtype>.gguf`（talker+token2wav），`--outtype f16\|q8_0\|q4_0`、`--vision-f32`、`--no-talker` |
| `convert_smolvlm.py` | `smolvlm-<outtype>.gguf`（单文件，`--outtype f16\|q8_0\|q4_0`，默认 f16） |
| `convert_embedgemma2.py` | `embeddinggemma2-<outtype>.gguf`（四模态嵌入，单文件，`--outtype f16\|q8_0\|q4_0`；f16 约 1.5GB） |

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
python convert/convert_qwen3_5.py --src models/Qwen3.5-2B   # → qwen3.5-2b-q8_0.gguf（2.67GB，Qwen3.5/Marlin 通用）
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

### ViViT

`google/vivit-b-16x2` 在 HF 里就是"ViT + tubelet 卷积 + cls token"，所以直接复用 `videomae`
后端，差别全部走 GGUF 元数据：

```bash
python convert/convert_vivit.py --src models/vivit-b-16x2      # → vivit-f16.gguf（约 183MB）
```

- **cls token**：学出来的参数（`v.cls_token.weight`），在 patch 嵌入之后、位置相加之前 concat
  到序列最前面；位置表本身有 3137 行（含 cls）
- **位置表是学出来的参数**（`vivit.embeddings.position_embeddings`），与 VideoMAE 的固定
  sin-cos 不同
- **激活是 `gelu_fast`**：注意 transformers 的 `FastGELUActivation` 是**tanh 形式**，不是
  `x·sigmoid(1.702x)`（后者叫 `quick_gelu`）——一开始按 sigmoid 理解导致 block0 就对不上，
  用 numpy 复算三种激活才定位到
- LN eps **1e-6**；q/k/v **三路都有 bias**（VideoMAE 的 key 没有）
- 预处理：短边 256 → center crop 224，**双线性**重采样（`preprocessor_config.json` 里
  `resample: 2` 是 PIL 的 BILINEAR，不是 BICUBIC），零中心化（等价 mean=std=0.5）
- 32 帧 → 3136 patch token + 1 cls；CPU（19 核）单 clip 约 2.2 s

对拍：`python scripts/ref_vivit.py --model models/vivit-b-16x2 --image photo.jpg --out ref_vivit`，
引擎 `--dump-dir ref_vivit` 跑同一张图，实测 `v_patch_out` mean|d| 3.5e-3、block0 1.0e-2、
12 层全长 1.4e-2（值域 22，corr 0.99977）、400 类概率最大差 **6.0e-3**（top-5 标签顺序一致），
残差全部来自重采样舍入（像素级差 ~1.8 灰阶）。

### V-JEPA 2

```bash
hf download facebook/vjepa2-vitl-fpc64-256 --local-dir models/vjepa2-vitl-fpc64-256
python convert/convert_vjepa2.py --src models/vjepa2-vitl-fpc64-256 -o vjepa2-f16.gguf   # 608MB
vqa-cli --model vjepa2-f16.gguf --video clip.mp4 --ask x --mode embed   # 1024 维特征 JSON
```

预转好的权重（免转换直接下）：
`hf download SkyContact/VJEPA2-ViT-L-fpc64-256-GGUF`（编码器 + masked-latent predictor，魔搭同名）·
`hf download SkyContact/VJEPA2-ViT-L-fpc16-256-SSv2-GGUF`（SSv2 174 类分类微调，魔搭同名）。

这是纯编码器（世界模型的一半）：输出 1024 维 mean-pooled 视频特征，不生成文本、也没有分类头。

- Conv3d（`kernel=stride=(tubelet,patch,patch)`）折成一次 matmul，token 顺序 `(t', h', w')`、
  元素顺序 `(c, dt, dh, dw)`，与 `videomae` 共用同一份 clip 打包代码（`videomae_preprocess`）
- **没有位置表，位置在层内的 RoPE 里**：head_dim 64 切成 `20(时间) + 20(高) + 20(宽) + 4(不动)`，
  规则是 `seg = 2*((head_dim//3)//2)`；token 的 `(t', h', w')` 网格下标就是三趟旋转的 position id
- **RoPE 有一个必须照抄的上游 bug**：`rotate_queries_or_keys` 里 `repeat(1,1,1,2)` 是**平铺**
  而不是交织，于是相邻维对 `(2u, 2u+1)` 的偶分量用频率 `2u`、奇分量用 `2u+1`（不是正交旋转）。
  Meta 源码注释明确写了"修正会破坏与预训练权重的兼容性"（vjepa2 PR #15），所以必须复现。
  转换器把 q/k 的每个旋转段重排成 `[偶数维 | 奇数维]`，引擎就能用 `mul/add` 直接算，
  q 和 k 用同一重排 → 注意力打分不受影响（GGUF 里写 `vjepa2.rope_layout = "even-odd"`，
  引擎加载时校验，不匹配直接报错而不是静默出错）
- 4096 宽的 MLP 是 `hidden_size * mlp_ratio`，`hidden_act = gelu` 是 **erf 精确版**；LN eps 1e-6；
  q/k/v 与输出投影**全部带 bias**
- 预处理跟 `videomae` 不同：短边缩到 **292**（`video_preprocessor_config.json` 的
  `size.shortest_edge`，`resample: 2` 双线性）再中心裁 256；`default_to_square` 在
  `size` 是 dict 时不生效（实测 HF 仍按短边等比缩放，已按此实现）
- 64 帧 × 256² = **8192 token**，全时空注意力单层就是 16 头 × 8192²，CPU 上单 clip 25-30 s
  （19 核）；GPU 上建议直接跑 f16
- 张量：`v.patch_embd.*` / `v.blk.{i}.{ln1,wqkv,wo,ln2,fc1,fc2}` / `v.norm.*`，共 292 个
- `predictor.*`（12 层、`pred_hidden_size=384`，含 mask token 与 1024↔384 投影）只服务预训练的
  masked latent prediction，转换时整体丢弃（199 个张量），需要 rollout/规划再接

三个头（转换器按权重自动识别，`--no-predictor` / `--no-pooler` 可裁）：

- **attentive pooler**（`*-ssv2` 这类 `VJEPA2ForVideoClassification` 才有）：先 N 层
  普通 pre-norm 自注意力（**无 RoPE、无 mask**）跑完整 token 流，再用**一个学出来的 query
  token 对整段做 cross attention**——注意这个 cross attention **没有输出投影**
  （上游注释点明），query 本身也不做 LN，只有被查询的流走 LN；随后 query 残差 + MLP
- **分类器**：pooler 输出过一个线性层 → `--mode N` 打印 top-N 标签，`id2label` 写进
  `vjepa2.labels`；**嵌入口径也跟着变**：有 pooler 时 `--mode embed`/`--embed` 给 pooler 输出
  （官方 `get_vision_features` 同样这么选）
- **masked-latent predictor**（12 层 d=384，含 mask token 与 1024↔384 投影，f16 约 45MB）：
  `--mode predict --ask "ctx=0:4096,tgt=4096:8192[,mask=1]"`（区间写 `a:b` 半开、
  `a-b` 闭区间、多项用 `/` 分隔）。流程与官方一致：取编码器流在 ctx token 上的特征 →
  1024→384 投影 → 拼上 mask token → **按位置排序** → 12 层窄栈（自己的 RoPE 表）→ LN →
  反排序取 target token → 384→1024 投影。predictor 图按 `(n_ctx, n_tgt)` 缓存，换掩码形状才重建
  （~10 ms），每次只回填索引与表
- 坑：transformers 的 `VJEPA2Predictor.forward` **不会把 `mask_index` 传下去**，永远用索引 1
  （发布权重里 1..9 号 mask token 全零，只有 0 号有范数 0.769）——引擎默认也用 1 以对齐，
  `mask=0` 才用那颗真训过的 token
- 第二个坑：predictor 的 head_dim 是 32 → 每段只有 `seg/2 = 5` 个频率，**频率表长度是奇数**，
  奇分量频率 `(2u+1) % 5` 会回绕（不是 `偶分量 + 1`）。编码器 seg=20 时 `seg/2=10` 为偶数，
  这个差别看不出来，只有 predictor 会错

对拍：`python scripts/ref_vjepa2.py --model <hf_dir> --video clip.mp4 --out ref_vjepa2`
导出 `v_patch_out / v_blk0_out / v_final / v_pooled`（微调版再加 `v_logits / v_probs`），
以及 `ref_p_latents.npy` + `predict_spec.txt`（直接喂 `--mode predict --ask "$(cat ...)"`）；
`vqa-cli --dump-dir ref_vjepa2` 跑同一段视频逐级比 `cpp_*.bin`（按 ggml 的 `ne` 顺序读，
二维张量是 `[dim, tokens]` 列主序）。实测（ViT-L/fpc64-256，本机 19 核）：

| 对照 | 结果 |
|---|---|
| **同输入**（把引擎的 `patches` 灌回官方实现）24 层全长 | cos **1.000000**、max 0.065 —— 图实现本身没有偏差 |
| 端到端 `v_patch_out` / block0 | 相对差 6.9e-4 / 7.1e-4 |
| 端到端 24 层全长 | 逐 token cos 0.91（8192 token 全注意力对**亚灰阶**输入差极敏感，均值池化后收敛） |
| mean-pooled 特征 | cos **0.9994~0.9998** |
| SSv2 pooler + 174 类头 | top-1 标签一致（0.916 vs 0.921），概率最大差 **4.5e-3**，top-4 集合一致 |
| predictor（4096 / 1024 target） | cos **0.998**，max 0.5 |

端到端残差全部来自重采样：引擎走 PIL 语义，HF 的 video processor 走 torch 抗锯齿，
像素级 mean 差 1.7e-4（max 0.0175 归一化 ≈ 半个灰阶）。想要更硬的对照就用
`--outtype f32`（1.2GB）+ 同输入法。调试：`VJEPA2_MAX_LAYERS=N` 只跑前 N 层，
`--dump-dir` 额外导出 `v_blk0_qkv / v_blk0_q_rope / v_blk0_k_rope / v_blk0_attn / v_logits / v_probs`。

复用了什么：block 体、flash-attn 接线、成对旋转 RoPE 全部来自 `core/vit_block.hpp`
（moondream / VideoChat3 / VideoMAE / ViViT 同一份，本后端只补了「第二条角度表」和
「无 rope」两个开关）；patch 打包直接用 videomae 的 `pack_clip`；卷积折 matmul、ggml 的
`flash_attn_ext` / `norm` / `gelu_erf` / `mul_mat` / `get_rows / concat`、GGUF 读写
（`gguf` + `_lib`）都与其它后端同源。真正手写的只有本架构特有的图接线（三维 RoPE 表、
pooler 的 cross attention、predictor 的排序/反排序）——ggml 的 `ggml_rope_*` 表达不了上游那个
「频率平铺」的 bug，所以旋转只能落成 mul/add。

### Qwen2.5-VL

```bash
python convert/convert_qwen2vl.py --src models/qwen2.5-vl-3b     # → qwen2vl-f16.gguf（7.5GB）
vqa-cli --model qwen2vl-f16.gguf --tokenizer models/qwen2.5-vl-3b \
        --image photo.jpg --ask "How many people are in the image?"
```

视觉塔（Qwen2.5 ViT）和文本主干（Qwen2）各自的图都照 llama.cpp / HF 的结构实现，踩到并修掉的
三个坑值得记下来，它们都属于"数值看着像、结果全错"的类型：

1. **patch 排布是 2×2 分组优先**。HF 的 `Qwen2VLImageProcessor` 会
   `reshape(grid_t, gh/2, 2, gw/2, 2, C, T, P, P)` 再 `transpose(0,3,6,4,7,2,1,5,8)`，
   所以 `pixel_values` 的行序是 `(t, gh/2, gw/2, 2, 2)`，不是光栅序。按光栅序打包会让
   图像"看起来还在"，但特征全错。
2. **merger 的 RMSNorm 是逐 token 1280 维**，归一化完才 reshape 成 5120 喂 fc1。先 reshape
   再 norm（5200 维上归一化）会静默毁掉特征——这一条让我在"塔输出 corr 0.98 但答案错"上卡了很久。
3. **KV cache 的槽位 ≠ MRoPE 的 t 轴**。图像 token 的 t 坐标全是同一个值（`st`），拿它当
   `ggml_set_rows` 的索引会把 612 个 token 全写进同一格。槽位必须单独用 0..T-1。

另外：预处理用 **CLIP mean/std**（不是 0.5）、bicubic 缩放、`smart_resize` 到 28 的倍数；
文本是 section MRoPE（16/24/24），图像 token 位置 `(st, st+h, st+w)`，图像结束后 st += max(gh, gw)，
解码时下一个位置取 `max(pos)+1` 而不是 `T`。

对拍：`python scripts/ref_qwen2vl.py --model models/qwen2.5-vl-3b --image photo.jpg --out ref_q2v`，
引擎 `--dump-dir ref_q2v` 跑同一输入。实测：patch 逐元素一致、block0 maxdiff 0.16（值域 15）、
塔输出 corr 0.982（值域 2.3e4）、merged 特征 corr **0.9935**、首 token 命中官方 argmax；
问答结果与官方一致（"There are five people in the image."）。调试开关：`Q2V_MAX_VLAYERS=N`
截断视觉层数、`Q2V_FEATS=<file>` 直接用磁盘上的特征（隔离塔与文本侧）、`Q2V_DEBUG=1` 打印
token 与首 token top5。

### SmolVLM（最小可用后端）

256M 的 Idefics3 模型，量化后 ~270 MB，适合塞进容器/CI 里当"永远能跑"的 demo：

```bash
python convert/convert_smolvlm.py --src models/smolvlm-256m        # → smolvlm-256m-f16.gguf（515MB）
python convert/convert_smolvlm.py --src models/smolvlm-256m --outtype q8_0   # ~270MB
vqa-cli --model smolvlm-256m-f16.gguf --tokenizer models/smolvlm-256m \
        --image photo.jpg --ask "How many people are in the image?"
```

结构（都是实测的，不是照文档推的）：

- **视觉塔**是 SigLIP-base：`Conv2d(3,768,16,16)` 折成一次 matmul（带 bias）、**学出来的 32×32 位置表**（无 cls）、12 层 **LayerNorm 带 bias** + 注意力 qkv/out 全带 bias（**没有 rope**）+ 普通 MLP（`gelu_pytorch_tanh`，不是 SwiGLU）
- **连接器**是 4×4 pixel shuffle → 一次 `Linear(12288→576)`，所以每张 512² 瓦片恰好 64 个文本 token
- **文本主干**是 SmolLM2-135M：30 层、d=576、9/3 heads、**所有投影都没有 bias**、RMSNorm eps 1e-5、rope theta 1e5、**lm_head 不共享**

预处理照官方 Idefics3（这条最容易做错）：先把长边缩到 **2048**（不是 512），再把两边**补齐到 512 的倍数**（第二次缩放会拉伸、不保比例），切成 512² 瓦片（行优先），**最后追加一张整图的全局视图**；`<image>` 会展开成
`<fake><row_R_col_C><image>×64`（每块）与结尾的 `<fake><global-img><image>×64<fake>`。960×504 的样图是 4×3 瓦片 + 全局 = **13 块 / 832 token**，与官方处理器逐项一致。

分词器是 SmolLM2 的 `Digits + ByteLevel/GPT-2 regex`，为此给 `core/tokenizer` 加了第三种模式（`Gpt2Regex`）——和 Qwen 的差别在"标点能否前缀字母"和标点后是否吞 CR/LF；三种测试串（特殊 token / 缩略语 / 数字 / 制表符 / 多空格）与官方分词器**逐 id 相同**。

对拍：`python scripts/ref_smolvlm.py --model models/smolvlm-256m --image photo.jpg --out ref_smolvlm`，
引擎 `--dump-dir ref_smolvlm` 跑同一输入。实测像素 corr **0.99997**（maxdiff 0.046，来自第二次缩放我们把中间结果取整回 u8）、last-token logits corr **0.99991**（maxdiff 0.23），**生成文本与官方逐字相同**（" There are five persons in the image."）。

性能（CPU 19 线程，f16 权重，960×504 样图）：

| 配置 | 图像块 | 端到端 |
|---|---|---|
| `SMOLVLM_MAX_EDGE=2048`（官方预处理原样） | 4×3 瓦片 + 全局 = 13 | 8.6 s |
| **默认（长边上限 1024）** | 2×2 + 全局 = 5 | **3.8 s** |
| `SMOLVLM_MAX_EDGE=512` | 1×1 + 全局 = 2 | 1.6 s |

三档答案完全相同。开销几乎全在视觉塔的**全注意力**：一块 1024 token = 2·1024²·64·12 ≈ 1.6 GFLOP/层 × 12 层 ≈ 19 GFLOP，所以"块数"就是时间，`--dump-dir` 那种逐块模式并不会更慢。

顺带记录一个反直觉的结论：把所有瓦片**批合并**成一次前向（batch = 块数）在 CPU 上**几乎没有收益**（8.7 s → 8.7 s）——瓶颈是 FLOP 不是调度，批合并只是少几次 launch；代码保留了批处理形式（GPU 上才有意义），并且 `--dump-dir` 时自动退化成单块以保证可与官方逐级对比。

模型权重遵循各自的许可证（moondream 为
[Moondream Model License 1.0](https://moondream.ai/licenses/model/1.0)），**不在本仓库分发**。
该许可明确允许量化、转换、再分发、自托管与商用集成；仅对公众开放的通用 moondream API
转售需要单独授权——那是部署者的义务，与本引擎无关。

### Qwen2.5-Omni

```bash
python convert/convert_qwen25omni.py --src models/qwen25omni/hf --outtype f16 --vision-f32
# → qwen25omni-f16.gguf（thinker）+ qwen25omni-talker-f16.gguf（talker+token2wav+音色）
vqa-cli --model qwen25omni-f16.gguf --tokenizer models/qwen25omni/hf \
        --image photo.jpg --audio clip.wav --ask "What is shown?" \
        --speak reply.wav          # 语音输出（24kHz wav，默认音色 Chelsie）
```

thinker 侧：文本主干走 QWEN2VL 同构图（注意 lm_head **不 tied**，转换器写 `t.output.weight`）；
音频塔与原味 whisper 的三点差异都按 HF 5.20 官方实现复刻——mel 按 n_window×2=200 帧分块后
**逐块独立卷积**（ggml 侧用 4 零列 + conv1 后零化边界两列精确复现）、块内注意力、帧对 (2i, 2i+1)
平均池化（floor 语义，奇尾帧丢弃）。`ggml_conv_1d` 的三要素（kernel 字节 = torch [OC,IC,K] 原样、
数据通道连续、输出 [T,C] 且 bias 在转置后加）在 `qwen25omni_audio.cpp` 顶部有完整注释。

talker 侧：预填只有 3 个位置（[h0+e0, bos+codec_pad, h1+e1+codec_bos]），prompt 只通过 rope 绝对
偏移起作用；codec 采样 temp 0.9 / top-k 40 / top-p 0.8 / 重复惩罚 1.05。Token2Wav 的 DiT 走 ggml 图
（AdaLN-zero、block=24 分块注意力 mask、RoPE），BigVGAN 与 ECAPA 说话人塔是宿主 C++（抗锯齿激活的
分组 FIR 没有 ggml 算子；kaiser 滤波器按 HF 公式现算）。已知差距：语音数值尚未与 HF 对齐（HF 的
token2wav 在 CPU bf16 下有 dtype 问题，参考输出待补），音量偏小，BigVGAN 的 snake/fir 语义值得用
官方 fp32 权重再核对一遍。

量化版：

| 精度 | thinker / talker | 验证（多模态消融，test-1.jpeg + 4 s wav vs HF bf16 ref） | 用法 |
|---|---|---|---|
| f16 | 11.2 GB / 1.66 GB (q8) | logits0 corr 0.9946；音频塔 corr 0.9997；贪心答案与 HF 一致 | 12G 卡可跑（配 q8 talker） |
| q8_0 | 5.64 GB / 1.66 GB | 与 ggml-org 官方 Q8_0 **逐张量 bit 级一致**；logits corr 0.9928（CPU/GPU 均 0.993）；音频塔 corr 0.9992；答案与 f16 相同 | CPU 全功能（含 `--speak`）；12G 卡全功能 |
| q4_0 | 3.63 GB / 1.50 GB | logits corr 0.9475；音频塔 corr 0.9714；答案为同一事实的更短表述 | 6G 卡文本→语音整链；12G 卡全功能 |

语音输出（全部量化档）：token2wav 修复后 wav rms 0.16-0.31（HF 量级 0.08-0.10），
贪心解码对 talker 双方（HF 与本引擎）都是退化解，采样（temp 0.9/top-k 40/top-p 0.8）
下 EOS 正常。e2e 多模态 QA+7.9 s 语音在 12G 卡 54 s（q8，宿主声码器已并行化）。

GPU 上所有"宿主直读权重"的位置（mel 滤波器、BigVGAN/ECAPA/spk）都必须走
`ggml_backend_tensor_get`——GPU 上 `t->data` 是空指针，`tensor_to_f32` 直读必崩。

### EmbeddingGemma-2（四模态嵌入）

```bash
modelscope download google/embeddinggemma-2 --local_dir models/embeddinggemma-2   # 或 hf download
python convert/convert_embedgemma2.py --src models/embeddinggemma-2 -o embeddinggemma2-f16.gguf
```

740M（文本 270M + 视觉 170M + 音频 300M），不生成文本，只出嵌入。要点（都是对拍实测的）：

- **文本塔**是 Gemma3 风格编码器：24 层 d=512、GQA 4:1、SWA(1024)/full 交错、**逐层 PLE**
  （per-layer embedding：12288→512 投影 + RMSNorm，每层一份加到隐状态上）、末尾均值池化 + 投影到 768
- **视觉塔** Gemma4 ViT：16 层 patch16、轴向 2D RoPE（x/y 各 32 维，theta 100）、3×3 位置池化；
  图像预算 280 soft、**视频预算 140/帧**（`vision.video_max_soft_tokens`）；帧间注意力严格不跨帧
- **音频塔** USM conformer：128-bin log-mel → 双 3×3 stride-2 卷积子采样 → 12 层
  （macaron FFN×2 + LightConv1d(GLU + 因果深度卷积) + chunk=12 局部注意力 + softcap 50），
  相对位置偏置是 Transformer-XL 式 rel-shift（每头独立的 rel-k 投影），
  **所有线性层都是 ClippableLinear**（检查点里带 finite input/output clamp 标量，转换器逐个导出）
- **占位块**：`<|image|>`/`<|video|>`/`<|audio|>` 展开成 `<|image>` + N×soft + `<image|>`，
  prompt 里没写标记时媒体块追加在文本后
- **四模态 parity**（fp32 HF 参考，f16 权重）：文本 0.9999999 / 图像 0.99999 /
  音频 0.99999 / 视频 0.9999998。音频塔曾长期停在 0.84——公式（全矩阵 + 13 条对角线 mask 等价
  rel-shift）先用 numpy 验证过是对的，真凶是 5 个 ggml 图实现 bug，逐层 hook dump 定位
  （详见提交 7af56f7）；视频的坑是 rope 位置轴、预处理预算、占位块展开与跨帧注意力（1b8680a）
- Matryoshka：`--embed-dim 512|256|128|64` 截断；`--embed-preset query|document|sts|
  classification|clustering` 套官方 sentence-transformers 任务前缀
- **性能**（f16，3060 Laptop 6GB / i7 8 线程，含 ~1.2s 权重上传）：GPU 文本 1.2s、图像 1.4s、
  30s 音频 1.5s、32 帧视频 3.9s；CPU 图像 4.0s / 32 帧视频 39s——多模态走 GPU 收益 ~10×。
  全路径 flash attention：CUDA fattn 要求 head_dim 256/512 的 KV 长度是 256 的倍数，序列按
  标准做法补齐到 256 倍数、mask 屏蔽补齐位（均值池化只算真实 token），GPU 与 CPU 数值一致
  0.999996+。文本 n_ctx 默认 8192（32 帧视频 ≈3.9k token）。CPU 上 q8_0 权重只快 ~8%
  （瓶颈在 f32 attention，不在权重带宽）

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
                       图像与视频帧解码（stb_image 实现所在）、按架构发现后端、
                       模型发现与单实例驻留（model_registry，Web UI 与 MCP 共用）
  models/moondream/    moondream3.1 后端（引擎 + 视觉塔 + 多 crop 预处理 + region 头 + 服务端/批量）
  models/videochat3/   VideoChat3-4B 后端（视觉预处理 + 计算图）
  models/videomae/     VideoMAE 视频塔后端（clip 预处理 + 全时空注意力前向 + 分类/嵌入）
  models/vjepa2/       V-JEPA 2 后端（三维 RoPE + 全时空注意力 + pooler/分类器/predictor）
  models/qwen3_5/      Qwen3.5 / Marlin / VideoJev 后端（gated-delta + full attention 混合图、caption/find 技能、打分头）
  models/moss_vl/      MOSS-VL-Realtime 后端（self+cross 双路注意力图、多轮/实时流会话）
  models/mage_vl/      Mage-VL 后端（Codec-ViT 视觉塔 + System 1 门控）
  models/qwen2vl/      Qwen2.5-VL 后端（窗口注意力 ViT + 2×2 分组 patch 预处理）
  models/qwen25omni/   Qwen2.5-Omni 后端（thinker/talker 双 GGUF + 宿主 Token2Wav）
  models/smolvlm/      SmolVLM-256M 后端（Idefics3 瓦片预处理）
  models/embedgemma2/  EmbeddingGemma-2 后端（四模态嵌入）
  cli/                 统一前端 vqa-cli（推理 / --serve Web UI / --mcp MCP 服务器）
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
  约为 F16 权重噪声 + 线程归约顺序噪声）；V-JEPA 2 用 `--dump-dir` 导出
  `patches / v_patch_out / v_blk0_qkv / v_blk0_q_rope / v_blk0_attn / v_blk0_attn_out /
  v_blk0_out / v_final / v_pooled`，`VJEPA2_MAX_LAYERS=N` 同理截断层数。
- **共享层（`src/core`）**：后端之间不放重复实现，目录按职责分层，索引见
  [`src/core/README.md`](src/core/README.md)；include 一律从 `src/` 根写
  （`#include "core/engine/weight_load.hpp"`），所以再加子目录也不用改 include 路径：
  - `vqa.hpp` — 唯一放在根上的头：`vqa::Model` 接口与能力表，前端只认它
  - `engine/` — ggml 侧公共件：`backend.hpp`（CUDA 优先 + CPU 回退 + 线程数按**物理核**，
    SMT 对这类模型的 GEMV 是负收益）、`weight_load.hpp`（mmap 绑定、1D 张量转 host F32、
    CPU 零拷贝 / GPU 上传的 buffer 归属）、`gguf_util.hpp`（元数据读取 / 张量转 F32）、
    `ggml_dump.hpp`（对拍导出）、`attention.hpp`（KV cache + mask）、`stage_buf.hpp`（device 侧
    跨图暂存）、`sampler.hpp`
  - `blocks/` — 可复用图体：`vit_block.hpp`（所有 ViT 塔共用的 block 体 + 通用成对旋转 RoPE，
    moondream / VideoChat3 / VideoMAE / ViViT / V-JEPA 2 都用它，差异走 hooks）、
    `llm_block.hpp`（Qwen 系因果解码层，VideoChat3 / SmolVLM / MOSS-VL 已入住，权重字段为空即跳过
    对应 op）、`qwen2vl_tower.hpp`（Qwen2.5-VL 视觉塔整塔共享，qwen2vl 与 qwen25omni 逐 op 一致）
  - `media/` — `resample.hpp`（smart_resize、PIL 语义 antialias bicubic、位置表插值；
    原来寄居在 videochat3 目录被三个模型反向 include）、`media_io.hpp`（图像 / 视频帧解码）、
    `jpeg_decode.hpp`（libjpeg-turbo，与 PIL 逐像素一致）、`audio_feats.hpp`（FFT / Hann / mel /
    log-mel：Qwen2.5-Omni 的 whisper 系与 EmbedGemma-2 的 Gemma4 系两套数值不可互换，禁止跨族
    "统一"浮点行为）、`video_pipe.cpp`（ffmpeg 抽帧）
  - `text/` — `tokenizer`（byte-level BPE + Qwen2 正则预切分）、`utf8`（UTF-8 编解码 + GPT-2
    byte↔unicode 字符表）、`json` 与 `json_scan`（前者递归下降解析器，后者只做 HF 配置的键扫描）、
    `unicode_ranges.inc`（生成文件）
  - `runtime/` — `model_registry`（能跑哪些 GGUF / tokenizer 解析 / 单实例驻留）、`vqa_registry`
    （架构名 → 工厂表，唯一知道后端清单的文件）、`backend_factories`、`load_util`（LoadOptions 落地
    + 各后端 `<PREFIX>_DUMP_DIR` + 特殊 token id）、`vram_guard`、`http_server`（内嵌 Web UI）
  - `util/` — `mmap`（只读映射 + 整文件读取）、`aligned_buf`（64 字节对齐缓冲）
- **加/改共享层前先读**：`src/core/README.md` 里有"新代码该放哪"的判定表和"把重复实现收敛进来"
  的四步流程（两份实现先各自数值一致 → 合并前 diff 掉漂移的守卫 → 新行为默认关闭 → 跑
  `scripts/ref_*.py` + `scripts/smoke.sh`）。
- **改分词器数据**：`scripts/gen_unicode_ranges.py` 重新生成 `src/core/text/unicode_ranges.inc`。
- **回归冒烟**：`scripts/smoke.sh` 跑跨后端金标准电池（smolvlm/vc3/moss/qwen3.5 各一条贪婪，CPU
  + 钉死 `--threads 1` 保证可复现）；`--golden FILE` 对既有金标准逐字节比对。改共享层/做迁移后先跑它。

## License

MIT，见 [LICENSE](LICENSE)。ggml 为 MIT（ggml-org）。模型权重与 tokenizer 遵循其各自条款，单独下载。

