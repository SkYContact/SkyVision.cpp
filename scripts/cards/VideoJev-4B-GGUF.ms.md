---
license: apache-2.0
base_model: MCG-NJU/VideoJev-4B
base_model_relation: quantized
library_name: gguf
pipeline_tag: video-text-to-text
language:
- en
- zh
tags:
- vision
- video
- gguf
- temporal-grounding
- video-question-answering
- decision-head
- hybrid-attention
- skyvision.cpp
---

# VideoJev-4B GGUF

[VideoJev-4B](https://modelscope.cn/models/MCG-NJU/VideoJev-4B)（Qwen3.5-4B 微调：32 层混合主干 + 24 层视觉塔，
**用训练出来的 cosine 决策头替代 lm head**）的社区 GGUF 量化版，由自研推理引擎
[SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp)（纯 ggml，无 llama.cpp 依赖）转换并逐级对拍验证。

> **这是一个打分模型，不是生成模型。** 它不能"聊天"：官方把末层的 MLP 与 post-attention-norm 都删了，
> 也没有 lm_head，整条路径是"一次前向 → 对候选列表逐项打分"。
> 驱动方式为 `vqa-cli --score questions.json`，不是 `--ask`。
>
> **仅供 [SkyVision.cpp](https://github.com/SkYContact/SkyVision.cpp) 使用。** GGUF 里
> `general.architecture = videojev`，且 llama.cpp 的接口里没有"候选打分/任意位置隐状态读出"这一层，
> **llama.cpp / ollama / LM Studio 等通用 GGUF 运行时无法加载或使用本仓库文件**。

## 量化档位

| 文件 | 大小 | 说明 |
|---|---|---|
| `videojev-4b-q8_0.gguf` | 4.73 GiB (5.08 GB) | 视觉塔 F16 保留，决策头 FP32；CPU 841 token 前向 ≈ 9.8 s |
| `tokenizer.json` | 19 MB | 官方检查点的分词器，本仓库一并附带，不用再单独下 |

## 模型做了什么

| 任务 | 输出 |
|---|---|
| `grounding`（时序定位） | 每个时间片（temporal patch = 2 帧）一个 sigmoid 概率 + 一个区间（阈值 0.25，可覆盖） |
| `choice`（单选视频问答） | 对选项做 softmax + `answer_index` / `answer` |

内部机制与官方实现逐条对齐：候选/查询位置（`<|option_end|>` / `<|decide|>`）的隐状态各过一层线性投影
（2560→512，FP32），归一化后做 cosine × 10 + bias。

## 使用

```bash
git clone https://github.com/SkYContact/SkyVision.cpp
cd SkyVision.cpp && git clone --depth 1 https://github.com/ggml-org/ggml
scripts\build-cuda.bat

# 权重与 tokenizer 都在本仓库里
modelscope download SkyContact/VideoJev-4B-GGUF --local_dir models/VideoJev-4B-GGUF

cat > qs.json <<'JSON'
{"questions": [{"id": "q1", "task": "grounding", "query": "when does the person sit down"},
               {"id": "q2", "task": "choice", "query": "where does it happen",
                "options": ["A. indoors", "B. outdoors"]}]}
JSON

vqa-cli --model videojev-4b-q8_0.gguf --tokenizer models/VideoJev-4B-GGUF \
        --video clip.mp4 --fps 2 --score qs.json
# [{"id":"q1",...,"probabilities":[...],"times":[...],"start":..,"end":..},
#  {"id":"q2",...,"probabilities":[...],"answer_index":1,"answer":"B. outdoors"}]
```

`--threshold` 覆盖模型自带阈值；`--fps` 决定采样密度（时间片 = 2 帧，官方默认 2 fps）。

## 精度验证（vs 官方 transformers 5.14 参考实现）

- **prompt token 序列逐位一致**：794 tokens（含 chat template 视频占位符外层保留的
  `<|vision_start|>` / `<|vision_end|>`），候选/查询索引也一致
- 以 CPU **float32** 参考为真值，8 秒合成视频的时序定位曲线：

  | | patch0 | patch3（变化点附近） | patch7 | 最大偏差 |
  |---|---|---|---|---|
  | f32 参考 | 0.000408 | 0.075201 | 0.024380 | — |
  | 本仓库 q8_0 | 0.000397 | 0.081081 | 0.021828 | **5.9e-3** |
  | 引擎 f16 | 0.000401 | 0.081979 | 0.023538 | 6.8e-3 |
  | 官方默认 bf16 | 0.000406 | 0.018292 | 0.034594 | 5.7e-2 |

  即量化版比官方仓库自身的 bf16 跑法更贴近 fp32；choice 端偏差 6.6e-6（官方 bf16 为 5.1e-5）。
- 线性注意力的 **16 键头 / 32 值头 GQA** 已在转换期按 ggml 的 tile 语义重排；消融（不重排 / 反序）会让
  打分曲线退化成一片 0.1~0.4 的糊状，可作为该步骤的必要性证据。

## 已知限制

- 只能打分，不能生成；`--ask` 对它无意义
- 一次前向只处理一个视频 + 一组问题（官方同样如此），候选数与分辨率受 `--n-ctx` 约束
- 官方在 GPU 上跑 bf16；本项目 CPU 前向 841 token ≈ 9.8 s（4070 SUPER 主机的 19 线程 CPU）
