# 端到端语音助手：技术调研与实现流程

## 项目概述

本项目构建了一个端到端的中文语音助手框架，核心流程为 **语音输入 → ASR 语音识别 → LLM Agent 对话决策 → TTS 语音合成 → 语音输出**。系统采用双 vLLM 服务架构（LLM + ASR 各自独立部署），结合进程内 TTS 模型，实现了完整的语音交互闭环。

```
用户语音 ──→ [ASR: Qwen3-ASR-0.6B] ──→ 文本
                                          │
                                          ▼
                                    [Agent: Ministral-3-8B]
                                    (LangChain + LangGraph ReAct)
                                          │
                                   synthesize_voice_reply
                                          │
                                          ▼
                                   [TTS: OmniVoice] ──→ 音频输出
```

---

## 一、ASR 语音识别调研与选型

### 1.1 选型考虑

语音识别（ASR）是语音助手的第一环，需要满足以下要求：

| 需求维度 | 要求 |
|---------|------|
| 语言支持 | 中文为主，兼顾英文，支持自动语言检测 |
| 延迟 | 流式场景需低延迟，支持实时分块识别 |
| 部署方式 | 本地 GPU 部署，不依赖云 API |
| 模型大小 | 足够轻量，可与 LLM 共存于同一 GPU |

### 1.2 候选方案总览

共调研 6 个方案，涵盖开源模型与闭源商业服务：

| 模型 | 开发团队 | 参数量 | 架构 | 语言数 | 中文 WER (AISHELL-1) | 延迟 | 显存需求 | 开源 |
|------|---------|--------|------|--------|---------------------|------|---------|------|
| **Qwen3-ASR-0.6B** | 阿里 Qwen | 0.6B | AR 自回归 | 52 (含22方言) | ~3.x% (1.7B: 2.71% AISHELL-2) | 中等 | ~6GB | ✅ |
| SenseVoice-Small | 阿里 FunAudioLLM | ~240M | NAR 非自回归 | 50+ | 优秀 (优于Whisper) | **70ms**/10s | ~2GB | ✅ |
| Fun-ASR-Nano | 阿里通义 | 800M | E2E 端到端 | 31 | **1.80%** | 中等 | 中等 | ✅ |
| Moonshine | Moonshine AI | 34M~245M | NAR+流式 | 8 | 一般 (英文为主) | **34ms**~107ms | 极低 | ✅ |
| Step-Audio | 阶跃星辰 | 130B+3B | 理解生成一体 | 中/英/日 | 1.95% | 高 (130B) | **265GB** (4×A800) | ✅ |
| 科大讯飞 | 科大讯飞 | 闭源 | 闭源 | 100+ | 业界领先 | 毫秒级 | 云端 | ❌ |

### 1.3 候选方案详细分析

#### Qwen3-ASR（阿里 Qwen 团队）

中文识别精度最强，WER 大幅领先 Whisper：

| 数据集 | Qwen3-ASR-1.7B | Whisper-Large-v3 | GPT-4o-Transcribe |
|--------|----------------|------------------|-------------------|
| WenetSpeech-net | **4.97%** | 9.86% | 15.30% |
| WenetSpeech-meeting | **5.88%** | 19.11% | 32.27% |
| AISHELL-2 | **2.71%** | 5.06% | 4.24% |
| SpeechIO | **2.88%** | 7.56% | 12.86% |
| Fleurs-zh | **2.41%** | 4.09% | 2.44% |

- 0.6B 版本吞吐量 2000x @ 128 并发，vLLM 加速
- 22 种中文方言（粤语 WER 3.98% vs Whisper 9.18%）
- 提供 0.6B / 1.7B 两个版本

#### SenseVoice-Small（阿里 FunAudioLLM）

延迟极低的非自回归模型，额外提供情感识别和事件检测：

| 模型 | 参数量 | 10s音频推理 |
|------|--------|-----------|
| SenseVoice-Small | ~240M | **70ms** |
| Whisper-Small | ~244M | ~350ms (5x) |
| Whisper-Large | ~1.5B | ~1050ms (15x) |

- 7 种情感识别 (HAPPY/SAD/ANGRY/NEUTRAL/FEARFUL/DISGUSTED/SURPRISED)
- 事件检测 (BGM/掌声/笑声/哭声等)
- ONNX / sherpa-onnx / SenseVoice.cpp 多平台部署

#### Fun-ASR-Nano（阿里通义）

方言和口音支持最全面的模型：

| 数据集 | Fun-ASR-Nano | Whisper-Large-v3 | GLM-ASR-nano |
|--------|-------------|------------------|-------------|
| AISHELL-1 | **1.80%** | 4.72% | 1.81% |
| Fleurs-zh | **2.56%** | 5.18% | - |
| Librispeech-clean | **1.76%** | 1.86% | 2.00% |

- 7 种方言 + 26 种地区口音
- 歌词识别、说唱识别、远场高噪声场景专项优化

#### Moonshine（Moonshine AI）

边缘设备首选，极致低延迟：

| 模型 | 参数量 | WER | MacBook Pro | 树莓派 5 |
|------|--------|------|-------------|---------|
| Moonshine Medium Streaming | 245M | **6.65%** | 107ms | 802ms |
| Moonshine Small Streaming | 123M | **7.84%** | 73ms | 527ms |
| Moonshine Tiny Streaming | 34M | 12.00% | **34ms** | **237ms** |
| Whisper Large v3 | 1.5B | 7.44% | 11,286ms | N/A |

- 原生流式、增量缓存，无 Whisper 的固定 30s 窗口限制
- 覆盖 Python/iOS/Android/macOS/Windows/Linux/树莓派
- 中文支持较弱（无方言），仅 8 种语言

#### Step-Audio（阶跃星辰）

全栈语音系统（ASR + TTS + 对话），但资源需求极高：

| 能力 | 表现 |
|------|------|
| ASR (AISHELL-1) | 1.95% WER |
| TTS (中文 CER) | **1.17%** (优于 GLM-4-Voice 2.19%) |
| 语音对话 (StepEval-360) | Chat Score **4.11** (GLM-4-Voice: 3.49) |

- 130B 参数需 4×A800 (265GB)，资源门槛过高
- 支持语音克隆、情感控制、RAP/哼唱、ToolCall

#### 科大讯飞（闭源商业）

中文语音市场领导者，但依赖云端：

| 维度 | 科大讯飞 | 开源模型 |
|------|---------|---------|
| 中文精度 | 业界领先 | 优秀 |
| 部署 | 云端 API（数据上传） | 本地部署 |
| 成本 | 按量付费 | 一次性 |
| 定制化 | 有限 | 完全可控 |
| 隐私 | 数据上传云端 | 本地 |

### 1.4 最终选型：Qwen3-ASR-0.6B

**选择理由：**

1. **中文识别精度最优**：WER 全面领先 Whisper-Large-v3（如 WenetSpeech 4.97% vs 9.86%），与 GPT-4o-Transcribe 对比也大幅胜出
2. **方言覆盖广**：22 种中文方言，粤语/四川话等识别能力远超竞品
3. **vLLM 原生支持**：vLLM 0.19.x 提供原生 Transcriptions API，0.6B 版本 2000x 吞吐量 @ 128 并发
4. **轻量高效**：0.6B 参数仅需 15% GPU 显存（~1.2GB），与 LLM 共存无压力
5. **流式友好**：支持音频分块识别（默认 2s/chunk），满足实时交互需求

**落选原因简述：**
- SenseVoice：延迟极低但需 VAD 才能流式，且无专门方言优化
- Fun-ASR-Nano：方言最强但 800M 参数偏大，vLLM 兼容性不如 Qwen3-ASR
- Moonshine：英文场景极致低延迟，但中文支持弱，仅 8 种语言
- Step-Audio：功能最全但 130B 需 4×A800，资源门槛远超项目条件
- 科大讯飞：精度最高但闭源云端，不满足本地部署和隐私要求

### 1.5 ASR 情绪/语气识别能力对比

语音助手若要根据用户语气调整回复风格（如用户愤怒时温柔安抚），需要 ASR 在输出文本的同时识别语音中的情绪。以下对比各候选方案的这方面能力：

| 模型 | 情绪/语气识别 | 输出格式 | 说明 |
|------|-------------|---------|------|
| **SenseVoice-Small** | ✅ **7种情感** | 文本标签嵌入识别结果 | HAPPY/SAD/ANGRY/NEUTRAL/FEARFUL/DISGUSTED/SURPRISED，另有事件检测（笑声/哭声/掌声/BGM 等） |
| **科大讯飞** | ✅ 情感分析 | API 返回结构化数据 | 闭源云端服务，商业级情感分析 |
| **Step-Audio** | ⚠️ 部分支持 | 130B 多模态理解 | 能理解语音中的情感，但它是端到端对话模型而非纯 ASR，资源门槛极高 |
| Qwen3-ASR | ❌ | 仅文本 | 纯 ASR，无情感识别 |
| Fun-ASR-Nano | ❌ | 仅文本 | 纯 ASR，无情感识别 |
| Moonshine | ❌ | 仅文本 | 有意图识别 (Intent Recognition) 但非情感识别 |

**SenseVoice-Small 情感识别示例：**

输出会将情感标签和事件标签直接嵌入转录文本中：

```
<|HAPPY|>今天天气真好啊<|Speech|><|Laughter|>
```

通过 `rich_transcription_postprocess()` 可以提取这些标签，供下游 Agent 使用。

**SenseVoice 支持的情感类型：**

| 类别 | 标签 |
|------|------|
| 情感 (SER) | HAPPY (开心)、SAD (悲伤)、ANGRY (生气)、NEUTRAL (中性)、FEARFUL (恐惧)、DISGUSTED (厌恶)、SURPRISED (惊讶) |
| 事件 (AED) | BGM (背景音乐)、Speech (语音)、Applause (掌声)、Laughter (笑声)、Cry (哭声)、Sneeze (喷嚏)、Breath (呼吸)、Cough (咳嗽) |

**后续扩展方向：** 若项目需要"识别用户语气 → 调整 TTS 回复语气"的能力，有两条路径：

1. **换用 SenseVoice-Small 做 ASR**：自带情感识别，Agent 可直接根据情感标签选择对应的 TTS voice-design 参数（如用户 SAD → 回复用 gentle/warm 语气）
2. **保持 Qwen3-ASR + 外挂情感模型**：ASR 不换，额外加一个 SER 模型并行推理，解耦但增加系统复杂度

### 1.6 实现架构

ASR 模块提供双后端设计，通过 `streaming` 参数切换：

```
ASRModule
├── transformers 后端 (streaming=False)
│   └── qwen-asr 库直接加载模型，进程内推理
│       优点：无外部依赖，离线可用
│       缺点：加载慢，占用进程内显存
│
└── vLLM API 后端 (streaming=True)  ← 推荐
    └── 独立 vLLM 服务，OpenAI 兼容客户端调用
        优点：流式分块、GPU 独立管理、HTTP 解耦
        缺点：需额外终端启动服务
```

**流式识别流程：**

```
麦克风录音 → sounddevice 回调采集
    → 按 2s chunk 累积 (pending_audio + enqueue_ready_chunks)
    → 音频队列 (audio_queue)
    → 后台 worker 线程逐块调用 vLLM Transcriptions API
    → 拼接所有 chunk 识别结果
```

**音频预处理链路：**

```
原始音频 → 单声道提取 → 重采样至 16kHz → float→int16 转换 → WAV 编码 → API 发送
```

### 1.7 关键配置参数

| 参数 | 值 | 说明 |
|------|----|------|
| `ASR_MODEL_PATH` | 本地 Qwen3-ASR-0.6B 路径 | 模型文件目录 |
| `ASR_VLLM_PORT` | 8001 | ASR vLLM 服务端口 |
| `ASR_VLLM_GPU_MEMORY_UTILIZATION` | 0.15 | GPU 显存占用比 |
| `ASR_SAMPLE_RATE` | 16000 | 音频采样率 |
| `ASR_CHUNK_SIZE_SEC` | 2.0 | 流式分块大小（秒） |
| `ASR_LANGUAGE` | zh | 默认识别语言 |

---

## 二、Agent 框架选取

### 2.1 选型考虑

Agent 是语音助手的"大脑"，负责理解用户意图并生成回复。核心要求：

| 需求维度 | 要求 |
|---------|------|
| LLM 能力 | 支持中文对话 + 工具调用（Function Calling） |
| Agent 框架 | 支持工具注册、多轮对话、可扩展 |
| 推理方式 | 可通过 vLLM 本地部署 |
| 扩展性 | 方便后续添加新工具（如天气查询、日程管理等） |

### 2.2 LLM 选型对比

| 模型 | 参数量 | 中文对话 | 工具调用 | vLLM 兼容 | 备注 |
|------|--------|---------|---------|-----------|------|
| **Ministral-3-8B-Instruct** | 8B | 良好 | 原生支持 | 原生支持 | Mistral 出品，3B/8B 均可 |
| Qwen2.5-7B-Instruct | 7B | 优秀 | 原生支持 | 原生支持 | 阿里出品，中文首选 |
| Llama-3.1-8B-Instruct | 8B | 一般 | 支持 | 原生支持 | Meta 出品，中文偏弱 |
| ChatGLM4-9B | 9B | 优秀 | 支持 | 需适配 | 清华出品，部署稍复杂 |

> 当前选用 Ministral-3-8B-Instruct-2512，后续可无缝切换为其他模型（只需修改 `config.py` 中的 `LLM_MODEL_PATH`）。

### 2.3 Agent 框架选型对比

| 框架 | 优势 | 劣势 |
|------|------|------|
| **LangChain + LangGraph** | 生态丰富、工具注册简洁、ReAct 模式成熟、社区活跃 | 抽象层较重 |
| 原生 OpenAI SDK | 最轻量、无额外依赖 | 需手动实现 Agent 循环、工具路由 |
| LlamaIndex | 擅长 RAG 场景 | Agent 能力偏弱 |
| AutoGen | 多 Agent 协作 | 架构复杂，单 Agent 场景过重 |

### 2.4 最终方案：LangChain + LangGraph ReAct Agent

**选择理由：**

1. **一行创建 Agent**：`create_react_agent(llm, tools, prompt)` 即可完成 Agent 构建
2. **工具注册简洁**：`@tool` 装饰器 + 类型注解即可定义工具，LLM 根据文档字符串自动选择
3. **多轮对话**：内置 `chat_history` 管理，支持上下文连续对话
4. **可扩展**：添加新工具只需编写 `@tool` 函数并加入 `TOOLS` 列表

### 2.5 核心设计：单工具强制路由

系统 Prompt 强制 LLM **只能调用 `synthesize_voice_reply` 工具**，确保每次回复都经过 TTS 合成：

```
System Prompt 要点:
├── 每次回复必须且只能调用 synthesize_voice_reply 一次
├── 禁止直接文字回复（不经过 TTS）
├── 根据用户意图选择语音参数（性别/音高/风格）
└── 语音参数映射规则（如"男声"→gender=male, "低沉"→pitch=low pitch）
```

**工具参数设计：**

| 参数 | 类型 | 可选值 | 说明 |
|------|------|--------|------|
| `reply_text` | str | 自由文本 | LLM 生成的回复内容 |
| `gender` | str | male / female | 语音性别 |
| `pitch` | str | low pitch / moderate pitch / high pitch | 音高 |
| `style` | str | "" / whisper | 声音风格 |

### 2.6 Agent 处理流程

```
用户输入 (文本)
    │
    ▼
VoiceAssistant.chat_with_tts(user_input)
    │
    ├── 构造消息: chat_history + HumanMessage
    │
    ├── agent.invoke() → LangGraph ReAct 循环
    │   ├── LLM 决策: 调用 synthesize_voice_reply 工具
    │   └── 返回 response_messages
    │
    ├── 提取最后一次工具调用参数
    │   └── _extract_last_tool_call() → {reply_text, gender, pitch, style}
    │
    ├── 调用 TTS 生成音频 (一次)
    │   └── OmniVoiceTTS.generate(text, gender, pitch, style)
    │
    ├── Fallback: 若 LLM 未调用工具，用原始回复 + 默认语音参数
    │
    └── 返回 {reply_text, audio_path, instruct, raw_agent_reply}
```

### 2.7 关键配置参数

| 参数 | 值 | 说明 |
|------|----|------|
| `LLM_MODEL_PATH` | Ministral-3-8B-Instruct-2512 | LLM 模型路径 |
| `VLLM_PORT` | 8010 | LLM vLLM 服务端口 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.85 | GPU 显存占用比 |
| `VLLM_MAX_MODEL_LEN` | 8192 | 最大上下文长度 |
| `temperature` | 0.15 | 生成温度（低温度保证稳定输出） |
| `max_tokens` | 2048 | 单次最大生成 token 数 |

---

## 三、TTS 语音合成选型

### 3.1 选型考虑

TTS 是语音助手的最后一环，将文本回复转化为语音输出。核心要求：

| 需求维度 | 要求 |
|---------|------|
| 语音质量 | 自然度高、可懂度好 |
| 声音可控 | 支持性别、音高、风格等参数调节 |
| 中文支持 | 中文合成效果好 |
| 部署方式 | 本地 GPU 部署 |

### 3.2 候选方案对比

| 模型 | 中文质量 | 声音可控 | 部署方式 | 模型大小 | 备注 |
|------|---------|---------|---------|---------|------|
| **OmniVoice** | 优秀 | voice-design（性别/音高/风格/语速） | 本地 GPU | 中等 | k2-fsa 团队出品，支持 instruct 控制 |
| CosyVoice | 优秀 | 支持 | 本地 GPU | 较大 | 阿里出品，依赖复杂 |
| ChatTTS | 良好 | 有限 | 本地 GPU | 中等 | 开源社区，中文效果好 |
| VITS / Bert-VITS2 | 良好 | 有限 | 本地 GPU | 较小 | 需要单独训练音色 |
| Edge-TTS | 良好 | 固定音色 | 云 API | 无需本地 | 依赖网络，延迟不可控 |

### 3.3 最终选型：OmniVoice

**选择理由：**

1. **Voice-Design 机制**：通过自然语言 instruct 控制 voice（如 "female, high pitch, whisper"），无需录制参考音频
2. **中文质量优秀**：k2-fsa 团队（Kaldi/Sherpa 作者）出品，中文合成自然度高
3. **参数化控制**：支持 `gender`、`pitch`、`style`、`speed` 四维调节，与 Agent 工具参数完美对应
4. **简洁 API**：`OmniVoice.from_pretrained()` + `model.generate(text, instruct, speed)` 即可完成合成
5. **本地部署**：单 GPU 即可运行，float16 精度，24kHz 采样率输出

### 3.4 Voice-Design Instruct 格式

OmniVoice 使用逗号分隔的 instruct 字符串控制语音属性：

```
instruct = "{gender}, {pitch}[, {style}]"

示例:
  "female, moderate pitch"           → 女声，中等音高
  "male, low pitch"                  → 男声，低沉
  "female, high pitch, whisper"      → 女声，高音，耳语风格
```

| 属性 | 可选值 | 说明 |
|------|--------|------|
| gender | male / female | 性别 |
| pitch | low pitch / moderate pitch / high pitch | 音高 |
| style | "" / whisper | 风格（空字符串为正常，whisper 为耳语） |
| speed | 浮点数（默认 1.0） | 语速倍率 |

### 3.5 TTS 实现细节

```
OmniVoiceTTS
├── 懒加载：首次调用 generate() 时才加载模型
│   └── OmniVoice.from_pretrained(path, device_map, dtype)
│
├── 参数校验：build_instruct() 验证 gender/pitch/style 合法性
│   └── 不合法参数自动回退到默认值
│
├── 合成流程：
│   model.generate(text, instruct, speed)
│   → audio tensor → soundfile.write(wav_path, audio, 24000)
│
└── 输出：
    └── assets/output/tts/tts_{timestamp}.wav
        + 返回 {reply_text, instruct, audio_path, gender, pitch, style, speed}
```

### 3.6 关键配置参数

| 参数 | 值 | 说明 |
|------|----|------|
| `OMNIVOICE_MODEL_PATH` | 本地 OmniVoice 路径 / "k2-fsa/OmniVoice" | 模型路径或 HuggingFace ID |
| `OMNIVOICE_DEVICE` | cuda:0 | 推理设备（Apple Silicon 可用 mps） |
| `OMNIVOICE_DTYPE` | float16 | 推理精度 |
| `TTS_DEFAULT_GENDER` | female | 默认性别 |
| `TTS_DEFAULT_PITCH` | moderate pitch | 默认音高 |
| `TTS_DEFAULT_SPEED` | 1.0 | 默认语速 |
| 输出采样率 | 24000 Hz | WAV 输出质量 |

---

## 四、端到端实现流程

### 4.1 系统部署架构

```
┌─────────────────────────────────────────────────────┐
│                    GPU 服务器                          │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ vLLM Server  │  │ vLLM Server  │  │  In-process │ │
│  │ LLM (8B)     │  │ ASR (0.6B)   │  │  TTS        │ │
│  │ Port: 8010   │  │ Port: 8001   │  │  (OmniVoice)│ │
│  │ GPU: 85%     │  │ GPU: 15%     │  │  GPU: 共享   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬─────┘ │
│         │                 │                  │        │
│         └────────┬────────┘──────────────────┘        │
│                  │                                    │
│         step4_main.py (主进程)                         │
│         ├── ASRModule (HTTP Client → ASR Server)      │
│         ├── VoiceAssistant (HTTP Client → LLM Server) │
│         └── OmniVoiceTTS (进程内调用)                   │
└─────────────────────────────────────────────────────┘
```

### 4.2 启动流程

```bash
# Step 1: 启动 LLM vLLM 服务
python3 step1_setup_vllm.py
# → Ministral-3-8B-Instruct-2512, port 8010, GPU 85%

# Step 2: 启动 ASR vLLM 服务（独立终端）
vllm serve /path/to/Qwen3-ASR-0.6B --port 8001 --gpu-memory-utilization 0.15 --dtype auto

# Step 3: 运行语音助手
python3 step4_main.py --mode stream   # 推荐：流式 ASR 模式
python3 step4_main.py --mode voice    # 离线 ASR 模式
python3 step4_main.py --mode text     # 纯文本模式（跳过 ASR）
```

### 4.3 三种交互模式

| 模式 | ASR 后端 | 输入方式 | 特点 |
|------|---------|---------|------|
| `stream` | vLLM API | 实时录音 / 音频文件 | 推荐，低延迟，分块流式识别 |
| `voice` | transformers | 固定时长录音 | 离线推理，无需额外服务 |
| `text` | 无 | 键盘输入 | 纯文本对话 + TTS 输出，调试用 |

### 4.4 一次完整交互的数据流

以流式模式为例，一次"用户说话 → 助手语音回复"的完整数据流：

```
1. [录音] 用户按回车开始/停止录音
   └── sounddevice 回调采集 → 16kHz int16 单声道 PCM

2. [流式 ASR] 音频按 2s 分块 → vLLM Transcriptions API
   └── POST /v1/audio/transcriptions (每 chunk 一个请求)
   └── 拼接所有 chunk 识别结果 → 完整文本

3. [Agent] 文本 → LangGraph ReAct Agent
   └── LLM 推理: 分析意图 → 决定调用 synthesize_voice_reply
   └── 工具参数: {reply_text: "加油，你做得很棒！", gender: "female", pitch: "high pitch", style: ""}

4. [TTS] OmniVoice 合成
   └── instruct: "female, high pitch"
   └── model.generate(text="加油，你做得很棒！", instruct="female, high pitch", speed=1.0)
   └── 输出: assets/output/tts/tts_{timestamp}.wav (24kHz WAV)

5. [输出] 返回给用户
   └── reply_text: "加油，你做得很棒！"
   └── audio_path: "assets/output/tts/tts_1716000000000.wav"
   └── instruct: "female, high pitch"
```

### 4.5 代码模块职责

| 文件 | 职责 | 核心类/函数 |
|------|------|-----------|
| `config.py` | 全局配置（模型路径、端口、参数） | 所有常量 |
| `step1_setup_vllm.py` | 启动 LLM vLLM 服务 | 子进程管理 |
| `step2_asr_module.py` | ASR 语音识别 | `ASRModule` |
| `step3_agent_core.py` | LLM Agent + TTS 调度 | `VoiceAssistant`, `synthesize_voice_reply` |
| `step4_main.py` | CLI 主入口（录音/交互/流程编排） | `main()`, 三种模式函数 |
| `step5_tts_module.py` | TTS 语音合成 | `OmniVoiceTTS` |

### 4.6 依赖关系

```
pip install torch  # 先装 CUDA 版 PyTorch

# requirements.txt 核心依赖:
qwen-asr              # ASR transformers 后端
vllm==0.19.1          # LLM + ASR 推理服务
mistral-common>=1.8.6 # Mistral tokenizer

langchain>=0.3        # Agent 框架
langchain-openai>=0.3 # OpenAI 兼容客户端
langgraph             # ReAct Agent 图执行引擎

omnivoice             # TTS 模型
soundfile             # WAV 输出
sounddevice           # 麦克风录音
scipy                 # 音频重采样
```

> **注意**：不要安装 `qwen-asr[vllm]`，它会与 vLLM 0.19.1 冲突。流式 ASR 使用 vLLM 原生 Transcriptions API 替代。

---

## 五、设计亮点与后续方向

### 5.1 设计亮点

1. **双 vLLM 服务解耦**：LLM 和 ASR 独立部署，互不干扰，可独立扩缩容
2. **单工具强制路由**：System Prompt + 唯一工具确保每次回复都经过 TTS，实现完整语音闭环
3. **懒加载 TTS**：OmniVoice 仅在首次需要时加载，节省启动时间和显存
4. **Fallback 容错**：LLM 未调用工具时自动使用默认语音参数，保证系统鲁棒性
5. **流式优先设计**：2s 分块 + 后台 worker 线程，实现实时录音识别

### 5.2 后续优化方向

| 方向 | 内容 |
|------|------|
| 更多工具 | 添加天气查询、日程管理、知识问答等工具到 Agent |
| 流式 TTS | 边生成边播放，降低端到端延迟 |
| VAD 集成 | 语音活动检测，自动判断用户说话开始/结束 |
| 多轮优化 | 压缩长对话历史，避免上下文溢出 |
| 评测体系 | ASR WER 测试、Agent 意图准确率、TTS MOS 评分 |
