# 语音助手实验 — 方案与启动流程

> 平台: Jetson Thor (ARM64, 122GB 统一内存) · Docker + NVIDIA runtime

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│ Docker 容器 — vLLM 0.21rc1                                   │
│                                                              │
│  vLLM serve Ministral-3-8B, port 8010                        │
│  gpu-mem 0.35 (≈43GB)                                        │
│                                                              │
│  vLLM serve Mega-ASR (materialized), port 8001               │
│  gpu-mem 0.12 (≈15GB)                                        │
│                                                              │
│  step4_main.py --mode stream                                 │
│  录音完成 → 提交完整音频 → vLLM ASR 流式返回 token            │
│         → Agent(LLM) → TTS(OmniVoice) → 播放                 │
└──────────────────────────────────────────────────────────────┘
```

### 为什么只需要一个容器

| 之前的问题 | 现在的解法 |
|-----------|-----------|
| qwen-asr[vllm] 锁定 vLLM==0.14.0 | vLLM 0.21 **原生支持** Qwen3-ASR 架构，不需要 qwen-asr 包 |
| 需要 qwen-asr[vllm] 才能用 vLLM 推理 | Mega-ASR materialized checkpoint 保留原始 `model_type: "qwen3_asr"`，vLLM 0.21 直接识别 |
| vLLM 0.14 不支持 Thor Blackwell 架构 | 只用 vLLM 0.21，完全兼容 Thor |

### 内存分配

| 组件 | gpu_memory_utilization | 实际占用 (×122GB) |
|------|----------------------|-------------------|
| LLM (Ministral-3-8B) | 0.35 | ≈43GB |
| ASR (Mega-ASR 1.7B materialized) | 0.12 | ≈15GB |
| **合计** | **0.47** | **≈58GB** |
| 剩余系统可用 | — | **≈64GB** |

---

## 2. ASR 工作方式

### 当前方案：提交完整音频 + token 流式输出

```
录音中 ●●●●●● (完整音频，用户按回车停止)
         ↓ 提交完整 wav 到 vLLM ASR (port 8001)
vLLM 处理音频编码 + 解码
         ↓ Chat Completions stream: true
文字逐 token 返回: "你" → "你好" → "你好请问" → "你好请问今天天气怎么样"
```

- **Transcriptions API**（`/v1/audio/transcriptions`）：提交完整音频 → 返回完整识别文字
- **Chat Completions + `stream: true`**（`/v1/chat/completions`）：提交完整音频 → 文字 token 逐个流式返回

### 真流式为什么不可用

真流式需要 qwen-asr 的 in-process vLLM 后端（`init_streaming_state` / `streaming_transcribe` / `finish_streaming_transcribe`），这依赖 `qwen-asr[vllm]` → 锁定 vLLM==0.14.0 → 不支持 Thor Blackwell 架构。

| | 真流式 (不可用) | 当前方案 (可用) |
|---|---|---|
| **输入** | 边录音边送增量音频 | 录完一次性提交 |
| **KV cache** | 复用，增量推理 | 无，一次推理 |
| **输出** | 边录边出文字 | 提交后文字逐 token 流出 |
| **延迟** | 实时 | 录音时长 + 处理时间 |

### stream 和 voice 模式对比

| 模式 | ASR 后端 | 推理方式 | 输出方式 |
|------|---------|---------|---------|
| **stream** | vLLM serve (port 8001) | materialized checkpoint | token 流式 (Chat Completions stream) |
| **voice** | transformers (进程内) | LoRA 动态路由 | 一次性返回完整文字 |

---

## 3. 完整启动流程

### Step 0: 配置 config.py（首次必须）

```python
LLM_MODEL_PATH = "/models/Ministral-3-8B-Instruct-2512"
OMNIVOICE_MODEL_PATH = "/models/OmniVoice"
MEGA_ASR_REPO_DIR = "/Mega-ASR-main"
MEGA_ASR_CKPT_DIR = "/asr-ckpt/Mega-ASR"
```

### Step 1: 合并 LoRA 权重（首次，可在本地完成）

```bash
python3 step2_asr_module.py --materialize
# 输出: <MEGA_ASR_CKPT_DIR>/mega-asr-vllm-materialized/
```

### Step 2: 启动 Docker 容器

```bash
docker run -it --runtime=nvidia --network=host \
  -v /path/to/models:/models \
  -v /path/to/exp1_voice_assistant:/workspace \
  -v /path/to/Mega-ASR/ckpt:/asr-ckpt \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor
```

### Step 3: 启动 LLM 服务

```bash
# 容器内 Terminal 1
python3 step1_setup_vllm.py
# 等待 "[vLLM] 服务就绪！"
```

### Step 4: 启动 ASR 服务

```bash
# 容器内 Terminal 2
python3 -m vllm serve <MEGA_ASR_CKPT_DIR>/mega-asr-vllm-materialized \
  --port 8001 \
  --gpu-memory-utilization 0.12 \
  --dtype auto \
  --max-model-len 8192 \
  --max-num-seqs 1
# 等待 "Application startup complete"
```

### Step 5: 启动主程序

```bash
# 容器内 Terminal 3
python3 step4_main.py --mode stream
```

### 交互流程

```
[You] (按回车开始录音)
[录音] ● 录音中... (按回车停止)
[录音] (按回车停止)
[ASR] 你 → 你好 → 你好请问今天天气怎么样  ← token 逐个流出
[Agent] 今天北京晴，最高温度28度...
[TTS] 音频已生成: assets/output/tts/tts_xxx.wav
[播放] ▶ 正在播放...
```

---

## 4. 三种模式对比

| 模式 | 命令 | ASR 后端 | 识别延迟 | 特点 |
|------|------|----------|---------|------|
| **text** | `--mode text` | 无 | — | 最简，键盘输入 |
| **voice** | `--mode voice` | transformers (进程内) | ~1-3s | LoRA 动态路由 |
| **stream** | `--mode stream` | vLLM serve (port 8001) | 录音时长 + 处理 | token 流式输出 |

---

## 5. 依赖环境

### 容器 (vLLM 0.21rc1)

```
# Docker 镜像已包含: vllm 0.21rc1, torch (CUDA)
# 额外安装:
mistral-common>=1.8.6
langchain>=0.3
langchain-openai>=0.3
langchain-community>=0.3
langgraph
transformers>=4.56.0,<5.0.0
accelerate
numpy
sounddevice
soundfile
scipy
prompt_toolkit
wcwidth
openai
safetensors
peft
omnivoice
```

### 注意

- **不装** `qwen-asr` — vLLM 0.21 原生支持 Qwen3-ASR 架构，ASR 推理走 vLLM serve
- **不装** `qwen-asr[vllm]` — 与 vLLM 0.21 冲突
- `omnivoice` 要求 `transformers>=5.3.0`，`vllm 0.21` 要求 `transformers>=4.56.0, !=5.0-5.5.0`，两者兼容

---

## 6. TTS 限制

OmniVoice **不支持流式输出**：必须等 LLM 输出完整文本 → TTS 合成完整音频 → 才能播放。

端到端延迟链:
```
ASR: 录音时长 + ~0.5-2s (token 流式，第一个字更快出现)
LLM 推理:   ~2-5s (完整回复)
TTS 合成:   ~3-8s (完整音频)    ← 瓶颈，不支持流式
播放:       即时
```

**未来优化方向**: LLM 流式输出 → 分句 → 逐句 TTS → 拼接播放。

---

## 7. 实验推进历程

### Round 1: 初始方案设计
- **动机**: 在 Jetson Thor 上搭建端到端语音助手
- **方案**: Ministral-3-8B (LLM) + Qwen3-ASR-0.6B (ASR) + OmniVoice (TTS)
- **问题**: vLLM 默认 gpu_memory_utilization=0.9 吃掉 110GB/122GB 内存

### Round 2: 升级 Mega-ASR
- **动机**: 提升识别准确率，使用更大的模型 + LoRA
- **方案**: Qwen3-ASR-1.7B + LoRA + 音频质量路由器
- **问题**: qwen-asr[vllm] 锁定 vLLM==0.14，与 Docker vLLM 0.21 冲突
- **发现**: stream 模式的"流式"是分块 HTTP 伪流式，无 KV cache 复用

### Round 3: Jetson Thor 适配
- **动机**: 统一内存架构下显存管理
- **方案**: LLM gpu-mem 0.35 (43GB) + ASR gpu-mem 0.12 (15GB) = 58GB，留 64GB 给系统

### Round 4: 真流式方案（已放弃）
- **动机**: 实现实时 ASR 识别，边录边出文字
- **方案**: 双容器隔离 — 容器 1 (vLLM 0.21 + LLM) + 容器 2 (vLLM 0.14 + qwen-asr[vllm])
- **失败原因**: vLLM 0.14 不支持 Thor Blackwell (SM100) 架构

### Round 5: 单容器方案（当前）
- **动机**: 简化架构，去掉不可行的双容器方案
- **发现**: vLLM 0.21 原生支持 `Qwen3ASRForConditionalGeneration` 架构（day-0 model support），不需要 qwen-asr 包
- **方案**: 单容器 vLLM 0.21，同时 serve LLM (port 8010) 和 ASR materialized checkpoint (port 8001)
- **ASR 输出**: Chat Completions + `stream: true`，提交完整音频后文字 token 逐个流式返回
- **trade-off**: 放弃真流式（增量音频 + KV cache），但架构大幅简化
