# 语音助手工作流 — 对接文档

## 项目概述

基于 **Ministral-3-8B-Instruct-2512** + **Qwen3-ASR-0.6B** + **OmniVoice TTS** + **LangChain** 搭建的本地语音助手。支持录音识别 → LLM 对话 → 语音合成全链路，运行在 H200 GPU 上。

## 架构

```
终端 1: vllm serve Ministral-3-8B (port 由 config.VLLM_PORT 控制)  — LLM 服务
终端 2: vllm serve Qwen3-ASR-0.6B (port 8001)                     — ASR 服务

step4_main.py:
  录音 / 音频文件 → vLLM Transcriptions API (port 8001) → 识别文本
                  → LangChain Agent (VLLM_PORT)
                  → synthesize_voice_reply 工具 → OmniVoice TTS（进程内）
                  → 音频文件 (assets/output/tts/)
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `config.py` | 配置文件：模型路径、vLLM 参数、ASR 参数、TTS 参数、录音参数 |
| `requirements.txt` | Python 依赖 |
| `step1_setup_vllm.py` | 启动 LLM vLLM 服务（终端 1） |
| `step2_asr_module.py` | ASR 模块：transformers 后端（文件模式）+ vLLM Transcriptions API（流式模式）|
| `step3_agent_core.py` | LangChain Agent 核心（LLM + synthesize_voice_reply 工具 + 多轮对话）|
| `step4_main.py` | 主入口：录音 → ASR → Agent + TTS → 输出 |
| `step5_tts_module.py` | OmniVoice TTS 模块（语音合成，懒加载）|

## 显存占用预估（H200 80GB）

| 模型 | 显存占用 | 端口 |
|------|---------|------|
| Ministral-3-8B-Instruct-2512 (FP8, vLLM) | ~9 GB | config.VLLM_PORT |
| Qwen3-ASR-0.6B (vLLM, gpu=0.15) | ~2 GB | 8001 |
| OmniVoice TTS (in-process, 懒加载) | ~2 GB | — |
| **合计** | **~13 GB** | — |

## 快速开始

### 1. 配置模型路径

编辑 `config.py`，填写本地模型路径：

```python
LLM_MODEL_PATH = "/your/path/to/Ministral-3-8B-Instruct-2512"
ASR_MODEL_PATH = "/your/path/to/Qwen3-ASR-0.6B"
OMNIVOICE_MODEL_PATH = "/your/path/to/OmniVoice"
```

### 2. 安装依赖

```bash
# 先安装 CUDA 匹配的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu126
# 再安装其余依赖
pip install -r requirements.txt
```

**注意**：流式 ASR 使用 vLLM 0.19.x 原生 Transcriptions API，**无需安装 `qwen-asr[vllm]`**，避免版本冲突。

### 3. 启动服务（两个终端）

```bash
# 终端 1: LLM 服务
python3 step1_setup_vllm.py
# 或: vllm serve <LLM_MODEL_PATH> --port <VLLM_PORT> --tool-call-parser mistral

# 终端 2: ASR 服务
vllm serve <ASR_MODEL_PATH> --port 8001 --gpu-memory-utilization 0.15 --dtype auto
```

### 4. 运行助手

```bash
python3 step4_main.py                # 交互式选择模式
python3 step4_main.py --mode text    # 文本模式 + TTS 合成
python3 step4_main.py --mode voice   # 语音模式（transformers 后端 ASR + TTS）
python3 step4_main.py --mode stream  # 流式 ASR 模式 + TTS（推荐）
```

## 交互模式说明

### 文本模式 (`--mode text`)

直接键入文字 → Agent 回复 + TTS 合成音频。

### 语音模式 (`--mode voice`)

- 输入 `r` → 录音（保存 WAV 文件）→ transformers 后端 ASR → Agent + TTS
- 输入其他文字 → 直接对话 + TTS

### 流式 ASR 模式 (`--mode stream`)

- **按回车** → 录音 → vLLM Transcriptions API → Agent + TTS
- **`f assets/input/test.wav`** → 读取本地音频文件 → Transcriptions API → Agent + TTS
- **输入文字** → 直接对话 + TTS，跳过 ASR
- **`q`** → 退出

## TTS 语音参数

Agent 的 `synthesize_voice_reply` 工具接受以下参数：

| 参数 | 可选值 | 默认值 |
|------|--------|--------|
| `gender` | `male`, `female` | `female` |
| `pitch` | `low pitch`, `medium pitch`, `high pitch` | `medium pitch` |
| `style` | `""`, `whisper` | `""` |

用户可以通过自然语言控制这些参数（如"用男声低沉一点回复我"），LLM 会自动映射。

## 如何添加自定义工具

在 `step3_agent_core.py` 的工具定义区添加 `@tool` 函数（完整类型注解 + docstring），然后加到 `TOOLS` 列表。LLM 会根据用户输入自动判断是否调用。

### 工具编写规范

- 必须有 `@tool` 装饰器
- 参数必须有完整类型注解
- docstring 要写清楚功能（LLM 依据此判断调用）
- 返回值统一 `str`
- 自行处理异常，不向上抛出

## 当前已注册工具

| 工具名 | 功能 |
|--------|------|
| `synthesize_voice_reply` | 合成语音回复（OmniVoice TTS），生成文本 + 音频文件 |

## 关键设计决策

1. **双 vLLM 服务架构**：LLM 和 ASR 各自独立 vLLM 实例，避免版本冲突
2. **TTS 进程内懒加载**：OmniVoice 首次调用时加载，无需独立服务
3. **强制 TTS 路由**：System prompt 要求 LLM 必须通过 `synthesize_voice_reply` 工具回复，保证每条回复都有语音输出
4. **ASR 双后端设计**：transformers 后端用于文件模式（进程内加载），vLLM API 后端用于流式模式（HTTP 调用）
5. **config.py 统一配置**：所有路径和参数集中管理，方便适配不同环境

## 实验推进历程

### Round 1: 初始实现

- **动机**: 搭建 Ministral-3-8B + Qwen3-ASR + LangChain 的语音助手原型
- **方案**: vLLM 服务化部署 LLM，ASR 独立模块，LangChain Agent 带 tool calling
- **结果**: 完成四步代码框架，支持语音/文本两种交互模式
- **待验证**: 需填写实际模型路径后端到端测试

### Round 2: 流式 ASR 模式

- **动机**: 原有语音模式需要保存 WAV 文件再识别，流程较重；希望支持流式推理和直接上传音频文件
- **方案**: 为 ASRModule 新增 vLLM 后端（`Qwen3ASRModel.LLM()`），通过流式 API 实现增量识别
- **问题**: `qwen-asr[vllm]` 依赖 vLLM 0.14.0，与已安装的 vLLM 0.19.1 冲突（GitHub Issue #119）
- **结果**: 代码已编写但无法运行

### Round 3: 双 vLLM 服务架构

- **动机**: 解决 `qwen-asr[vllm]` 与 vLLM 0.19.1 的版本冲突
- **方案**: 利用 vLLM 0.19.x 原生 Qwen3-ASR 支持，在独立端口（8001）启动 ASR 服务，通过 OpenAI 兼容 Transcriptions API 识别音频
- **关键变化**: ASR 从 `qwen-asr` Python API 改为 vLLM Transcriptions API（HTTP）；无需 `qwen-asr[vllm]`
- **结果**: 端到端测试通过

### Round 4: OmniVoice TTS 集成（当前）

- **动机**: 实现完整的语音对话闭环：ASR 输入 → LLM 理解 → TTS 语音输出
- **方案**: 新增 `step5_tts_module.py`（OmniVoice 懒加载），`step3_agent_core.py` 注册 `synthesize_voice_reply` 工具，System prompt 强制 LLM 每次回复都走 TTS 工具
- **关键变化**: 移除旧工具（weather/calculator 等），只保留 `synthesize_voice_reply`；`step4_main.py` 调用改为 `chat_with_tts`
- **下一步**: 端到端测试全链路
