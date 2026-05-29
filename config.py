"""语音助手配置文件 — 模型路径和运行参数"""

import os

# ============================================================
# 模型路径（请根据实际本地路径修改）
# ============================================================
LLM_MODEL_PATH = "/path/to/Ministral-3-8B-Instruct-2512"   # TODO: 填写实际路径
OMNIVOICE_MODEL_PATH = "/path/to/OmniVoice"                  # TODO: 填写实际路径，或使用 "k2-fsa/OmniVoice"

# ============================================================
# Mega-ASR 配置
# ============================================================
MEGA_ASR_REPO_DIR = "/path/to/Mega-ASR"                              # TODO: 填写 Mega-ASR 仓库实际路径
MEGA_ASR_CKPT_DIR = "/path/to/Mega-ASR/ckpt/Mega-ASR"                # TODO: 填写 checkpoint 实际路径
MEGA_ASR_ROUTING_ENABLED = True       # 是否启用音频质量路由器
MEGA_ASR_QUALITY_THRESHOLD = 0.5      # 路由器判定阈值

# ASR 基础模型路径（由 Mega-ASR checkpoint 自动解析）
ASR_MODEL_PATH = os.path.join(MEGA_ASR_CKPT_DIR, "Qwen3-ASR-1.7B")

# ============================================================
# vLLM 服务配置
# ============================================================
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8010
VLLM_GPU_MEMORY_UTILIZATION = 0.35  # Jetson 统一内存: 0.35 × 122GB ≈ 43GB，够 Ministral-3-8B
VLLM_MAX_MODEL_LEN = 8192
VLLM_SERVED_MODEL_NAME = ""  # 留空则使用 os.path.basename(LLM_MODEL_PATH)

# ============================================================
# ASR 配置
# ============================================================
ASR_SAMPLE_RATE = 16000
ASR_LANGUAGE = "zh"                  

# ============================================================
# OmniVoice TTS 配置
# ============================================================
OMNIVOICE_DEVICE = "cuda:0"          # Apple Silicon 可改为 "mps"，CPU 可改为 "cpu"
OMNIVOICE_DTYPE = "float16"
TTS_OUTPUT_DIR = "assets/output/tts"

TTS_DEFAULT_GENDER = "female"
TTS_DEFAULT_PITCH = "moderate pitch"
TTS_DEFAULT_SPEED = 1.0

# ============================================================
# Agent 配置
# ============================================================
SYSTEM_PROMPT = "你是一个智能语音助手。根据用户意图给出简洁、自然的中文回复。"

# ============================================================
# ASR vLLM 服务配置（vLLM 0.19.x 原生支持，无需 qwen-asr[vllm]）
# ============================================================
ASR_VLLM_HOST = "0.0.0.0"
ASR_VLLM_PORT = 8001
ASR_VLLM_GPU_MEMORY_UTILIZATION = 0.12  # Jetson 统一内存: 0.12 × 122GB ≈ 15GB，够 Mega-ASR 1.7B
ASR_SERVED_MODEL_NAME = ""  # ASR 服务注册的模型名，启动后 curl 查看，如 "./model/Qwen3-ASR-0.6B"

# ============================================================
# 流式 ASR 配置
# ============================================================
ASR_STREAMING = True           # 是否启用流式 ASR（False 则退回文件模式）
STREAM_SAMPLE_RATE = 16000     # 流式 ASR 采样率
ASR_CHUNK_SIZE_SEC = 2.0       # 每次送入 ASR 的音频段长度（秒）

# ============================================================
# 录音配置
# ============================================================
RECORD_SAMPLE_RATE = 16000
RECORD_CHANNELS = 1
RECORD_DURATION = 5  # 默认录音频长（秒），0 表示手动停止
