"""语音助手配置文件 — 模型路径和运行参数"""

# ============================================================
# 模型路径（请根据实际本地路径修改）
# ============================================================
LLM_MODEL_PATH = "/path/to/Ministral-3-8B-Instruct-2512"   # TODO: 填写实际路径
ASR_MODEL_PATH = "/path/to/Qwen3-ASR"                        # TODO: 填写实际路径
OMNIVOICE_MODEL_PATH = "/path/to/OmniVoice"                  # TODO: 填写实际路径，或使用 "k2-fsa/OmniVoice"

# ============================================================
# vLLM 服务配置
# ============================================================
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8010
VLLM_GPU_MEMORY_UTILIZATION = 0.85
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
TTS_DEFAULT_PITCH = "medium pitch"
TTS_DEFAULT_SPEED = 1.0

# ============================================================
# Agent 配置
# ============================================================
SYSTEM_PROMPT = (
    "你是一个智能语音助手。每次用户提问，你只能调用 synthesize_voice_reply 工具一次，"
    "绝对不要重复调用，不要直接把回复写在普通消息里。"
    "根据用户意图写出简洁、自然的中文 reply_text，再选择语音设计参数。"
    "gender 只能是 male 或 female；pitch 只能是 low pitch、medium pitch 或 high pitch；"
    "style 只能是空字符串或 whisper。"
    "如果用户明确要求男声、女声、高音、低音、耳语等，请映射到对应字段；"
    "否则使用默认、自然的声音。"
)

# ============================================================
# ASR vLLM 服务配置（vLLM 0.19.x 原生支持，无需 qwen-asr[vllm]）
# ============================================================
ASR_VLLM_HOST = "0.0.0.0"
ASR_VLLM_PORT = 8001
ASR_VLLM_GPU_MEMORY_UTILIZATION = 0.15  # 0.6B 模型足够
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
