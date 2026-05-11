"""Step 2: Qwen3-ASR 语音识别模块

提供两种识别模式:
  1. 文件模式: transcribe(audio_path) — 从文件路径识别（transformers 后端）
  2. API 模式: transcribe_via_api(audio_path_or_array) — 通过 vLLM Transcriptions API 识别

vLLM 0.19.x 原生支持 Qwen3-ASR，无需 qwen-asr[vllm] 包。

用法:
    from step2_asr_module import ASRModule

    # 文件模式（transformers 后端，需在进程内加载模型）
    asr = ASRModule()
    text = asr.transcribe("test.wav")

    # API 模式（vLLM Transcriptions API，需先启动 ASR 服务）
    asr = ASRModule(streaming=True)
    text = asr.transcribe_via_api("test.wav")
    text = asr.transcribe_audio_array(audio_ndarray)

启动 ASR 服务:
    vllm serve /path/to/Qwen3-ASR-0.6B --port 8001 --gpu-memory-utilization 0.15 --dtype auto
"""

from __future__ import annotations

import io
import os
import struct
import tempfile
import wave

import numpy as np

import config


TRANSFORMERS_LANGUAGE_ALIASES = {
    "auto": None,
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "中文": "Chinese",
    "en": "English",
    "english": "English",
    "英文": "English",
}

API_LANGUAGE_ALIASES = {
    "auto": None,
    "zh": "zh",
    "cn": "zh",
    "chinese": "zh",
    "中文": "zh",
    "en": "en",
    "english": "en",
    "英文": "en",
}


class ASRModule:
    """Qwen3-ASR 语音识别模块

    支持两种后端:
      - transformers 后端 (streaming=False): from_pretrained + transcribe(file)
      - vLLM API 后端 (streaming=True): 通过 Transcriptions API 识别（需先启动 ASR vLLM 服务）

    初始化参数:
        model_path: 模型本地路径，默认从 config.ASR_MODEL_PATH 读取
        device: 推理设备，默认 "cuda"
        language: 识别语言，默认从 config.ASR_LANGUAGE 读取
        streaming: 是否使用 vLLM API 后端，默认从 config.ASR_STREAMING 读取
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda",
                 language: str | None = None, streaming: bool | None = None):
        self.model_path = model_path or config.ASR_MODEL_PATH
        self.device = device
        raw_language = language or config.ASR_LANGUAGE
        self.language = self._normalize_transformers_language(raw_language)
        self.api_language = self._normalize_api_language(raw_language)
        self.streaming = streaming if streaming is not None else config.ASR_STREAMING
        self.model = None
        self.client = None
        self.model_name = None

        if "/path/to/" in self.model_path:
            print(f"[ASR] WARNING: 请先在 config.py 中设置 ASR_MODEL_PATH")
            print(f"  当前值: {self.model_path}")
            return

        if self.streaming:
            self._init_vllm_client()
        else:
            self._load_model()

    @staticmethod
    def _normalize_transformers_language(language: str | None) -> str | None:
        """转换为 qwen-asr transformers 后端接受的完整语言名。"""
        if language is None:
            return None
        normalized = str(language).strip()
        if not normalized:
            return None
        return TRANSFORMERS_LANGUAGE_ALIASES.get(normalized.lower(), normalized)

    @staticmethod
    def _normalize_api_language(language: str | None) -> str | None:
        """转换为 vLLM Transcriptions API 接受的语言代码。"""
        if language is None:
            return None
        normalized = str(language).strip()
        if not normalized:
            return None
        return API_LANGUAGE_ALIASES.get(normalized.lower(), normalized)

    def _language_code(self) -> str | None:
        """返回语言代码供 API 使用。"""
        return self.api_language

    def _load_model(self):
        """通过 qwen-asr transformers 后端加载模型（文件模式）"""
        import torch
        from qwen_asr import Qwen3ASRModel

        print(f"[ASR] 正在加载模型 (transformers): {self.model_path}")
        device_map = "cuda:0" if self.device == "cuda" else self.device
        dtype = torch.bfloat16 if str(device_map).startswith("cuda") else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=1,
            max_new_tokens=512,
        )
        print(f"[ASR] 模型加载完成，设备: {device_map}, language: {self.language or 'auto'}")

    def _init_vllm_client(self):
        """初始化 vLLM Transcriptions API 客户端

        使用 OpenAI 兼容客户端连接到 vLLM ASR 服务。
        需要先在另一终端启动: vllm serve <model_path> --port 8001
        """
        from openai import OpenAI

        host = config.ASR_VLLM_HOST
        port = config.ASR_VLLM_PORT
        self.model_name = getattr(config, "ASR_SERVED_MODEL_NAME", None) or os.path.basename(self.model_path.rstrip(os.sep))

        # 客户端连接用 localhost（与 step3 Agent 一致），服务端绑定用 0.0.0.0
        connect_host = "localhost" if host == "0.0.0.0" else host

        print(f"[ASR] 初始化 vLLM API 客户端: http://{connect_host}:{port}/v1")
        print(f"[ASR] 模型名: {self.model_name}")
        print(f"[ASR] 请确保已在另一终端启动 ASR 服务:")
        print(f"  vllm serve {self.model_path} --host {host} --port {port} --gpu-memory-utilization {config.ASR_VLLM_GPU_MEMORY_UTILIZATION} --dtype auto")

        self.client = OpenAI(
            base_url=f"http://{connect_host}:{port}/v1",
            api_key="EMPTY",
        )

        # 检查服务是否在线
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"[ASR] 服务在线，可用模型: {model_ids}")
            if self.model_name not in model_ids:
                print(f"[ASR] WARNING: 模型名 '{self.model_name}' 不在服务端列表中")
                print(f"  服务端模型: {model_ids}")
                if model_ids:
                    self.model_name = model_ids[0]
                    print(f"  自动切换为: {self.model_name}")
        except Exception as e:
            print(f"[ASR] WARNING: 无法连接 ASR 服务 ({e})")
            print(f"  请先启动: vllm serve {self.model_path} --port {port}")
            self.client = None

    def transcribe(self, audio_path: str) -> str:
        """将音频文件转为文本（transformers 后端）

        参数:
            audio_path: 音频文件路径，支持 wav/mp3/flac

        返回:
            识别出的文本字符串
        """
        if self.model is None:
            return "[ERROR] ASR 模型未加载，请检查 config.ASR_MODEL_PATH"

        if not os.path.exists(audio_path):
            return f"[ERROR] 音频文件不存在: {audio_path}"

        results = self.model.transcribe(
            audio=audio_path,
            language=self.language,
        )
        if not results:
            return ""

        result = results[0] if isinstance(results, list) else results
        if hasattr(result, "text"):
            return result.text.strip()
        if isinstance(result, dict) and "text" in result:
            return str(result["text"]).strip()
        return str(result).strip()

    def transcribe_via_api(self, audio_path: str) -> str:
        """通过 vLLM Transcriptions API 识别音频文件

        参数:
            audio_path: 音频文件路径

        返回:
            识别出的文本字符串
        """
        if self.client is None:
            return "[ERROR] ASR API 客户端未初始化"

        if not os.path.exists(audio_path):
            return f"[ERROR] 音频文件不存在: {audio_path}"

        try:
            with open(audio_path, "rb") as f:
                transcription = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=f,
                    language=self._language_code(),
                )
            return transcription.text.strip()
        except Exception as e:
            return f"[ERROR] ASR API 调用失败: {e}"

    def transcribe_audio_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将 numpy 音频数组通过 vLLM Transcriptions API 转为文本

        将 numpy 数组按 config.ASR_CHUNK_SIZE_SEC 切块后，通过 API 分段发送。

        参数:
            audio: PCM 音频数据（int16 或 float32），单声道
            sample_rate: 音频采样率，会被重采样到 16kHz

        返回:
            识别出的文本字符串
        """
        return self.transcribe_audio_stream(audio, sample_rate=sample_rate)

    def transcribe_audio_stream(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将 numpy 音频数组切成多个 chunk，通过 vLLM API 顺序识别。"""
        if self.client is None:
            return "[ERROR] ASR API 客户端未初始化"

        try:
            normalized_audio = self._normalize_audio(audio, sample_rate)
            if len(normalized_audio) == 0:
                return ""

            texts = []
            for chunk in self._iter_audio_chunks(normalized_audio):
                text = self.transcribe_audio_chunk(chunk, sample_rate=config.STREAM_SAMPLE_RATE)
                if text.startswith("[ERROR]"):
                    return text
                if text:
                    texts.append(text)
            return " ".join(texts).strip()
        except Exception as e:
            return f"[ERROR] ASR API 调用失败: {e}"

    def transcribe_audio_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """识别单个音频 chunk。chunk 会被标准化并编码为 WAV bytes。"""
        if self.client is None:
            return "[ERROR] ASR API 客户端未初始化"

        try:
            normalized_audio = self._normalize_audio(audio, sample_rate)
            if len(normalized_audio) == 0:
                return ""

            wav_bytes = self._encode_wav(normalized_audio, config.STREAM_SAMPLE_RATE)
            transcription = self.client.audio.transcriptions.create(
                model=self.model_name,
                file=("audio.wav", wav_bytes, "audio/wav"),
                language=self._language_code(),
            )
            return transcription.text.strip()
        except Exception as e:
            return f"[ERROR] ASR API 调用失败: {e}"

    @staticmethod
    def _normalize_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """标准化音频为 STREAM_SAMPLE_RATE、单声道、int16。"""
        if audio is None:
            return np.array([], dtype=np.int16)

        audio = np.asarray(audio)
        if audio.size == 0:
            return np.array([], dtype=np.int16)

        # 单声道
        if audio.ndim > 1:
            audio = audio[:, 0]

        # 重采样到 16kHz
        if sample_rate != config.STREAM_SAMPLE_RATE:
            import scipy.signal
            if audio.dtype in (np.int16, np.int32):
                audio = audio.astype(np.float32)
                if np.max(np.abs(audio)) > 1.0:
                    audio = audio / 32768.0
            audio = scipy.signal.resample_poly(audio, config.STREAM_SAMPLE_RATE, sample_rate).astype(np.float32)

        # 转 int16
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            if np.max(np.abs(audio)) <= 1.0:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        return audio

    @staticmethod
    def _iter_audio_chunks(audio: np.ndarray):
        """按 ASR_CHUNK_SIZE_SEC 生成音频 chunk。"""
        chunk_samples = max(1, int(config.ASR_CHUNK_SIZE_SEC * config.STREAM_SAMPLE_RATE))
        for start in range(0, len(audio), chunk_samples):
            chunk = audio[start:start + chunk_samples]
            if len(chunk) > 0:
                yield chunk

    @staticmethod
    def _encode_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """将 int16 numpy 数组编码为 WAV 格式的 bytes

        参数:
            audio: int16 单声道音频数组
            sample_rate: 采样率

        返回:
            WAV 文件的 bytes
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()


def quick_transcribe(audio_path: str, model_path: str | None = None) -> str:
    """快捷函数：单次识别音频文件

    适合测试和简单使用场景。内部每次都会加载模型，
    如果需要多次识别，建议使用 ASRModule 类避免重复加载。

    参数:
        audio_path: 音频文件路径
        model_path: 模型路径（可选，默认从 config 读取）

    返回:
        识别出的文本
    """
    asr = ASRModule(model_path=model_path)
    return asr.transcribe(audio_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python3 step2_asr_module.py <audio_path> [model_path]")
        print("  语音模式: 使用 transformers 后端直接识别")
        print("  API 模式: 需先启动 vllm serve ASR 服务")
        sys.exit(1)

    audio = sys.argv[1]
    mp = sys.argv[2] if len(sys.argv) > 2 else None
    result = quick_transcribe(audio, mp)
    print(f"[ASR Result] {result}")
