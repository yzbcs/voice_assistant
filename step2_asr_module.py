"""Step 2: Qwen3-ASR 语音识别模块

提供两种识别模式:
  1. 文件模式: transcribe(audio_path) — 从文件路径识别（transformers 后端）
  2. 流式模式: transcribe_audio_array(audio) — 从 numpy 数组流式识别（vLLM 后端）

用法:
    from step2_asr_module import ASRModule

    # 文件模式
    asr = ASRModule()
    text = asr.transcribe("test.wav")

    # 流式模式
    asr = ASRModule(streaming=True)
    text = asr.transcribe_audio_array(audio_ndarray)

支持的音频格式: wav, mp3, flac
"""

from __future__ import annotations

import os

import numpy as np

import config


LANGUAGE_ALIASES = {
    "auto": None,
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "中文": "Chinese",
    "en": "English",
    "english": "English",
    "英文": "English",
}


class ASRModule:
    """Qwen3-ASR 语音识别模块

    支持两种后端:
      - transformers 后端 (streaming=False): from_pretrained + transcribe(file)
      - vLLM 后端 (streaming=True): Qwen3ASRModel.LLM + 流式推理

    初始化参数:
        model_path: 模型本地路径，默认从 config.ASR_MODEL_PATH 读取
        device: 推理设备，默认 "cuda"
        language: 识别语言，默认从 config.ASR_LANGUAGE 读取
        streaming: 是否使用 vLLM 流式后端，默认从 config.ASR_STREAMING 读取
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda",
                 language: str | None = None, streaming: bool | None = None):
        self.model_path = model_path or config.ASR_MODEL_PATH
        self.device = device
        self.language = self._normalize_language(language or config.ASR_LANGUAGE)
        self.streaming = streaming if streaming is not None else config.ASR_STREAMING
        self.model = None

        if "/path/to/" in self.model_path:
            print(f"[ASR] WARNING: 请先在 config.py 中设置 ASR_MODEL_PATH")
            print(f"  当前值: {self.model_path}")
            return

        if self.streaming:
            self._load_streaming_model()
        else:
            self._load_model()

    @staticmethod
    def _normalize_language(language: str | None) -> str | None:
        """转换为 qwen-asr 官方 transcribe 接口接受的语言名。"""
        if language is None:
            return None
        normalized = str(language).strip()
        if not normalized:
            return None
        return LANGUAGE_ALIASES.get(normalized.lower(), normalized)

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

    def _load_streaming_model(self):
        """通过 qwen-asr vLLM 后端加载模型（流式模式）

        使用 Qwen3ASRModel.LLM() 初始化，支持 init_streaming_state + streaming_transcribe。
        gpu_memory_utilization=0.3 对 0.6B 模型足够，避免和 LLM 的 vLLM 争显存。
        """
        from qwen_asr import Qwen3ASRModel

        print(f"[ASR] 正在加载模型 (vLLM streaming): {self.model_path}")
        self.model = Qwen3ASRModel.LLM(
            model=self.model_path,
            gpu_memory_utilization=0.3,
            max_inference_batch_size=1,
            max_new_tokens=512,
        )
        print(f"[ASR] vLLM 流式模型加载完成, language: {self.language or 'auto'}")

    def transcribe(self, audio_path: str) -> str:
        """将音频文件转为文本

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
            language=self.language,  # None=自动检测；若自动检测效果差，强制指定如 "Chinese"
        )
        if not results:
            return ""

        result = results[0] if isinstance(results, list) else results
        if hasattr(result, "text"):
            return result.text.strip()
        if isinstance(result, dict) and "text" in result:
            return str(result["text"]).strip()
        return str(result).strip()

    def transcribe_audio_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将 numpy 音频数组通过 vLLM 流式 ASR 转为文本

        流程: init_streaming_state → 分段 streaming_transcribe → finish_streaming_transcribe
        每送入一个 chunk 都会增量更新 state.text，最终返回完整识别结果。

        参数:
            audio: PCM 音频数据（int16 或 float32），单声道
            sample_rate: 音频采样率，会被重采样到 16kHz

        返回:
            识别出的文本字符串
        """
        if self.model is None:
            return "[ERROR] 流式 ASR 模型未加载"

        # 统一转 float32 并 normalize 到 [-1, 1]
        wav = audio.astype(np.float32)
        if wav.dtype == np.float32 and np.max(np.abs(wav)) > 1.0:
            wav = wav / 32768.0

        # 重采样到 16kHz
        if sample_rate != 16000:
            import scipy.signal
            wav = scipy.signal.resample_poly(wav, 16000, sample_rate)

        # 流式推理：分段送入 ASR
        chunk_sec = config.ASR_CHUNK_SIZE_SEC
        state = self.model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=chunk_sec,
        )
        step = int(16000 * chunk_sec)
        pos = 0
        while pos < wav.shape[0]:
            seg = wav[pos : pos + step]
            self.model.streaming_transcribe(seg, state)
            pos += step
            if pos < wav.shape[0]:
                print(f"\r[ASR 流式] 当前识别: {state.text}", end="", flush=True)

        self.model.finish_streaming_transcribe(state)
        print()  # 换行
        return state.text.strip()


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
        sys.exit(1)

    audio = sys.argv[1]
    mp = sys.argv[2] if len(sys.argv) > 2 else None
    result = quick_transcribe(audio, mp)
    print(f"[ASR Result] {result}")
