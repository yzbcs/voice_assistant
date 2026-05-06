"""Step 2: Qwen3-ASR 语音识别模块

提供 transcribe(audio_path) 接口，将音频文件转为文本。

用法:
    from step2_asr_module import ASRModule

    asr = ASRModule()              # 加载模型
    text = asr.transcribe("test.wav")  # 识别音频
    print(text)

支持的音频格式: wav, mp3, flac
"""

from __future__ import annotations

import os
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

    加载本地 Qwen3-ASR 模型，提供音频转文本接口。

    初始化参数:
        model_path: 模型本地路径，默认从 config.ASR_MODEL_PATH 读取
        device: 推理设备，默认 "cuda"
        language: 识别语言，默认从 config.ASR_LANGUAGE 读取
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda", language: str | None = None):
        self.model_path = model_path or config.ASR_MODEL_PATH
        self.device = device
        self.language = self._normalize_language(language or config.ASR_LANGUAGE)

        if "/path/to/" in self.model_path:
            print(f"[ASR] WARNING: 请先在 config.py 中设置 ASR_MODEL_PATH")
            print(f"  当前值: {self.model_path}")
            self.model = None
            return

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
        """通过 qwen-asr 官方接口加载 Qwen3-ASR 模型"""
        import torch
        from qwen_asr import Qwen3ASRModel

        print(f"[ASR] 正在加载模型: {self.model_path}")
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
