"""Step 5: OmniVoice TTS 模块 — 官方 Python API 语音合成."""

from __future__ import annotations

import os
import time
from typing import Any

import config


ALLOWED_GENDERS = {"male", "female"}
ALLOWED_PITCHES = {"low pitch", "medium pitch", "high pitch"}
ALLOWED_STYLES = {"", "whisper"}
SAMPLE_RATE = 24000


class OmniVoiceTTS:
    """OmniVoice voice-design TTS wrapper with lazy model loading."""

    def __init__(
        self,
        model_path: str = config.OMNIVOICE_MODEL_PATH,
        device_map: str = config.OMNIVOICE_DEVICE,
        dtype: str = config.OMNIVOICE_DTYPE,
        output_dir: str = config.TTS_OUTPUT_DIR,
    ):
        self.model_path = model_path
        self.device_map = device_map
        self.dtype = dtype
        self.output_dir = output_dir
        self._model: Any | None = None

    def _resolve_torch_dtype(self):
        import torch

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(str(self.dtype).lower(), torch.float16)

    def _load_model(self):
        if self._model is not None:
            return self._model

        from omnivoice import OmniVoice

        dtype = self._resolve_torch_dtype()
        self._model = OmniVoice.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            dtype=dtype,
        )
        return self._model

    @staticmethod
    def build_instruct(gender: str, pitch: str, style: str = "") -> str:
        """Build OmniVoice voice-design instruct from validated attributes."""
        normalized_gender = (gender or config.TTS_DEFAULT_GENDER).strip().lower()
        normalized_pitch = (pitch or config.TTS_DEFAULT_PITCH).strip().lower()
        normalized_style = (style or "").strip().lower()

        if normalized_gender not in ALLOWED_GENDERS:
            normalized_gender = config.TTS_DEFAULT_GENDER
        if normalized_pitch not in ALLOWED_PITCHES:
            normalized_pitch = config.TTS_DEFAULT_PITCH
        if normalized_style not in ALLOWED_STYLES:
            normalized_style = ""

        parts = [normalized_gender, normalized_pitch]
        if normalized_style:
            parts.append(normalized_style)
        return ", ".join(parts)

    def generate(
        self,
        text: str,
        gender: str = config.TTS_DEFAULT_GENDER,
        pitch: str = config.TTS_DEFAULT_PITCH,
        style: str = "",
        speed: float = config.TTS_DEFAULT_SPEED,
    ) -> dict[str, Any]:
        """Generate a wav file and return synthesis metadata."""
        if not text or not text.strip():
            raise ValueError("TTS text cannot be empty")

        import soundfile as sf

        os.makedirs(self.output_dir, exist_ok=True)
        instruct = self.build_instruct(gender=gender, pitch=pitch, style=style)
        model = self._load_model()
        audio = model.generate(text=text.strip(), instruct=instruct, speed=float(speed))
        if not audio:
            raise RuntimeError("OmniVoice returned no audio")

        wav_path = os.path.join(self.output_dir, f"tts_{int(time.time() * 1000)}.wav")
        sf.write(wav_path, audio[0], SAMPLE_RATE)
        return {
            "reply_text": text.strip(),
            "instruct": instruct,
            "audio_path": wav_path,
            "gender": gender,
            "pitch": pitch,
            "style": style,
            "speed": float(speed),
        }


if __name__ == "__main__":
    tts = OmniVoiceTTS()
    result = tts.generate(
        text="你好，这是 OmniVoice 语音合成测试。",
        gender=config.TTS_DEFAULT_GENDER,
        pitch=config.TTS_DEFAULT_PITCH,
        style="",
    )
    print(result)
