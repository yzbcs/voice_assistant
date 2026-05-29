"""Step 2: Mega-ASR 语音识别模块

基于 Mega-ASR（Qwen3-ASR-1.7B + LoRA + 音频质量路由器）。

提供两种识别模式:
  1. Transformers 模式: transcribe(audio_path) — MegaASR 类，支持 LoRA 动态路由
  2. vLLM API 模式: transcribe_via_api(audio_path) — 通过 vLLM Chat Completions API（需预先合并 LoRA 权重）

用法:
    from step2_asr_module import ASRModule

    # Transformers 模式（进程内加载模型，支持音频质量路由器）
    asr = ASRModule()
    text = asr.transcribe("test.wav")

    # vLLM API 模式（需先启动 ASR 服务）
    asr = ASRModule(streaming=True)
    text = asr.transcribe_via_api("test.wav")

启动 ASR vLLM 服务:
    # 先一次性合并 LoRA 权重:
    python3 step2_asr_module.py --materialize
    # 然后用标准 vllm serve 启动:
    vllm serve <ckpt>/mega-asr-vllm-materialized --port 8001 --gpu-memory-utilization 0.30 --dtype auto
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import wave

import numpy as np

import config


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


def _import_mega_asr():
    """导入 MegaASR 类，自动将 Mega-ASR 仓库加入 sys.path。"""
    repo_dir = config.MEGA_ASR_REPO_DIR
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from MegaASR.model.megaASR import MegaASR
    return MegaASR


def materialize_mega_asr():
    """一次性操作：将 LoRA 合并到基础权重，生成 vLLM 可用的 checkpoint。

    输出目录: <MEGA_ASR_CKPT_DIR>/mega-asr-vllm-materialized/
    """
    MegaASR = _import_mega_asr()
    ckpt_dir = config.MEGA_ASR_CKPT_DIR
    base_path = os.path.join(ckpt_dir, "Qwen3-ASR-1.7B")
    lora_path = os.path.join(ckpt_dir, "mega-asr-merged")
    materialized_dir = os.path.join(ckpt_dir, "mega-asr-vllm-materialized")

    print(f"[Mega-ASR] 开始合并 LoRA 权重...")
    print(f"  基础模型: {base_path}")
    print(f"  LoRA 权重: {lora_path}")
    print(f"  输出目录: {materialized_dir}")

    model = MegaASR(
        model_path=base_path,
        lora_dir=lora_path,
        routing_enabled=False,
        backend="vllm",
        vllm_apply_lora_on_load=True,
        vllm_materialized_lora_dir=materialized_dir,
    )
    print(f"[Mega-ASR] LoRA 合并完成: {materialized_dir}")
    return materialized_dir


class ASRModule:
    """Mega-ASR 语音识别模块

    支持两种后端:
      - Transformers 后端 (streaming=False): MegaASR 类，支持 LoRA 动态路由
      - vLLM API 后端 (streaming=True): 通过 Chat Completions API（需先合并 LoRA 并启动 vLLM 服务）

    初始化参数:
        model_path: 基础模型路径，默认从 config.ASR_MODEL_PATH 读取
        device: 推理设备，默认 "cuda"
        language: 识别语言，默认从 config.ASR_LANGUAGE 读取
        streaming: 是否使用 vLLM API 后端，默认从 config.ASR_STREAMING 读取
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda",
                 language: str | None = None, streaming: bool | None = None):
        self.model_path = model_path or config.ASR_MODEL_PATH
        self.device = device
        self.language = language or config.ASR_LANGUAGE
        self.api_language = self._normalize_api_language(self.language)
        self.streaming = streaming if streaming is not None else config.ASR_STREAMING
        self.model = None
        self.client = None
        self.model_name = None

        if "/path/to/" in config.MEGA_ASR_REPO_DIR:
            print(f"[ASR] WARNING: 请先在 config.py 中设置 MEGA_ASR_REPO_DIR 和 MEGA_ASR_CKPT_DIR")
            print(f"  MEGA_ASR_REPO_DIR 当前值: {config.MEGA_ASR_REPO_DIR}")
            return

        if self.streaming:
            self._init_vllm_client()
        else:
            self._load_model()

    @staticmethod
    def _normalize_api_language(language: str | None) -> str | None:
        if language is None:
            return None
        normalized = str(language).strip()
        if not normalized:
            return None
        return API_LANGUAGE_ALIASES.get(normalized.lower(), normalized)

    def _language_code(self) -> str | None:
        return self.api_language

    def _load_model(self):
        """通过 Mega-ASR transformers 后端加载模型，支持 LoRA 动态路由。"""
        MegaASR = _import_mega_asr()

        ckpt_dir = config.MEGA_ASR_CKPT_DIR
        base_path = os.path.join(ckpt_dir, "Qwen3-ASR-1.7B")
        lora_path = os.path.join(ckpt_dir, "mega-asr-merged")
        router_path = os.path.join(ckpt_dir, "audio_quality_router", "best_acc_model.safetensors")
        device_map = f"{self.device}:0" if self.device == "cuda" else self.device

        print(f"[Mega-ASR] 正在加载模型 (transformers)")
        print(f"  基础模型: {base_path}")
        print(f"  LoRA: {lora_path}")
        print(f"  路由器: {router_path}")
        print(f"  路由: {'启用' if config.MEGA_ASR_ROUTING_ENABLED else '禁用'}")

        self.model = MegaASR(
            model_path=base_path,
            lora_dir=lora_path,
            router_checkpoint=router_path,
            routing_enabled=config.MEGA_ASR_ROUTING_ENABLED,
            quality_threshold=config.MEGA_ASR_QUALITY_THRESHOLD,
            device_map=device_map,
        )
        print(f"[Mega-ASR] 模型加载完成，设备: {device_map}")

    def _init_vllm_client(self):
        """初始化 vLLM Chat Completions API 客户端

        使用 OpenAI 兼容客户端连接到 vLLM ASR 服务。
        通过 Chat Completions API 传递 base64 音频进行识别。
        需要先合并 LoRA 权重并在另一终端启动 vLLM 服务。
        """
        from openai import OpenAI

        host = config.ASR_VLLM_HOST
        port = config.ASR_VLLM_PORT

        # vLLM 服务端使用合并后的 checkpoint
        materialized_dir = os.path.join(config.MEGA_ASR_CKPT_DIR, "mega-asr-vllm-materialized")
        self.model_name = getattr(config, "ASR_SERVED_MODEL_NAME", None) or os.path.basename(materialized_dir.rstrip(os.sep))

        connect_host = "localhost" if host == "0.0.0.0" else host

        print(f"[Mega-ASR] 初始化 vLLM API 客户端: http://{connect_host}:{port}/v1")
        print(f"[Mega-ASR] 模型名: {self.model_name}")
        print(f"[Mega-ASR] 请确保已合并 LoRA 并启动 ASR 服务:")
        print(f"  python3 step2_asr_module.py --materialize")
        print(f"  vllm serve {materialized_dir} --host {host} --port {port} --gpu-memory-utilization {config.ASR_VLLM_GPU_MEMORY_UTILIZATION} --dtype auto")

        self.client = OpenAI(
            base_url=f"http://{connect_host}:{port}/v1",
            api_key="EMPTY",
        )

        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"[Mega-ASR] 服务在线，可用模型: {model_ids}")
            if self.model_name not in model_ids:
                print(f"[Mega-ASR] WARNING: 模型名 '{self.model_name}' 不在服务端列表中")
                if model_ids:
                    self.model_name = model_ids[0]
                    print(f"  自动切换为: {self.model_name}")
        except Exception as e:
            print(f"[Mega-ASR] WARNING: 无法连接 ASR 服务 ({e})")
            self.client = None

    def transcribe(self, audio_path: str) -> str:
        """将音频文件转为文本（Mega-ASR transformers 后端）

        参数:
            audio_path: 音频文件路径

        返回:
            识别出的文本字符串
        """
        if self.model is None:
            return "[ERROR] Mega-ASR 模型未加载，请检查 config.MEGA_ASR_REPO_DIR 和 MEGA_ASR_CKPT_DIR"

        if not os.path.exists(audio_path):
            return f"[ERROR] 音频文件不存在: {audio_path}"

        try:
            result = self.model.infer(
                audio=audio_path,
                language=self.language,
            )
            if isinstance(result, dict) and "text" in result:
                return str(result["text"]).strip()
            if isinstance(result, str):
                return result.strip()
            return str(result).strip()
        except Exception as e:
            return f"[ERROR] Mega-ASR 推理失败: {e}"

    def transcribe_audio_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """将 numpy 音频数组通过 vLLM Chat Completions API 转为文本

        音频经标准化、WAV 编码、base64 编码后，通过 Chat Completions API
        以 audio_url 方式发送给 Qwen3-ASR 模型进行识别。

        参数:
            audio: PCM 音频数据（int16 或 float32），单声道
            sample_rate: 音频采样率，会被重采样到 16kHz

        返回:
            识别出的文本字符串
        """
        if self.client is None:
            return "[ERROR] ASR API 客户端未初始化"

        try:
            normalized_audio = self._normalize_audio(audio, sample_rate)
            if len(normalized_audio) == 0:
                return ""

            wav_bytes = self._encode_wav(normalized_audio, config.STREAM_SAMPLE_RATE)
            b64_audio = base64.b64encode(wav_bytes).decode("utf-8")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64_audio}"}},
                    ]
                }],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR] ASR API 调用失败: {e}"

    def transcribe_via_api(self, audio_path: str) -> str:
        """通过 vLLM Chat Completions API 识别音频文件

        读取音频文件并以 base64 编码通过 Chat Completions API
        发送给 Qwen3-ASR 模型进行识别。

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
                audio_bytes = f.read()

            ext = os.path.splitext(audio_path)[1].lower()
            mime_map = {".wav": "audio/wav", ".mp3": "audio/mpeg",
                        ".flac": "audio/flac", ".ogg": "audio/ogg"}
            mime_type = mime_map.get(ext, "audio/wav")

            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": f"data:{mime_type};base64,{b64_audio}"}},
                    ]
                }],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[ERROR] ASR API 调用失败: {e}"

    # ============================================================
    # 音频预处理工具
    # ============================================================

    @staticmethod
    def _normalize_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """标准化音频为 STREAM_SAMPLE_RATE、单声道、int16。"""
        if audio is None:
            return np.array([], dtype=np.int16)

        audio = np.asarray(audio)
        if audio.size == 0:
            return np.array([], dtype=np.int16)

        if audio.ndim > 1:
            audio = audio[:, 0]

        if sample_rate != config.STREAM_SAMPLE_RATE:
            import scipy.signal
            if audio.dtype in (np.int16, np.int32):
                audio = audio.astype(np.float32)
                if np.max(np.abs(audio)) > 1.0:
                    audio = audio / 32768.0
            audio = scipy.signal.resample_poly(audio, config.STREAM_SAMPLE_RATE, sample_rate).astype(np.float32)

        if audio.dtype == np.float32 or audio.dtype == np.float64:
            if np.max(np.abs(audio)) <= 1.0:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        return audio

    @staticmethod
    def _encode_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """将 int16 numpy 数组编码为 WAV 格式的 bytes"""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()


def quick_transcribe(audio_path: str) -> str:
    """快捷函数：单次识别音频文件"""
    asr = ASRModule()
    return asr.transcribe(audio_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mega-ASR 语音识别模块")
    parser.add_argument("--materialize", action="store_true",
                        help="一次性合并 LoRA 权重，生成 vLLM 可用 checkpoint")
    parser.add_argument("audio_path", nargs="?", help="音频文件路径（测试用）")
    args = parser.parse_args()

    if args.materialize:
        materialize_mega_asr()
    elif args.audio_path:
        result = quick_transcribe(args.audio_path)
        print(f"[ASR Result] {result}")
    else:
        print("用法:")
        print("  python3 step2_asr_module.py --materialize          # 合并 LoRA 权重")
        print("  python3 step2_asr_module.py <audio_path>           # 测试识别")
