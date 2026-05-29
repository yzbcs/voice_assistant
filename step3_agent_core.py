"""Step 3: LangChain Agent 核心 — Ministral 文本回复 + OmniVoice TTS 后处理.

设计:
  LLM 通过 LangChain 直接回复文本，TTS 作为后处理步骤自动执行。
  LangChain 框架保留，未来可注册真正的外部工具（搜索、天气、IoT 控制等）。
  TTS 参数（gender/pitch/style）由用户指令解析 + 默认配置决定，不经过 LLM tool。
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import config
from step5_tts_module import OmniVoiceTTS

_TTS: OmniVoiceTTS | None = None


def _get_tts() -> OmniVoiceTTS:
    global _TTS
    if _TTS is None:
        _TTS = OmniVoiceTTS()
    return _TTS


# ============================================================
# 用户语音参数解析
# ============================================================

def _parse_voice_params(user_input: str) -> dict[str, str]:
    """从用户输入中提取 TTS 参数意图，未匹配到的使用默认值。"""
    text = user_input.lower()
    params = {
        "gender": config.TTS_DEFAULT_GENDER,
        "pitch": config.TTS_DEFAULT_PITCH,
        "style": "",
    }

    # 性别
    if re.search(r"男[声声音]|男生|man|male", text):
        params["gender"] = "male"
    elif re.search(r"女[声声音]|女生|woman|female", text):
        params["gender"] = "female"

    # 音高
    if re.search(r"高[一一]?[点点声]|高音|high\s*pitch", text):
        params["pitch"] = "high pitch"
    elif re.search(r"低[一一]?[点点声]|低沉|低音|low\s*pitch", text):
        params["pitch"] = "low pitch"

    # 风格
    if re.search(r"耳语|悄悄|轻声|whisper", text):
        params["style"] = "whisper"

    return params


# ============================================================
# VoiceAssistant
# ============================================================

_SYSTEM_PROMPT = (
    "你是一个智能语音助手。根据用户意图给出简洁、自然的中文回复。"
)


class VoiceAssistant:
    """Ministral agent: LLM 文本回复 + TTS 后处理自动合成语音。

    调用 chat() 获取纯文本回复，调用 chat_with_tts() 获取文本 + TTS 音频。
    """

    def __init__(
        self,
        vllm_base_url: str = f"http://localhost:{config.VLLM_PORT}/v1",
        model_name: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.15,
        max_tokens: int = 2048,
    ):
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        self.temperature = temperature
        self.max_tokens = max_tokens

        model_name = (
            model_name
            or getattr(config, "VLLM_SERVED_MODEL_NAME", None)
            or os.path.basename(config.LLM_MODEL_PATH.rstrip(os.sep))
        )

        self.llm = ChatOpenAI(
            base_url=vllm_base_url,
            api_key="not-needed",
            model=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.chat_history: list = []

    def chat(self, user_input: str) -> str:
        """纯文本对话：返回 LLM 回复文本，不触发 TTS。"""
        messages = [SystemMessage(content=self.system_prompt)] + self.chat_history + [HumanMessage(content=user_input)]
        response = self.llm.invoke(messages)
        reply_text = response.content.strip() if response.content else ""

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=reply_text))
        return reply_text

    def chat_with_tts(self, user_input: str) -> dict[str, Any]:
        """LLM 回复 + 自动 TTS 合成。

        流程: LLM 生成文本 → 解析用户语音参数意图 → TTS 合成音频。
        TTS 是后处理步骤，与 LLM 推理解耦。
        """
        reply_text = self.chat(user_input)
        if not reply_text:
            return {"reply_text": "", "audio_path": "", "instruct": "", "error": "LLM 未返回内容"}

        # 解析用户输入中的语音参数意图
        voice_params = _parse_voice_params(user_input)

        tts_result = _get_tts().generate(
            text=reply_text,
            gender=voice_params["gender"],
            pitch=voice_params["pitch"],
            style=voice_params["style"],
            speed=config.TTS_DEFAULT_SPEED,
        )

        result = {
            "reply_text": reply_text,
            "audio_path": str(tts_result.get("audio_path") or ""),
            "instruct": str(tts_result.get("instruct") or ""),
        }
        if tts_result.get("error"):
            result["error"] = str(tts_result["error"])
        return result

    def reset(self):
        """重置对话历史。"""
        self.chat_history = []


def create_assistant(**kwargs) -> VoiceAssistant:
    """创建 VoiceAssistant 实例的便捷函数。"""
    return VoiceAssistant(**kwargs)


if __name__ == "__main__":
    print("=" * 50)
    print("LangChain Agent + OmniVoice TTS 测试模式")
    print("确保 vLLM 服务已启动: python3 step1_setup_vllm.py")
    print("=" * 50)

    assistant = create_assistant()
    test_cases = [
        "用女生高一点的声音鼓励我一句。",
        "用男生低沉一点回复我：今天辛苦了",
    ]

    for query in test_cases:
        print(f"\n[User] {query}")
        try:
            reply = assistant.chat_with_tts(user_input=query)
            print(f"[Assistant] {reply['reply_text']}")
            print(f"[TTS instruct] {reply['instruct']}")
            print(f"[Audio] {reply['audio_path']}")
            if reply.get("error"):
                print(f"[ERROR] {reply['error']}")
        except Exception as exc:
            print(f"[ERROR] {exc}")
