"""Step 3: LangChain Agent 核心 — Ministral 工具调用 + OmniVoice TTS."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import config
from step5_tts_module import OmniVoiceTTS


_TTS: OmniVoiceTTS | None = None


def _get_tts() -> OmniVoiceTTS:
    global _TTS
    if _TTS is None:
        _TTS = OmniVoiceTTS()
    return _TTS


@tool
def synthesize_voice_reply(reply_text: str, gender: str, pitch: str, style: str = "") -> str:
    """合成语音回复并返回 JSON。

    参数:
        reply_text: 要对用户说的中文回复文本，必须简洁自然。
        gender: 声音性别，只能是 "male" 或 "female"。
        pitch: 音高，只能是 "low pitch"、"medium pitch" 或 "high pitch"。
        style: 声音风格，只能是 "" 或 "whisper"。
    """
    try:
        result = _get_tts().generate(
            text=reply_text,
            gender=gender,
            pitch=pitch,
            style=style,
            speed=config.TTS_DEFAULT_SPEED,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {
                "reply_text": reply_text,
                "instruct": "",
                "audio_path": "",
                "error": f"TTS synthesis failed: {exc}",
            },
            ensure_ascii=False,
        )


TOOLS = [synthesize_voice_reply]


class VoiceAssistant:
    """Ministral agent that must route replies through the OmniVoice TTS tool."""

    def __init__(
        self,
        vllm_base_url: str = f"http://localhost:{config.VLLM_PORT}/v1",
        model_name: str | None = None,
        tools: list | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.15,
        max_tokens: int = 2048,
    ):
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.tools = tools or TOOLS
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
        self.agent = create_react_agent(self.llm, self.tools, prompt=self.system_prompt)
        self.chat_history: list = []

    @staticmethod
    def _extract_tts_payload(response_messages: list[Any]) -> dict[str, Any] | None:
        for msg in reversed(response_messages):
            if isinstance(msg, ToolMessage):
                try:
                    payload = json.loads(str(msg.content))
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and (
                    payload.get("audio_path") is not None or payload.get("reply_text") is not None
                ):
                    return payload
        return None

    @staticmethod
    def _extract_last_ai_reply(response_messages: list[Any]) -> str:
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and msg.content:
                return str(msg.content)
        return ""

    def chat_with_tts(self, user_input: str) -> dict[str, Any]:
        """处理用户输入，返回文本、OmniVoice instruct、音频路径和原始 agent 回复。"""
        messages = list(self.chat_history) + [HumanMessage(content=user_input)]
        response = self.agent.invoke({"messages": messages})
        response_messages = response.get("messages", [])

        raw_agent_reply = self._extract_last_ai_reply(response_messages)
        payload = self._extract_tts_payload(response_messages)

        if payload is None:
            fallback_text = raw_agent_reply or "好的。"
            payload = _get_tts().generate(
                text=fallback_text,
                gender=config.TTS_DEFAULT_GENDER,
                pitch=config.TTS_DEFAULT_PITCH,
                style="",
                speed=config.TTS_DEFAULT_SPEED,
            )

        reply_text = str(payload.get("reply_text") or raw_agent_reply or "")
        result = {
            "reply_text": reply_text,
            "audio_path": str(payload.get("audio_path") or ""),
            "instruct": str(payload.get("instruct") or ""),
            "raw_agent_reply": raw_agent_reply,
        }
        if payload.get("error"):
            result["error"] = str(payload["error"])

        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=reply_text))
        return result

    def chat(self, user_input: str) -> str:
        """兼容旧文本调用：返回 TTS 工具选择出的回复文本。"""
        return self.chat_with_tts(user_input)["reply_text"]

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
            reply = assistant.chat_with_tts(query)
            print(f"[Assistant] {reply['reply_text']}")
            print(f"[TTS instruct] {reply['instruct']}")
            print(f"[Audio] {reply['audio_path']}")
            if reply.get("error"):
                print(f"[ERROR] {reply['error']}")
        except Exception as exc:
            print(f"[ERROR] {exc}")
