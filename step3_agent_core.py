"""Step 3: LangChain Agent 核心 — Ministral-3-8B + Tool Calling

提供 VoiceAssistant 类，封装 LLM Agent 的创建和调用。

用法:
    from step3_agent_core import VoiceAssistant

    assistant = VoiceAssistant()                    # 初始化
    response = assistant.chat("北京今天天气怎么样？")  # 单轮对话
    response = assistant.chat("帮我算 123*456")       # 带 tool calling
    assistant.reset()                                # 重置对话历史

============================================================
🔧 如何添加自定义工具
============================================================

添加新工具只需三步：

1. 用 @tool 装饰器定义工具函数：

   from langchain_core.tools import tool

   @tool
   def my_new_tool(param1: str, param2: int) -> str:
       \"\"\"工具描述（LLM 会根据这个决定是否调用）\"\"\"
       # 你的实现逻辑
       return f"结果: {param1} {param2}"

2. 将工具函数加到 TOOLS 列表：

   TOOLS = [get_weather, calculator, get_current_time, my_new_tool]

3. 重启 VoiceAssistant 即可。LLM 会根据用户输入自动选择工具。

============================================================
📝 工具编写注意事项
============================================================

- 函数签名必须有完整的类型注解（LLM 需要知道参数类型）
- docstring 是 LLM 判断是否调用该工具的唯一依据，务必写清楚
- 返回值统一为 str 类型
- 如果工具需要调用外部 API，建议在函数内处理异常并返回错误信息
- 工具之间不要有依赖关系，保持独立可测

============================================================
🔌 工具接口规范
============================================================

每个工具函数需满足：
  - 被 @tool 装饰
  - 有完整的参数类型注解
  - 有清晰的 docstring（说明功能、参数含义）
  - 返回 str 类型
  - 自行处理异常，不向上抛出
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

import config


# ============================================================
# 🛠️ 工具定义区 — 在此处添加自定义工具
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"、"New York"
    """
    # TODO: 接入真实天气 API，如和风天气、OpenWeatherMap 等
    # 示例：使用 wttr.in 免费 API
    try:
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        current = data["current_condition"][0]
        return (
            f"{city} 当前天气: {current['weatherDesc'][0]['value']}, "
            f"温度 {current['temp_C']}°C, "
            f"体感温度 {current['FeelsLikeC']}°C, "
            f"湿度 {current['humidity']}%, "
            f"风速 {current['windspeedKmph']}km/h"
        )
    except Exception as e:
        return f"[天气查询失败] {city}: {e}"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式

    参数:
        expression: 数学表达式，如 "2+3*4"、"sqrt(144)"、"3.14*5**2"
    支持: 基本运算(+,-,*,/,**)、math 库函数(sqrt, sin, cos, log 等)
    """
    # 安全计算：只允许数学表达式，禁止任意代码执行
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names["abs"] = abs
    allowed_names["round"] = round
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"[计算错误] {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def web_search(query: str) -> str:
    """简单网络搜索（预留接口，需接入搜索引擎 API）

    参数:
        query: 搜索关键词
    """
    # TODO: 接入搜索引擎 API，如 Google Custom Search、Bing Search、SearXNG 等
    return f"[web_search] 搜索功能待接入，查询: {query}"


# ============================================================
# 📋 已注册工具列表 — 在此处注册新工具
# ============================================================

TOOLS = [
    get_weather,
    calculator,
    get_current_time,
    web_search,
]


# ============================================================
# 🤖 VoiceAssistant 类
# ============================================================

class VoiceAssistant:
    """语音助手 Agent 核心类

    封装 Ministral-3-8B LLM + Tool Calling + 多轮对话。

    初始化参数:
        vllm_base_url: vLLM 服务地址，默认 http://localhost:8000/v1
        model_name: 模型名称，默认 config 中路径的 basename
        tools: 工具列表，默认使用 TOOLS
        system_prompt: 系统提示词，默认从 config 读取
        temperature: 生成温度
        max_tokens: 最大生成 token 数
    """

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

        # 与 step1_setup_vllm.py 的 --served-model-name 保持一致
        model_name = model_name or os.path.basename(config.LLM_MODEL_PATH.rstrip(os.sep))

        # LLM：通过 OpenAI 兼容接口连接 vLLM
        self.llm = ChatOpenAI(
            base_url=vllm_base_url,
            api_key="not-needed",  # 本地服务不需要 key
            model=model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 对话历史
        self.chat_history: list = []

        # 构建提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 构建 Agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
        )

    def chat(self, user_input: str) -> str:
        """处理用户输入，返回回复文本

        参数:
            user_input: 用户输入的文本（可以来自 ASR 或直接键入）

        返回:
            助手回复的文本
        """
        response = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": self.chat_history,
        })

        # 更新对话历史
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response["output"]))

        return response["output"]

    def reset(self):
        """重置对话历史"""
        self.chat_history = []


# ============================================================
# 便捷函数
# ============================================================

def create_assistant(**kwargs) -> VoiceAssistant:
    """创建 VoiceAssistant 实例的便捷函数

    参数会透传给 VoiceAssistant.__init__()
    """
    return VoiceAssistant(**kwargs)


if __name__ == "__main__":
    # 简单测试：直接文本对话（需要 vLLM 服务已启动）
    print("="*50)
    print("LangChain Agent 测试模式")
    print("确保 vLLM 服务已启动: python3 step1_setup_vllm.py")
    print("="*50)

    assistant = create_assistant()

    # 测试 tool calling
    test_cases = [
        "北京今天天气怎么样？",
        "帮我算 (123 + 456) * 2",
        "现在几点了？",
    ]

    for query in test_cases:
        print(f"\n[User] {query}")
        try:
            reply = assistant.chat(query)
            print(f"[Assistant] {reply}")
        except Exception as e:
            print(f"[ERROR] {e}")
