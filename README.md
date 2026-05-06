# 语音助手工作流 — 对接文档

## 项目概述

基于 **Ministral-3-8B** + **Qwen3-ASR** + **LangChain** 搭建的本地语音助手，支持 tool calling（工具调用），运行在 H200 GPU 上。

## 架构

```
麦克风录音 → Qwen3-ASR (独立模块) → LangChain Agent (Ministral-3-8B + Tools) → 文本回复
                  ↑                                    ↓
            step2_asr_module.py               Tool Calling (天气/计算/搜索等)
```

## 文件清单

| 文件 | 说明 |
|------|------|
| `config.py` | 配置文件：模型路径、vLLM 参数、录音参数 |
| `requirements.txt` | Python 依赖 |
| `step1_setup_vllm.py` | 启动 vLLM 服务加载 Ministral-3-8B |
| `step2_asr_module.py` | Qwen3-ASR 语音识别模块 |
| `step3_agent_core.py` | LangChain Agent 核心（LLM + Tools + 多轮对话）|
| `step4_main.py` | 主入口：录音 → ASR → Agent → 输出 |

## 快速开始

### 1. 配置模型路径

编辑 `config.py`，填写本地模型路径：

```python
LLM_MODEL_PATH = "/your/path/to/Ministral-3-8B-Instruct-2512"
ASR_MODEL_PATH = "/your/path/to/Qwen3-ASR"
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动 vLLM 服务

```bash
python3 step1_setup_vllm.py
```

### 4. 运行助手

```bash
python3 step4_main.py              # 交互式选择模式
python3 step4_main.py --mode text  # 纯文本模式
python3 step4_main.py --mode voice # 语音模式
```

## 🔧 如何添加自定义工具

三步即可：

### Step 1: 定义工具函数

在 `step3_agent_core.py` 的工具定义区添加：

```python
from langchain_core.tools import tool

@tool
def my_new_tool(param1: str, param2: int) -> str:
    \"\"\"工具描述（LLM 会根据这个决定是否调用）\"\"\"
    # 你的实现逻辑
    return f"结果: {param1} {param2}"
```

### Step 2: 注册到 TOOLS 列表

```python
TOOLS = [get_weather, calculator, get_current_time, web_search, my_new_tool]
```

### Step 3: 重启助手即可

LLM 会根据用户输入自动判断是否调用新工具。

### 工具编写规范

- 必须有 `@tool` 装饰器
- 参数必须有完整类型注解
- docstring 要写清楚功能（LLM 依据此判断调用）
- 返回值统一 `str`
- 自行处理异常，不向上抛出

## 当前已注册工具

| 工具名 | 功能 | 状态 |
|--------|------|------|
| `get_weather` | 查询天气（wttr.in API） | ✅ 可用 |
| `calculator` | 数学表达式计算 | ✅ 可用 |
| `get_current_time` | 获取当前时间 | ✅ 可用 |
| `web_search` | 网络搜索 | ⚠️ 预留接口，需接入搜索 API |

## 关键设计决策

1. **ASR 作为独立前置模块**（不编入 LangChain）：语音识别是确定性流程，不需要 LLM 决策，独立可替换
2. **vLLM 服务化部署**：通过 OpenAI 兼容接口暴露 LLM，LangChain 用 `ChatOpenAI` 对接
3. **config.py 统一配置**：所有路径和参数集中管理，方便适配不同环境

## 实验推进历程

### Round 1: 初始实现

- **动机**: 搭建 Ministral-3-8B + Qwen3-ASR + LangChain 的语音助手原型
- **方案**: vLLM 服务化部署 LLM，ASR 独立模块，LangChain Agent 带 tool calling
- **结果**: 完成四步代码框架，支持语音/文本两种交互模式
- **待验证**: 需填写实际模型路径后端到端测试
- **下一步**: 接入真实模型路径，验证 vLLM tool calling 兼容性，接入 TTS