"""
====================================================================
P2 · Tools & Agents：让模型会用工具
====================================================================

这一步给模型装上"手"——让它能调用函数（工具），解决自己不会算、
不知道当前日期等问题。核心概念：

  1. 工具（Tool）   —— 用 @tool 装饰器把普通函数变成"模型可调用"的工具
  2. 工具绑定（bind_tools）—— 把工具清单"绑"到模型上，模型自己决定：
                        要不要调、调哪个、传什么参数
  3. 工具调用循环（ReAct）—— 模型请求调用 → 我们执行工具 → 把结果喂回去
                        → 模型基于结果给出最终回答，这就是 Agent 的本质
  4. create_react_agent —— LangGraph 预置的"现成 Agent"，一行代码搞定循环

------------------------------------------------------------------
运行方式（必须用项目 .venv 里的 python）：
  cd ~/Workbuddy/langchain-langGraph/p2_tools_agents

  # 依次演示 3 个层次（bind_tools → 手动循环 → create_react_agent）
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py demo

  # 直接跟 Agent 对话（它会自动决定要不要用工具）
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py agent "15 乘以 8 等于多少？"
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py agent "今天几号？顺便算下 7+9"

------------------------------------------------------------------
【Python 新手必读 3 件事】
  A) @tool 装饰器：写在函数定义上一行，作用是把下面的函数"注册"成工具。
     它读函数的【类型注解】和【docstring】（函数开头的三引号说明文字）
     来生成工具的"说明书"，所以 docstring 一定要写清楚用途——模型就是
     靠这段说明决定什么时候调用你的工具的！

  B) docstring 为什么重要：bind_tools 之后，模型"看到"的不是你的 Python
     代码，而是工具名 + 参数说明 + docstring。写得越清楚，模型越会用对。

  C) 为什么能"让模型调函数"？因为底层是让模型输出一个特殊结构
     （tool_calls），里面写着"我想调用 add，参数 a=12, b=34"。
     它不真正执行代码——执行是我们（Agent 运行时）做的，做完把结果
     再喂回给模型。这就是"模型负责想，代码负责做"的分工。

====================================================================
"""

# ---- 标准库 / 第三方库导入 ----
import os
import argparse
from datetime import date
import json

from dotenv import load_dotenv

# LangChain 组件
from langchain_openai import ChatOpenAI              # 聊天模型（DeepSeek 兼容）
from langchain_core.tools import tool                # @tool 装饰器：把函数变工具

# 预置的 ReAct Agent（P2 的"终极形态"）
# 注意：LangGraph 1.x 起 create_react_agent 已迁移到 langchain.agents，
#       新 API 名为 create_agent（旧名已弃用，将在 V2.0 移除）
from langchain.agents import create_agent

# 读取 .env（显式指定路径，PyCharm 调试时也能找到）
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# ==================================================================
# 1) 构建 LLM 模型（和 P0/P1 完全一样的写法）
# ==================================================================
def build_model() -> ChatOpenAI:
    """构建模型实例。DeepSeek V4 非思考版 deepseek-v4-flash 支持工具调用。"""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,                              # 0=最稳重，1=最发散
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


# ==================================================================
# 2) 自定义工具：@tool 装饰器（核心！）
# ==================================================================
# 要点：docstring（三引号说明）就是给模型看的"工具说明书"，
#       类型注解（float -> float）决定了模型传参的格式。

@tool
def add(a: float, b: float) -> float:
    """两个数相加，返回它们的和。例如 add(2, 3) 返回 5。"""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """两个数相乘，返回它们的积。例如 multiply(4, 5) 返回 20。"""
    return a * b


@tool
def get_today() -> str:
    """返回今天的日期（格式：YYYY-MM-DD），例如 2026-08-17。"""
    return date.today().isoformat()


# 把所有工具放进一个列表，方便后面 bind / 传给 agent
TOOLS = [add, multiply, get_today]


# ==================================================================
# 3) 层次一：bind_tools —— 模型"说出"想调什么，但还没执行
# ==================================================================
def demo_bind_tools():
    """演示 bind_tools：模型输出 tool_calls（想调 add），我们只打印不执行。"""
    print("【层次 1】bind_tools：模型「说出」想调的工具")
    print("-" * 60)

    model_with_tools = build_model().bind_tools(TOOLS)

    # 问一个需要"算"的问题
    result = model_with_tools.invoke("请帮我算一下 12 加 34 等于多少？")
    print(result)

    # result 是 AIMessage，里面有个 tool_calls 属性，装着模型"想调用的工具"
    print("模型回复的内容:", result.content)            # 通常是"我来帮你算"
    print("模型想调用的工具 (tool_calls):")
    for call in result.tool_calls:                    # tool_calls 是列表
        print(f"  - 工具名: {call['name']}")
        print(f"    参数  : {call['args']}")

    print("\n注意：到这里工具【还没有真正执行】！")
    print("模型只是输出了一段'我想调用 add(a=12, b=34)'的结构化文本。\n")


# ==================================================================
# 4) 层次二：手动执行工具调用循环（ReAct 的底层原理）
# ==================================================================
def demo_manual_loop():
    """手动模拟 Agent 循环：invoke → 执行工具 → 结果喂回 → 最终回答。

    这就是 ReAct（Reason + Act）的本质：
      思考(模型说想调工具) → 行动(我们执行工具) → 观察(把结果喂回去) → 再思考
    """
    print("【层次 2】手动执行工具调用循环（ReAct 底层原理）")
    print("-" * 60)

    model_with_tools = build_model().bind_tools(TOOLS)

    # 第一轮：模型决定要不要调工具
    response = model_with_tools.invoke("请帮我算一下 15 乘以 8 等于多少？")

    # 建一个"消息历史"，后面要把工具结果作为 assistant 消息之后的新消息追加
    messages = [response]

    # 只要模型还想调工具，就循环
    while response.tool_calls:
        for call in response.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]

            # 1) 执行工具（在工具列表里按名字找到对应函数并调用）
            matched = next(t for t in TOOLS if t.name == tool_name)
            tool_result = matched.invoke(tool_args)   # 真正运行代码！这里算出 120
            print(f"→ 执行工具 [{tool_name}]({tool_args}) = {tool_result}")

            # 2) 把工具结果包装成 ToolMessage 追加进对话历史
            #    role=tool 表示"这是工具的执行结果"，和模型的 tool_call 一一对应
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(
                content=str(tool_result),             # 工具返回值（必须是字符串）
                tool_call_id=call["id"],              # 关键！要匹配模型的调用编号
            ))

        # 3) 把完整历史（含工具结果）再喂回模型 → 模型给出最终回答
        response = model_with_tools.invoke(messages)

    print("\n→ 模型最终回答:", response.content)


# ==================================================================
# 5) 层次三：create_react_agent —— LangGraph 一行搞定整个循环
# ==================================================================
def build_agent():
    """用 langchain.agents 的 create_agent 构建完整 Agent。

    它内部自动做了层次二里的所有事（循环、执行工具、喂回结果、
    直到模型不再请求工具为止），还支持多轮、错误处理等。
    """
    return create_agent(
        model=build_model(),       # 用哪个模型
        tools=TOOLS,               # 给哪些工具
    )


def cmd_agent(question: str):
    """agent 子命令：直接和 Agent 对话。"""
    agent = build_agent()
    print(f"🤖 问题: {question}")
    print("-" * 60)

    # invoke 会返回最终消息，放在 result["messages"] 列表里
    result = agent.invoke({"messages": [("user", question)]})
    # 将 result 打印为 json 结构
    print(result)

    # 打印整个对话过程（能看到模型和工具的一来一回）
    for msg in result["messages"]:
        role = msg.type                     # human / ai / tool
        if role == "ai" and msg.content:
            print(f"🤖 [模型] {msg.content}")
        elif role == "tool":
            print(f"🔧 [工具 {msg.name}] -> {msg.content}")
    print("-" * 60)


# ==================================================================
# 6) 演示入口：依次展示 3 个层次
# ==================================================================
def demo():
    print("=" * 60)
    print("  P2 Demo：从 bind_tools 到 ReAct Agent")
    print("=" * 60)
    demo_bind_tools()
    demo_manual_loop()

    print("【层次 3】create_react_agent（LangGraph 自动循环）")
    print("-" * 60)
    cmd_agent("今天几号？顺便帮我算 7 加 9 等于多少？")
    print()


# ==================================================================
# 7) CLI 入口
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="P2 Tools & Agents")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("demo", help="依次演示 3 个层次")

    a = sub.add_parser("agent", help="和 Agent 对话")
    a.add_argument("question", help="你的问题")

    args = parser.parse_args()

    if args.cmd == "demo":
        demo()
    elif args.cmd == "agent":
        cmd_agent(args.question)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
