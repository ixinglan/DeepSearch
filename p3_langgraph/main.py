"""
====================================================================
P3 · LangGraph 入门：有状态的可循环编排
====================================================================

P0~P2 我们用的是"链"（prompt | model | parser）——一条直线走到底。
这一阶段用 LangGraph 升级成"图"（StateGraph）——可以分叉、循环、
在节点之间传递状态。核心演示：写→审→改 循环器。

  一个写作 Agent：写一段文字 → 让另一个"审稿人"打分反馈 →
  不达标就回去重写 → 达标才结束。

用到的 LangGraph 概念（逐个对应代码）：

  1. State（状态）      —— TypedDict 定义"图里传递的数据长什么样"
  2. StateGraph         —— 图骨架：把 State 类型告诉它
  3. add_node           —— 注册节点（每个节点是一个处理函数）
  4. add_edge           —— 连边：节点 A 处理完 → 固定走到节点 B
  5. add_conditional_edges —— 条件边：根据 State 里的值，决定下一步
                         走哪条路（这是"循环"和"分支"的开关）
  6. START / END        —— 图的入口 / 出口（LangGraph 1.x 的常量）
  7. compile + invoke   —— 编译成可运行对象，然后像函数一样调用
  8. draw_mermaid       —— 把图可视化，能生成流程图

数据流长这样（这就是本文件的核心，建议对照代码看）：

  START
    │
    ▼
  write（写作节点：生成/修改草稿，把 draft 写进 State）
    │
    ▼
  review（审稿节点：打分 + 反馈，把 score/feedback 写进 State）
    │
    ▼
  should_continue（条件函数：读 State 里的 score）
    │
    ├─ score < 8 ──▶ 回到 write（改！）── 这就是"循环"
    │
    └─ score ≥ 8 ──▶ END（达标，结束）

------------------------------------------------------------------
运行方式（必须用项目 .venv 里的 python）：
  cd ~/Workbuddy/langchain-langGraph/p3_langgraph

  # 跑一次完整演示（默认主题）
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py demo

  # 自定义主题
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py run "用三句话介绍 LangGraph"

  # 打印图的结构（Mermaid 语法，可贴到 mermaid.live 里看图）
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py graph

------------------------------------------------------------------
【Python 新手必读 3 件事】
  A) TypedDict：给字典"规定格式"。WritingState 规定了图里传递的状态
     必须有 topic / draft / review_score / feedback / round 这几个键。
     好处：写错键名 IDE 会提示，节点函数也清楚地知道能拿到什么。

  B) 节点函数签名：每个节点函数接收一个参数 state（当前状态的字典），
     返回一个"要更新哪些键"的字典（只更新返回的键，其它保持不变）。
     这是 LangGraph 的核心约定：读 state，返回部分更新。

  C) 条件函数：不返回"状态"，而是返回一个"路径名"（如 "continue"），
     add_conditional_edges 再用一个映射表把路径名翻译成具体节点。

====================================================================
"""

# ---- 标准库 / 第三方库导入 ----
import os
import argparse

from typing import TypedDict

from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# LangGraph（P3 主角）
from langgraph.graph import StateGraph, START, END

# 读取 .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# ==================================================================
# 1) 构建 LLM 模型（沿用 P0~P2 的写法）
# ==================================================================
def build_model() -> ChatOpenAI:
    """构建模型实例。DeepSeek V4 非思考版，支持工具调用与 json 模式。"""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.5,                      # 写作任务稍高一点，更有发挥空间
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


# ==================================================================
# 2) 定义 State（状态）：图里传递的数据结构
# ==================================================================
class WritingState(TypedDict):
    """写作循环器的状态。

    每个键的含义：
      topic        写作主题（用户在入口传入，全程不变）
      draft        当前草稿（write 节点生成/修改）
      review_score 审稿评分 0~10（review 节点写入）
      feedback     审稿反馈（review 节点写入，给 write 改进用）
      round        已经迭代了几轮（防止无限循环）
    """
    topic: str
    draft: str
    review_score: int
    feedback: str
    round: int


# ==================================================================
# 3) 审稿结果结构（复用 P0 学过的 with_structured_output + json_mode）
# ==================================================================
class ReviewResult(BaseModel):
    """审稿返回的固定结构：一个评分 + 一段反馈。"""
    score: int = Field(description="对草稿的打分，0 到 10 的整数")
    feedback: str = Field(description="给作者的修改建议，中文，2~3 句话")


# ==================================================================
# 4) 节点 1：write（写作节点）
# ==================================================================
def write_node(state: WritingState) -> dict:
    """根据主题（和上轮的反馈）写出一版草稿。

    第一次进来 feedback 为空 → 从零开始写；
    之后每轮进来 feedback 有内容 → 按反馈"改"。
    返回 {"draft": 新草稿, "round": 轮次+1}，其它键保持不动。
    """
    model = build_model()
    current_round = state["round"] + 1

    # 提示词：有反馈就按反馈改，没反馈就首写
    if state.get("feedback"):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位资深写手。请严格根据审稿人的反馈修改文章，"
                       "保留优点、补齐缺点，只输出修改后的正文。"),
            ("user", "主题：{topic}\n\n当前草稿：{draft}\n\n审稿反馈：{feedback}"),
        ])
        text = (prompt | model | StrOutputParser()).invoke({
            "topic": state["topic"],
            "draft": state["draft"],
            "feedback": state["feedback"],
        })
        print(f"✍️  第 {current_round} 轮：按反馈修改草稿")
        print(f"✍️  第 {current_round} 轮, draft：", text.strip())
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位资深写手。请围绕主题写一段 2~3 句话的短文，只输出正文。"),
            ("user", "主题：{topic}"),
        ])
        text = (prompt | model | StrOutputParser()).invoke({"topic": state["topic"]})
        print(f"✍️  第 {current_round} 轮：首次写作")
        print(f"✍️  第 {current_round} 轮, draft：", text.strip())

    return {"draft": text.strip(), "round": current_round}


# ==================================================================
# 5) 节点 2：review（审稿节点）
# ==================================================================
def review_node(state: WritingState) -> dict:
    """给当前草稿打分 + 提反馈。

    这里用 P0 学过的 with_structured_output(method="json_mode")：
      - DeepSeek 默认思考模式不支持强制 function_calling，
        所以用 json_mode 让模型吐 JSON，再用 Pydantic 校验成对象。
      - ⚠️ 坑：json 模式要求 prompt 里必须出现 "json" 这个词！
    """
    model = build_model()

    # 注意 system 里写了"JSON 格式"——这是 json_mode 能跑通的关键
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位极其严格的审稿编辑，习惯用挑剔的眼光审视文章。"
                   "请对草稿打分并给出具体修改建议，严格用 JSON 格式输出两个字段："
                   "score（0-10 整数）、feedback（中文 2~3 句建议）。"
                   "注意：满分 10 几乎不给，9 分代表接近完美，"
                   "大多数有改进空间的草稿应给 6~8 分。"),
        ("user", "主题：{topic}\n\n草稿：{draft}"),
    ])

    # json_mode：模型只输出 JSON，LangChain 自动用 ReviewResult 校验
    structured_model = model.with_structured_output(ReviewResult, method="json_mode")
    result = (review_prompt | structured_model).invoke({
        "topic": state["topic"],
        "draft": state["draft"],
    })

    print(f"🔍  第 {state['round']} 轮审稿：评分 {result.score}/10")
    print(f"     反馈：{result.feedback}")

    return {"review_score": result.score, "feedback": result.feedback}


# ==================================================================
# 6) 条件函数：决定审完稿之后下一步走哪条路
# ==================================================================
def should_continue(state: WritingState) -> str:
    """根据评分决定走向：返回路径名，由条件边映射成节点。

    两条规则：
      - 评分 >= 9，或改到第 3 轮了（防止无限循环）→ "done"（结束）
      - 否则 → "continue"（回到 write 再改一轮）
    """
    if state["review_score"] >= 9:
        print(f"🎉 评分 {state['review_score']} ≥ 9，达标！")
        return "done"
    if state["round"] >= 3:
        print(f"⏹️  已改 {state['round']} 轮仍未达标，强制收尾。")
        return "done"
    print(f"📝 评分 {state['review_score']} < 9，回去再改。")
    return "continue"


# ==================================================================
# 7) 组装 StateGraph：节点 + 边 + 条件边
# ==================================================================
def build_graph():
    """把上面的零件拼成一张图，编译后返回可调用的 app。

    拼图三步：
      1) StateGraph(State)          —— 告诉图"状态长什么样"
      2) add_node / add_edge / add_conditional_edges —— 铺节点和路线
      3) compile()                  —— 编译成可运行对象
    """
    graph = StateGraph(WritingState)              # 1) 图骨架 + 状态类型

    # 2a) 注册节点（名字自取，函数是指向的处理逻辑）
    graph.add_node("write", write_node)
    graph.add_node("review", review_node)

    # 2b) 固定边：入口 → 写；写 → 审（走完一定去下一步）
    graph.add_edge(START, "write")
    graph.add_edge("write", "review")

    # 2c) 条件边：审完 → 看 should_continue 的返回值决定去向
    #     映射表 {"continue": "write", "done": END}
    #     含义：返回 "continue" 就回到 write 节点（循环！），
    #           返回 "done" 就走到 END（结束）
    graph.add_conditional_edges(
        "review",
        should_continue,
        {"continue": "write", "done": END},
    )

    return graph.compile()                        # 3) 编译


# ==================================================================
# 8) 命令函数
# ==================================================================
def cmd_run(topic: str):
    """run 子命令：给个主题，跑完整循环直到达标。"""
    app = build_graph()

    print(f"📌 主题：{topic}")
    print("-" * 60)

    # invoke 传入初始状态（topic 必须给，其余用默认值兜底）
    result = app.invoke({
        "topic": topic,
        "draft": "",
        "review_score": 0,
        "feedback": "",
        "round": 0,
    })

    print("-" * 60)
    print(f"✅ 最终草稿（第 {result['round']} 轮，评分 {result['review_score']}）：")
    print(result["draft"])


def cmd_graph():
    """graph 子命令：打印图的结构（Mermaid 语法）。"""
    app = build_graph()
    print("图的结构（Mermaid 语法，可贴到 https://mermaid.live 渲染）：")
    print("-" * 60)
    print(app.get_graph().draw_mermaid())


def demo():
    """demo 子命令：用默认主题跑一次完整循环。"""
    cmd_run("用三句话介绍 LangGraph 是什么，以及它和 LangChain 的区别")


# ==================================================================
# 9) CLI 入口
# ==================================================================
def main():
    parser = argparse.ArgumentParser(description="P3 LangGraph 入门")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("demo", help="默认主题跑完整循环")
    sub.add_parser("graph", help="打印图结构 Mermaid")
    r = sub.add_parser("run", help="自定义主题跑循环")
    r.add_argument("topic")

    args = parser.parse_args()

    if args.cmd == "demo":
        demo()
    elif args.cmd == "run":
        cmd_run(args.topic)
    elif args.cmd == "graph":
        cmd_graph()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
