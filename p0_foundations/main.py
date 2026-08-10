"""
P0 · LangChain 基础热身：Model I/O 三件套
=========================================
本文件用一个小 CLI 演示 LangChain 最核心的「Model I/O」概念：
  1) Model    —— 大语言模型（ChatOpenAI）
  2) Prompt   —— 提示词模板（ChatPromptTemplate）
  3) Output   —— 输出解析（StrOutputParser / 结构化输出）

并展示如何用 LCEL（LangChain Expression Language）把三者用 `|` 串成一条「链」。

运行方式：
  python main.py demo                       # 依次演示 4 个核心概念
  python main.py chat "你好"                 # 直接对话
  python main.py translate "Hello" --to 中文 # 翻译
  python main.py summarize "一段长文本"       # 总结
"""

import os
import argparse
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()  # 从 .env 读取 OPENAI_API_KEY 等


# ---------------------------------------------------------------------------
# 1) Model：构建可复用的模型实例
#    关键点：base_url 留空则用 OpenAI；填了就能无缝切到 DeepSeek/通义/智谱。
# ---------------------------------------------------------------------------
def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),  # None 即走官方 OpenAI
    )


# ---------------------------------------------------------------------------
# 2) Prompt：用模板把「角色设定 + 用户输入」结构化
#    from_messages 支持多轮角色（system / user / assistant），最贴近真实对话。
# ---------------------------------------------------------------------------
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业翻译，只输出翻译结果，不要任何解释。"),
    ("user", "把下面内容翻译成{target_lang}：\n{text}"),
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "你擅长把长文本压缩成要点，用中文、分条列出。"),
    ("user", "请总结：\n{text}"),
])


# ---------------------------------------------------------------------------
# 3) Output：把模型返回的 AIMessage 变成你想要的格式
#    3a. StrOutputParser —— 最常用，提取纯文本
#    3b. with_structured_output —— 用 Pydantic 约束成结构化字段
# ---------------------------------------------------------------------------
str_parser = StrOutputParser()


class SummaryResult(BaseModel):
    title: str = Field(description="一句话标题")
    bullets: list[str] = Field(description="3 条以内的要点")


# ---------------------------------------------------------------------------
# 4) LCEL 组合：prompt | model | parser，像水管一样把数据流传过去
#    这就是「Chain 组合」—— LangChain 的精髓：把小零件拼成可复用管线。
# ---------------------------------------------------------------------------
translate_chain = translate_prompt | build_model() | str_parser
summarize_chain = summarize_prompt | build_model() | str_parser


# ---------------------------------------------------------------------------
# CLI 子命令
# ---------------------------------------------------------------------------
def cmd_chat(text: str):
    model = build_model()
    print(model.invoke(text).content)


def cmd_translate(text: str, target: str):
    print(translate_chain.invoke({"text": text, "target_lang": target}))


def cmd_summarize(text: str):
    print(summarize_chain.invoke({"text": text}))


def demo():
    model = build_model()

    print("== 1. 直接调用模型（Model I/O 的 Model）==")
    print(model.invoke("用一句话介绍 LangChain").content)

    print("\n== 2. PromptTemplate + 链 ==")
    print(translate_chain.invoke({"text": "Hello, LangGraph!", "target_lang": "中文"}))

    print("\n== 3. StrOutputParser 得到纯文本 ==")
    print(summarize_chain.invoke({"text": "LangChain 是构建 LLM 应用的框架；"
                                  "LangGraph 在其之上做有状态、可循环的多步编排。"}))
    print("类型：", type(summarize_chain.invoke({"text": "测试"})).__name__)

    print("\n== 4. 结构化输出（with_structured_output + Pydantic）==")
    structured = model.with_structured_output(SummaryResult)
    r = structured.invoke("人工智能正在改变软件工程，从代码补全到自主 agent。")
    print(r)


def main():
    p = argparse.ArgumentParser(description="P0 LangChain Model I/O 迷你练习")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("chat").add_argument("text")
    t = sub.add_parser("translate")
    t.add_argument("text")
    t.add_argument("--to", default="中文")
    s = sub.add_parser("summarize")
    s.add_argument("text")
    sub.add_parser("demo")

    args = p.parse_args()
    if args.cmd == "chat":
        cmd_chat(args.text)
    elif args.cmd == "translate":
        cmd_translate(args.text, args.to)
    elif args.cmd == "summarize":
        cmd_summarize(args.text)
    elif args.cmd == "demo":
        demo()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
