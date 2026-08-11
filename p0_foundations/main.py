"""
P0 · LangChain 基础热身：Model I/O 三件套
=========================================

本文件用一个小 CLI（命令行程序）演示 LangChain 最核心的「Model I/O」概念：
  1) Model    —— 大语言模型（ChatOpenAI），就是真正"会说话"的那个模型
  2) Prompt   —— 提示词模板（ChatPromptTemplate），告诉模型"怎么干"
  3) Output   —— 输出解析（StrOutputParser / 结构化输出），把模型回复整理成你要的格式

并展示如何用 LCEL（LangChain Expression Language）把三者用 `|` 串成一条「链」。

------------------------------------------------------------------
【Python 新手必读 4 件事】
------------------------------------------------------------------
A) 怎么运行本文件：
      python main.py demo                       # 依次演示 4 个核心概念
      python main.py chat "你好"                 # 直接对话
      python main.py translate "Hello" --to 中文 # 翻译（--to 是可选参数）
      python main.py summarize "一段长文本"       # 总结
      python main.py polish "那个事我搞定了" --tone 口语  # 润色（--tone 可选：正式/口语/简洁/幽默）
   （这里的 python 必须是项目里的 .venv 里的那个，见文末说明）

B) 配置从哪来：
   本文件第 27 行 load_dotenv() 会从项目目录下的 .env 文件里读取
   OPENAI_API_KEY 等配置，再塞进"环境变量"。之后用 os.getenv("OPENAI_API_KEY")
   就能读到。所以 .env 就是放密钥的地方，不会提交到 git。

C) 为什么用 .env 而不是直接写死在代码里？
   密钥属于隐私，写死在代码里容易误传到网上；放 .env 里、把 .env 加进
   .gitignore，就安全了。换模型/换 key 也只改 .env，不用动代码。

D) 环境变量 vs .env：
   .env 只是"方便本地开发"的替身。你也可以在操作系统里直接 export 环境变量，
   load_dotenv() 会优先用 .env 里的值（如果系统已设了同名变量，默认不覆盖，
   想覆盖可写成 load_dotenv(override=True)）。上线到服务器时一般直接设真环境变量。
"""

# ---- 标准库 / 第三方库导入 -------------------------------------------------
import os                          # Python 内置的"操作系统接口"模块，这里用来读环境变量
import argparse                    # Python 内置的"命令行参数解析"模块，用来支持 chat/translate 等子命令
from dotenv import load_dotenv    # python-dotenv：把 .env 文件加载成环境变量的小工具

# 下面这些都是 LangChain 的组件
from langchain_openai import ChatOpenAI              # OpenAI 兼容协议的聊天模型封装（也兼容 DeepSeek/通义/智谱）
from langchain_core.prompts import ChatPromptTemplate # 提示词模板：把"角色设定 + 占位符"结构化管理
from langchain_core.output_parsers import StrOutputParser  # 把模型返回的 AIMessage 提取成纯文本
from pydantic import BaseModel, Field               # Pydantic：用来定义"结构化输出"的数据格式（像一张表单）


# 关键一步：把 .env 文件里的键值对加载到"环境变量"里，后面 os.getenv 才能读到。
# 这行必须在读取配置之前执行（所以放在最上面）。
load_dotenv()


# ---------------------------------------------------------------------------
# 1) Model：构建可复用的模型实例
#    关键点：base_url 留空则用 OpenAI；填了就能无缝切到 DeepSeek/通义/智谱。
# ---------------------------------------------------------------------------
def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        # os.getenv("环境变量名", "读不到时的默认值") —— 第二个参数是兜底默认值
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,                          # 0=最稳重，1=最发散；教学用 0.3 比较均衡
        api_key=os.getenv("OPENAI_API_KEY"),      # 从环境变量读密钥（已在 .env 配好）
        base_url=os.getenv("OPENAI_BASE_URL"),    # 留空（None）即走官方 OpenAI；填了就指向兼容服务
    )


# ---------------------------------------------------------------------------
# 2) Prompt：用模板把「角色设定 + 用户输入」结构化
#    from_messages 支持多轮角色（system / user / assistant），最贴近真实对话。
#    {target_lang} / {text} 是占位符，运行时由 .invoke({"text":..., "target_lang":...}) 填进去。
# ---------------------------------------------------------------------------
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业翻译，只输出翻译结果，不要任何解释。"),
    ("user", "把下面内容翻译成{target_lang}：\n{text}"),
])

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "你擅长把长文本压缩成要点，用中文、分条列出。"),
    ("user", "请总结：\n{text}"),
])

# 【polish 练习】润色提示词：比 translate 多了个 {tone} 占位符，决定改写风格
# 你会发现——换一个提示词，链的结构（prompt | model | parser）一行都不用改，
# 这就是 LangChain 的爽点：换提示词 = 换能力。
polish_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个文本润色助手，把用户给的文字改写成{tone}风格，只输出润色后的结果，不要解释。"),
    ("user", "{text}"),
])


# ---------------------------------------------------------------------------
# 3) Output：把模型返回的 AIMessage 变成你想要的格式
#    3a. StrOutputParser —— 最常用，提取纯文本（去掉模型的元数据，只留文字）
#    3b. with_structured_output —— 用 Pydantic 约束成结构化字段（见下面 SummaryResult）
# ---------------------------------------------------------------------------
str_parser = StrOutputParser()


# 用 Pydantic 定义"结构化输出"长什么样：一个标题 + 几个要点
# 这就像给模型发一张"填空题答题卡"，它会按这个格式填。
class SummaryResult(BaseModel):
    title: str = Field(description="一句话标题")
    bullets: list[str] = Field(description="3 条以内的要点")


# ---------------------------------------------------------------------------
# 4) LCEL 组合：prompt | model | parser，像水管一样把数据流传过去
#    这就是「Chain 组合」—— LangChain 的精髓：把小零件拼成可复用管线。
#    `|` 就是"把左边的输出喂给右边当输入"。
#    注意：build_model() 在模块加载时就被调用了一次，得到一个固定模型实例。
# ---------------------------------------------------------------------------
translate_chain = translate_prompt | build_model() | str_parser
summarize_chain = summarize_prompt | build_model() | str_parser
# 【polish 练习】同样的"水管"结构，只是换了前面的提示词模板，多了 tone 参数
polish_chain = polish_prompt | build_model() | str_parser


# ---------------------------------------------------------------------------
# CLI 子命令：每个函数对应一个命令的具体逻辑
# ---------------------------------------------------------------------------
def cmd_chat(text: str):
    # 直接拿模型对话；.invoke(text) 发送消息，.content 取出回复文字
    model = build_model()
    print(model.invoke(text).content)


def cmd_translate(text: str, target: str):
    # 用链：把 {text, target_lang} 填进模板 → 模型 → 解析成纯文本
    print(translate_chain.invoke({"text": text, "target_lang": target}))


def cmd_summarize(text: str):
    print(summarize_chain.invoke({"text": text}))


# 【polish 练习】命令函数：把 {text, tone} 填进链里跑
def cmd_polish(text: str, tone: str):
    # 注意键名要和模板占位符一致：模板里是 {text} 和 {tone}
    print(polish_chain.invoke({"text": text, "tone": tone}))


def demo():
    """依次演示 4 个核心概念，方便一口气看效果。"""
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
    # with_structured_output 默认走 method="function_calling"（函数调用/tools）。
    # 但 DeepSeek V4 默认是"思考模式"，不支持 tool_choice，函数调用会报错：
    #   "Thinking mode does not support this tool_choice"
    # 解决办法：改用 method="json_mode"，让模型直接吐 JSON，再用 Pydantic 校验成对象。
    #
    # ⚠️ 坑（DeepSeek / OpenAI 都有）：json_object 模式要求 prompt 里必须出现 "json"
    #    这个词，否则报 "Prompt must contain the word 'json'"。所以下面 system 里特意写了"JSON 格式"。
    structured_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是信息提取助手。请严格用 JSON 格式输出，"
                   "包含两个字段：title（一句话标题，字符串）和 bullets（3条以内要点，字符串数组）。"),
        ("user", "{topic}"),
    ])
    # with_structured_output(SummaryResult, method="json_mode") 做两件事：
    #   1) 设置 response_format=json_object，让模型只输出 JSON；
    #   2) 把返回的 JSON 自动用 SummaryResult 这个 Pydantic 类校验、转成对象。
    structured_chain = structured_prompt | model.with_structured_output(SummaryResult, method="json_mode")
    r = structured_chain.invoke({"topic": "人工智能正在改变软件工程，从代码补全到自主 agent。"})
    print(r)
    print("类型：", type(r).__name__)  # SummaryResult —— 是个 Pydantic 对象，不是普通字符串


def main():
    """解析命令行参数，分发到对应子命令。"""
    p = argparse.ArgumentParser(description="P0 LangChain Model I/O 迷你练习")
    sub = p.add_subparsers(dest="cmd")   # dest="cmd" 用来记住用户选了哪个子命令

    # chat 子命令：需要一个位置参数 text（必填）
    sub.add_parser("chat").add_argument("text")

    # translate 子命令：text 必填，--to 可选（默认中文）
    t = sub.add_parser("translate")
    t.add_argument("text")
    t.add_argument("--to", default="中文")

    # summarize 子命令：text 必填
    s = sub.add_parser("summarize")
    s.add_argument("text")

    # 【polish 练习】polish 子命令：text 必填，--tone 可选（默认"正式"）
    pl = sub.add_parser("polish")
    pl.add_argument("text")
    pl.add_argument("--tone", default="正式")

    # demo 子命令：无参数
    sub.add_parser("demo")

    args = p.parse_args()   # 真正解析命令行，得到 args.cmd / args.text 等

    # 根据 args.cmd 决定调用哪个函数
    if args.cmd == "chat":
        cmd_chat(args.text)
    elif args.cmd == "translate":
        cmd_translate(args.text, args.to)
    elif args.cmd == "summarize":
        cmd_summarize(args.text)
    elif args.cmd == "polish":
        cmd_polish(args.text, args.tone)
    elif args.cmd == "demo":
        demo()
    else:
        p.print_help()      # 什么都没输就打印帮助


# 这是 Python 的"入口守卫"：只有当本文件被直接运行（python main.py ...）
# 时才执行 main()；如果被别的文件 import 当模块用，这段代码不会跑。
if __name__ == "__main__":
    main()
