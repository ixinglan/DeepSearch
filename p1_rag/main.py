"""
====================================================================
P1 · RAG 入门：让模型能查你自己的资料
====================================================================

这个文件演示 RAG（检索增强生成）的完整流程，分 5 步：
  1. 加载文档（Load）    —— 把 txt 文件读进内存，变成 Document 对象
  2. 切分文本（Split）   —— 把长文档切成小块 chunk
  3. 向量嵌入（Embed）   —— 用模型把文本块变成向量（数字数组）
  4. 向量存储（Store）   —— 存进 Milvus 向量数据库（Docker 运行的独立服务）
  5. 检索生成（Retrieve）—— 用户提问 → 找最相似的片段 → 拼进 prompt → 模型回答

运行方式（必须用项目 .venv 里的 python）：
  cd ~/Workbuddy/langchain-langGraph/p1_rag

  # 第一步：把 data/ 下的文档"灌"进 Milvus 向量库（只需跑一次，之后存到数据库里了）
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py ingest

  # 第二步：基于本地文档回答问题
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py ask "什么是 RAG？"

  # 第三步：看完整流程演示（5 步逐一展示）
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py demo

  # 查看知识库里的文档
  ~/Workbuddy/langchain-langGraph/.venv/bin/python main.py list

Python 新手必读 4 件事：
  1. import 顺序：标准库（os, argparse）→ 第三方库（langchain_xx）→ 本地模块
  2. os.getenv("KEY", "默认值")：读环境变量，不存在就用默认值
  3. load_dotenv()：把 .env 文件里的 KEY=VALUE 加载成环境变量
  4. if __name__ == "__main__"：这个文件被直接运行时才执行下面的代码，被 import 时不执行
====================================================================180011

"""

import os
import argparse

# ---- 第三方库 ----
from dotenv import load_dotenv

# LangChain 核心组件
from langchain_openai import ChatOpenAI                        # LLM（大语言模型）
from langchain_core.prompts import ChatPromptTemplate          # 提示词模板
from langchain_core.output_parsers import StrOutputParser      # 输出解析器（提取纯文本）
from langchain_core.runnables import RunnablePassthrough       # 透传器（原样传递数据）
from langchain_core.documents import Document                  # 文档对象（page_content + metadata）

# 文档加载器（从 langchain_community 导入）
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# 文本切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 向量存储（Milvus）—— 需要先装 langchain-milvus 和 pymilvus
from langchain_milvus import Milvus
from pymilvus import MilvusClient


# ==================================================================
# 第 1 步：配置加载 —— 读 .env 文件，拿到 API key 和模型名
# ==================================================================

# load_dotenv() 会自动找当前目录或上级目录的 .env 文件
# 我们显式指定路径，确保在 PyCharm 里调试时也能找到
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 数据目录和向量库配置（都用绝对路径，避免 cwd 不同导致找不到文件）
# __file__ 是当前脚本的路径，os.path.dirname 取它所在的目录
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")          # 存放原始文档的目录

# ---- Milvus 连接配置 ----
# Milvus 是一个独立运行的向量数据库服务（不是像 FAISS 那样读文件），
# 所以这里不需要 INDEX_DIR 这种"存到磁盘的路径"，而是需要"连接地址 + collection 名"。
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")  # Milvus 服务的地址+端口
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "p1_rag_docs")  # collection 名（类似数据库表名）


# ==================================================================
# 第 2 步：构建 LLM 模型（和 P0 一样的逻辑，复用）
# ==================================================================

def build_model() -> ChatOpenAI:
    """
    构建 LLM 实例。

    ChatOpenAI 是 LangChain 对 OpenAI 兼容协议的封装。
    DeepSeek / 通义 / 智谱都兼容这个协议，只要改 base_url 和 model 就能切换。

    os.getenv("KEY", "默认值") 的含义：
      - 先去环境变量里找 OPENAI_MODEL
      - 如果找不到（没设置），就用 "gpt-4o-mini" 作为默认值
      - 这样即使用户没配 .env，代码也不会报错（只是调用会失败）
    """
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,  # 0=最确定性，1=最随机；问答场景 0.3 比较合适
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),  # None 时走默认 OpenAI 官方地址
    )
    return model


# ==================================================================
# 第 3 步：构建 Embeddings（向量嵌入模型）
# ==================================================================

def build_embeddings():
    """
    构建 Embeddings 模型，用于把文本变成向量。

    向量是什么？—— 就是一串浮点数，比如 [0.12, -0.34, 0.56, ...]（通常几百到几千维）。
    语义相近的文本，向量也相近（余弦相似度高），这就是"向量搜索"能找到相关文档的原理。

    这里用 Ollama 本地部署的 embedding 模型（免费、离线可用、不占 Python 依赖）。
    用户已通过 Ollama 安装了 bge-m3 模型（一个优秀的中英双语 embedding 模型）。

    工作原理：
      - Ollama 在本地启动一个 HTTP 服务（默认 http://localhost:11434）
      - LangChain 通过 OllamaEmbeddings 调用这个服务，把文本变成向量
      - 向量计算在 Ollama 进程里完成，Python 这边只负责发请求和接收结果

    对比其他方案：
      - HuggingFace 本地模型：需要在 Python 里装 PyTorch（~500MB），重
      - OpenAI Embeddings API：要额外花钱，且 DeepSeek 不提供这个接口
      - Ollama：模型独立部署，Python 侧零依赖，最轻量 ✅
    """

    from langchain_community.embeddings import OllamaEmbeddings

    # 从 .env 读取配置，有默认值兜底
    model_name = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    print(f"[Embeddings] 使用 Ollama 本地模型: {model_name}（{base_url}）")
    return OllamaEmbeddings(
        model=model_name,
        base_url=base_url,
    )


# ==================================================================
# 第 4 步：文档加载（Load）
# ==================================================================

def load_documents(source: str = DATA_DIR) -> list:
    """
    加载 data/ 目录下所有 .txt 文件，返回 Document 对象列表。

    Document 是 LangChain 的标准文档格式，每个 Document 有两个属性：
      - page_content：文档的文本内容（字符串）
      - metadata：元信息（字典），比如来源文件名、页码等

    DirectoryLoader 会遍历目录下所有匹配 glob 模式的文件，
    逐个用 TextLoader 读取，最终返回一个 List[Document]。

    参数：
      source: 文档目录路径，默认是 data/
    """

    print(f"[Load] 正在从 {source} 加载文档...")

    # DirectoryLoader 参数说明：
    #   path: 目录路径
    #   glob: 文件匹配模式，"*.txt" 表示所有 txt 文件
    #   loader_cls: 用什么加载器读单个文件（TextLoader 读纯文本）
    loader = DirectoryLoader(
        path=source,
        glob="*.txt",
        loader_cls=TextLoader,
        # show_progress=True,  # 如果装了 tqdm 可以显示进度条
    )

    docs = loader.load()

    print(f"[Load] 加载完成：共 {len(docs)} 个文档")
    for i, doc in enumerate(docs):
        # doc.metadata["source"] 是文件路径，取 basename 只显示文件名
        filename = os.path.basename(doc.metadata.get("source", "未知"))
        print(f"  [{i+1}] {filename}（{len(doc.page_content)} 字符）")

    return docs


# ==================================================================
# 第 5 步：文本切分（Split）
# ==================================================================

def split_documents(docs: list, chunk_size: int = 300, chunk_overlap: int = 50) -> list:
    """
    把长文档切成小块（chunk）。

    为什么要切分？
      - 模型有 token 上限，太长的文档塞不进 prompt
      - 检索时按块搜索，太长的块会引入无关信息，太短又丢上下文
      - 一般 chunk_size 设 300~500 字符，overlap 设 50~100（重叠避免切断语义）

    RecursiveCharacterTextSplitter 的切分逻辑（递归切分）：
      1. 先尝试用 \\n\\n（段落）切分
      2. 如果某段还是太长，用 \\n（换行）切
      3. 还太长，用空格切
      4. 最后用字符切
      这样能尽量保持语义完整性（优先在自然边界处切）

    参数：
      docs: Document 列表
      chunk_size: 每块最大字符数（默认 300）
      chunk_overlap: 相邻块重叠的字符数（默认 50，防止把一句话切两半）
    """

    print(f"[Split] 正在切分文档（chunk_size={chunk_size}, overlap={chunk_overlap}）...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # 每块最多 300 个字符
        chunk_overlap=chunk_overlap, # 相邻块重叠 50 个字符
        # separators=["\n\n", "\n", "。", "；", "，", " ", ""],  # 中文可以加句号、分号做分隔
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    print(f"[Split] 切分完成：{len(docs)} 个文档 → {len(chunks)} 个文本块")
    # 展示前 3 个块的内容（截断显示）
    for i, chunk in enumerate(chunks[:3]):
        filename = os.path.basename(chunk.metadata.get("source", "未知"))
        preview = chunk.page_content[:80].replace("\n", " ")
        print(f"  [块 {i+1}] 来源={filename} | 内容: {preview}...")

    return chunks


# ==================================================================
# 第 6 步：构建向量库（Store）
# ==================================================================

def build_vectorstore(chunks: list, embeddings) -> Milvus:
    """
    把文本块变成向量，存进 Milvus 向量数据库。

    Milvus 和 FAISS 的关键区别：
      - FAISS：纯本地库，向量存在 Python 进程里，save_local() 存到磁盘文件
      - Milvus：独立运行的数据库服务（Docker），通过 gRPC 连接，数据持久化在服务端
      - 好处：多进程可共享同一个向量库；可以单独扩容；重启 Python 不丢数据

    Milvus.from_documents() 做了什么？
      1. 连接到 Milvus 服务（通过 connection_args 指定地址）
      2. 创建一个 collection（类似数据库表），指定向量维度等参数
      3. 遍历每个 chunk，用 embeddings 把 page_content 变成向量
      4. 把向量和原始文本一起插入 collection
      5. 返回 Milvus 实例，之后可以用来搜索

    参数：
      chunks: 切分后的文本块列表
      embeddings: 向量嵌入模型
    """

    print(f"[Store] 正在构建向量库（{len(chunks)} 个文本块 → Milvus）...")

    # connection_args: Milvus 服务的连接地址
    # collection_name: collection 名字（类似数据库表名），不同项目用不同名字避免冲突
    # drop_old: 如果 collection 已存在，先删掉再重建（避免数据重复/维度不匹配）
    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=MILVUS_COLLECTION,
        connection_args={"uri": MILVUS_URI},
        drop_old=True,  # True = 如果 collection 已存在，先删旧再建新（重新灌数据时用）
    )

    print(f"[Store] 向量库已写入 Milvus（collection: {MILVUS_COLLECTION}）")
    print("[Store] 向量库构建完成")
    return vectorstore


def load_vectorstore(embeddings) -> Milvus | None:
    """
    连接已有的 Milvus 向量库（collection）。

    和 FAISS 的区别：
      - FAISS：load_local() 从磁盘读取 index.faiss 文件
      - Milvus：直接连接服务，指定 collection_name 即可，数据全在服务端

    如果 collection 不存在（说明没 ingest 过），返回 None。
    """

    # 先检查 collection 是否存在
    # 用 MilvusClient（pymilvus 新版 API，替代旧的 connections + utility）
    # MilvusClient 是一个轻量级客户端，专门做"检查/管理"类操作
    try:
        client = MilvusClient(uri=MILVUS_URI)
        has_collection = client.has_collection(MILVUS_COLLECTION)
        client.close()
    except Exception as e:
        print(f"[Store] 连接 Milvus 失败: {e}")
        return None

    if not has_collection:
        return None

    print(f"[Store] 连接 Milvus collection: {MILVUS_COLLECTION}...")
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=MILVUS_COLLECTION,
        connection_args={"uri": MILVUS_URI},
    )
    print("[Store] 向量库连接完成")
    return vectorstore


# ==================================================================
# 第 7 步：构建 RAG 链（检索 + 生成）
# ==================================================================

def build_rag_chain(vectorstore: Milvus, model: ChatOpenAI):
    """
    构建 RAG 链：retriever | prompt | model | parser

    这是 P1 的核心——把"检索"和"生成"用 LCEL 的 | 管道串起来。

    数据流是这样的：

      用户问题（字符串）
         │
         ▼
      ┌──────────────────────────────────────────────┐
      │  retriever：从向量库找最相关的 3 个文本块      │
      │  format_docs：把文本块拼接成一个字符串         │
      │  → 输出 {"context": "拼接的文本",             │
      │          "question": 用户原始问题}             │
      └──────────────────────────────────────────────┘
         │
         ▼
      ┌──────────────────────────────────────────────┐
      │  prompt：把 context 和 question 填进模板      │
      │  → 输出 ChatPrompt（结构化的对话消息）         │
      └──────────────────────────────────────────────┘
         │
         ▼
      ┌──────────────────────────────────────────────┐
      │  model：LLM 根据检索到的资料生成回答           │
      │  → 输出 AIMessage                             │
      └──────────────────────────────────────────────┘
         │
         ▼
      ┌──────────────────────────────────────────────┐
      │  parser：StrOutputParser 提取纯文本            │
      │  → 输出字符串（最终的回答）                    │
      └──────────────────────────────────────────────┘

    关键概念：
      - RunnablePassthrough()：原样传递输入。这里把用户问题原封不动传给 question 字段。
      - retriever | format_docs：先检索，再把结果格式化成字符串。
      - 两个字典合并：{"context": ...} 和 {"question": ...} 会自动合并成一个字典。
    """

    # 创建检索器：从向量库搜索最相似的文本块
    # search_type="similarity"：余弦相似度搜索
    # search_kwargs={"k": 3}：返回最相似的 3 个文本块
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # RAG 提示词模板
    # 注意：prompt 里会包含检索到的资料（{context}），这就是 RAG 的核心——
    # 让模型"看着资料"回答，而不是仅凭自己的记忆
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识库问答助手。请根据下面提供的资料回答用户的问题。"
                   "如果资料中没有相关信息，请如实说'根据现有资料无法回答'。"
                   "回答时请引用资料中的关键信息。\n\n"
                   "【检索到的资料】\n{context}"),
        ("user", "{question}"),
    ])

    # 辅助函数：把检索到的 Document 列表拼接成一个字符串
    # 每个 chunk 之间用两个换行分隔，方便模型区分
    def format_docs(docs: list) -> str:
        formatted = "\n\n---\n\n".join(
            f"[来源: {os.path.basename(d.metadata.get('source', '未知'))}]\n{d.page_content}"
            for d in docs
        )
        return formatted

    # ========== 核心：用 | 组装 RAG 链 ==========
    #
    # 这就是 LCEL 的威力——用 | 把"检索→格式化→填模板→调模型→解析输出"一气呵成
    #
    # 第一段 {"context": ..., "question": ...} 是一个并行执行的字典：
    #   - "context" 走 retriever → format_docs 这条路（检索+格式化）
    #   - "question" 走 RunnablePassthrough() 这条路（原样传递用户输入）
    #   两路并行执行，结果合并成一个字典传给后面的 prompt
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )

    return rag_chain, retriever


# ==================================================================
# 命令函数：CLI 各子命令的具体逻辑
# ==================================================================

def cmd_ingest():
    """
    ingest 子命令：把 data/ 下的文档灌进向量库。

    完整执行：加载 → 切分 → 嵌入 → 存储
    这是"一次性"操作，之后 ask 时直接从磁盘加载向量库，不用重复计算。
    """
    print("=" * 60)
    print("  RAG Ingest：构建向量知识库")
    print("=" * 60)

    # 1. 加载文档
    docs = load_documents()

    # 2. 切分文本
    chunks = split_documents(docs)

    # 3. 构建 embeddings
    embeddings = build_embeddings()

    # 4. 构建向量库（会自动写入 Milvus）
    build_vectorstore(chunks, embeddings)

    print("\n✅ 向量知识库构建完成！现在可以用 ask 命令提问了。")
    print(f"   例: python main.py ask \"什么是 LangChain？\"")


def cmd_ask(question: str, show_sources: bool = True):
    """
    ask 子命令：基于本地文档回答问题。

    流程：
      1. 加载已有向量库（如果不存在，提示先 ingest）
      2. 构建 RAG 链
      3. 调用 RAG 链，获得回答
      4. 可选：显示检索到了哪些文本块（溯源）

    参数：
      question: 用户的问题
      show_sources: 是否显示检索到的来源文档
    """

    # 先加载向量库
    embeddings = build_embeddings()
    vectorstore = load_vectorstore(embeddings)

    if vectorstore is None:
        print("❌ 还没有向量库！请先运行: python main.py ingest")
        return

    # 构建 RAG 链
    model = build_model()
    rag_chain, retriever = build_rag_chain(vectorstore, model)

    # 先看看检索到了哪些文本块（让用户知道"模型是根据什么回答的"）
    if show_sources:
        print("\n📄 检索到的相关文档片段：")
        print("-" * 50)
        retrieved_docs = retriever.invoke(question)
        for i, doc in enumerate(retrieved_docs):
            filename = os.path.basename(doc.metadata.get("source", "未知"))
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i+1}] 来源: {filename}")
            print(f"      内容: {preview}...")
            print()
        print("-" * 50)

    # 调用 RAG 链生成回答
    print("🤖 回答：")
    print("-" * 50)
    answer = rag_chain.invoke(question)
    print(answer)
    print("-" * 50)


def cmd_list():
    """
    list 子命令：列出知识库中的文档。
    """
    if not os.path.exists(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not files:
        print("data/ 目录下没有 .txt 文件")
        return

    print(f"📚 知识库文档（{DATA_DIR}）：")
    for i, f in enumerate(sorted(files), 1):
        filepath = os.path.join(DATA_DIR, f)
        with open(filepath, "r", encoding="utf-8") as fp:
            content = fp.read()
        print(f"  [{i}] {f}（{len(content)} 字符）")

    # 检查向量库是否已构建（Milvus：检查 collection 是否存在）
    try:
        client = MilvusClient(uri=MILVUS_URI)
        has_collection = client.has_collection(MILVUS_COLLECTION)
        client.close()
    except Exception:
        has_collection = False

    if has_collection:
        print(f"\n✅ 向量库已构建（Milvus collection: {MILVUS_COLLECTION}）")
    else:
        print(f"\n⬜ 向量库未构建，请运行: python main.py ingest")


def cmd_demo():
    """
    demo 子命令：完整演示 RAG 的 5 个步骤。

    这个命令会把 ingest + ask 的过程串起来，逐步打印每一步的结果，
    方便你对照代码理解整个 RAG 流程。
    """
    print("=" * 60)
    print("  RAG Demo：完整流程演示")
    print("=" * 60)

    # ---- 步骤 1：加载文档 ----
    print("\n📥 步骤 1/5：加载文档（Load）")
    docs = load_documents()

    # ---- 步骤 2：切分文本 ----
    print("\n✂️  步骤 2/5：切分文本（Split）")
    chunks = split_documents(docs)

    # ---- 步骤 3：构建 Embeddings ----
    print("\n🔢 步骤 3/5：构建向量嵌入模型（Embed）")
    embeddings = build_embeddings()

    # ---- 步骤 4：构建向量库 ----
    print("\n🗄️  步骤 4/5：构建向量库（Store）")
    vectorstore = build_vectorstore(chunks, embeddings)

    # ---- 步骤 5：检索 + 生成 ----
    print("\n🔍 步骤 5/5：检索 + 生成（Retrieve & Generate）")
    model = build_model()
    rag_chain, retriever = build_rag_chain(vectorstore, model)

    # 用 3 个预设问题演示
    questions = [
        "什么是 LangChain？",
        "RAG 的标准流程是什么？",
        "LangGraph 和 LangChain 有什么区别？",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 50}")
        print(f"问题 {i}：{question}")
        print(f"{'─' * 50}")

        # 先看检索到了什么
        retrieved_docs = retriever.invoke(question)
        print(f"检索到 {len(retrieved_docs)} 个相关片段：")
        for j, doc in enumerate(retrieved_docs):
            filename = os.path.basename(doc.metadata.get("source", "未知"))
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"  [{j+1}] {filename}: {preview}...")

        # 生成回答
        print(f"\n🤖 回答：")
        answer = rag_chain.invoke(question)
        print(answer)

    print(f"\n{'=' * 60}")
    print("  Demo 完成！这就是 RAG 的完整流程。")
    print("=" * 60)


# ==================================================================
# CLI 入口：解析命令行参数，分发到对应函数
# ==================================================================

def main():
    """
    CLI 入口函数。

    argparse 是 Python 标准库的命令行参数解析器。
    subparser（子命令）模式：python main.py <子命令> [参数]
    类似 git 的用法：git add / git commit / git push
    """

    parser = argparse.ArgumentParser(
        description="P1 RAG 入门：基于本地文档的问答系统",
    )

    # 创建子命令解析器
    sub = parser.add_subparsers(dest="cmd", help="可用命令")

    # ingest 子命令：构建向量库（无额外参数）
    sub.add_parser("ingest", help="把 data/ 下的文档灌进向量库")

    # ask 子命令：提问
    a = sub.add_parser("ask", help="基于本地文档回答问题")
    a.add_argument("question", help="你的问题")
    a.add_argument("--no-sources", action="store_true", help="不显示检索来源")

    # list 子命令：列出文档
    sub.add_parser("list", help="列出知识库中的文档")

    # demo 子命令：完整流程演示
    sub.add_parser("demo", help="演示 RAG 完整 5 步流程")

    # 解析命令行参数
    args = parser.parse_args()

    # 根据子命令分发到对应函数
    if args.cmd == "ingest":
        cmd_ingest()
    elif args.cmd == "ask":
        cmd_ask(args.question, show_sources=not args.no_sources)
    elif args.cmd == "list":
        cmd_list()
    elif args.cmd == "demo":
        cmd_demo()
    else:
        # 没传子命令时，打印帮助信息
        parser.print_help()


# ==================================================================
# 入口守卫：只有直接运行这个文件时才执行 main()
# ==================================================================
# 如果这个文件被别的文件 import，下面的代码不会执行
# 这是 Python 的标准写法，几乎所有可运行脚本都有这行
if __name__ == "__main__":
    main()
