# LangChain + LangGraph 学习项目

> 主线项目：**Deep Research Agent（深度研究助手）** —— 对标 OpenAI Deep Research / Perplexity / Manus / Genspark
> 通过一个"会自己查资料、写报告"的智能体，把 LangChain 和 LangGraph 的核心能力串起来，从易到难分阶段落地。

---

## 学习路线总览

```
P0 基础热身 → P1 RAG 入门 → P2 Tools & Agents → P3 LangGraph 入门 → P4 综合项目 → P5(可选) 进阶
```

---

## 进度清单

### P0 · 基础热身：LangChain Model I/O 三件套 ✅ 已完成

> 目标：吃透 Model / Prompt / Output + LCEL 链组合，打最稳的地基。
> 目录：`p0_foundations/`

- [x] 搭建项目骨架（`main.py` + `requirements.txt` + `.env.example`）
- [x] 配置项目级虚拟环境 `.venv`，安装依赖（langchain 1.3.14 等）
- [x] **Model**：`ChatOpenAI` 封装，支持 OpenAI / DeepSeek / 通义 / 智谱（OpenAI 兼容协议）
- [x] **Prompt**：`ChatPromptTemplate.from_messages` 结构化提示词（system / user 角色 + 占位符）
- [x] **Output**：`StrOutputParser` 提取纯文本 + `with_structured_output` + Pydantic 结构化输出
- [x] **LCEL 链组合**：`prompt | model | parser` 水管式拼接
- [x] CLI 子命令：`chat` / `translate` / `summarize` / `polish` / `demo`
- [x] 修复 DeepSeek V4 结构化输出报错（思考模式不支持 function_calling → 改用 `json_mode` + prompt 含 "json" 关键词）
- [x] 代码加详细中文注释（Python 新手友好）

**掌握的概念**：`ChatOpenAI`、`ChatPromptTemplate`、`StrOutputParser`、`with_structured_output`、Pydantic、LCEL `|` 管道、`load_dotenv` / `os.getenv`、`argparse` CLI、`if __name__ == "__main__"`

---

### P1 · RAG 入门：让模型能查你自己的资料 ✅ 已完成

> 目标：做本地知识库问答，掌握"检索增强生成"全套组件。
> 目录：`p1_rag/`

- [x] 文档加载：`TextLoader` / `DirectoryLoader`（读 txt 文件，变成 Document 对象）
- [x] 文本切分：`RecursiveCharacterTextSplitter`（把长文档切成小块 chunk，中文友好分隔符）
- [x] 向量嵌入：`OllamaEmbeddings`（用本地 Ollama 的 bge-m3 模型，免费、离线、零 Python 依赖）
- [x] 向量存储：`Milvus`（Docker 独立部署的向量数据库，存向量、做相似度搜索，支持持久化）
- [x] 检索器：`retriever = vectorstore.as_retriever()`（从库里捞最相似的 3 个片段）
- [x] RAG 链：`{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | parser`
- [x] 命令行问答：`python main.py ask "你的问题"`（基于本地文档回答，带来源溯源）
- [x] 引用溯源：显示检索到的文档片段来源（文件名 + 内容预览）
- [ ] 可选：多文档 / 重排序（rerank）— 后续进阶

**掌握的概念**：`Document`、`RecursiveCharacterTextSplitter`、`Embeddings`、`Milvus`、`VectorStore`、`Retriever`、`RunnablePassthrough`、RAG 链组装、向量库持久化

**P1 运行方式**（向量库在 Milvus，不是本地文件）：
```bash
cd ~/Workbuddy/langchain-langGraph/p1_rag
~/Workbuddy/langchain-langGraph/.venv/bin/python main.py ingest   # 把 data/ 文档灌入 Milvus（首次或换文档后）
~/Workbuddy/langchain-langGraph/.venv/bin/python main.py ask "什么是 RAG？"   # 问答（带来源溯源）
~/Workbuddy/langchain-langGraph/.venv/bin/python main.py demo      # 完整 RAG 5 步流程演示
~/Workbuddy/langchain-langGraph/.venv/bin/python main.py list      # 查看知识库文档 + 向量库状态
```

**向量库架构**（Docker 部署的 Milvus 全家桶）：
```
Python 进程 ←gRPC→ Milvus 服务（localhost:19530）
                       ├─ etcd   ：元数据存储
                       ├─ MinIO  ：对象存储（向量数据落盘）
                       └─ Attu   ：可视化管理界面（http://localhost:8000）
```
> 配置在 `p1_rag/.env`：`MILVUS_URI=http://localhost:19530`、`MILVUS_COLLECTION=p1_rag_docs`。可在浏览器打开 `http://localhost:8000`（Attu）直观看到 collection 里的向量数据。

---

### P2 · Tools & Agents：让模型会用工具 ⬜ 待办

> 目标：给模型装上"手"，能调函数、搜网页、查数据库。
> 预计目录：`p2_tools_agents/`

- [ ] 自定义工具：`@tool` 装饰器（把普通函数变成模型可调用的工具）
- [ ] 内置工具：搜索（Tavily / DuckDuckGo）、计算器、Python REPL
- [ ] Tool calling：`model.bind_tools([...])`（让模型决定何时调哪个工具）
- [ ] ReAct Agent：用 `create_react_agent` 搭"思考→行动→观察→再思考"循环
- [ ] Agent Executor：运行 agent、管理工具调用循环、处理中间步骤
- [ ] 命令行演示：`python main.py agent "今天北京天气怎样？"`

**将掌握的概念**：`@tool`、`bind_tools`、`tool_call`、ReAct 模式、`create_react_agent`、`AgentExecutor`

---

### P3 · LangGraph 入门：有状态的可循环编排 ⬜ 待办

> 目标：掌握状态图编排，理解 LangGraph 的"图"思维，区别于 LangChain 的"链"。
> 预计目录：`p3_langgraph/`

- [ ] StateGraph 基础：定义 `State`（TypedDict）、添加节点（`add_node`）、添加边（`add_edge`）
- [ ] 条件边：`add_conditional_edges`（根据状态决定下一步走哪个节点）
- [ ] 循环控制：`END` 终止 + 条件边实现"写→审→改"循环
- [ ] 入口与编译：`set_entry_point` / `compile()` → 得到可运行的 graph
- [ ] 实战：多轮"写作→审稿→修订"循环器（写一段→自审→不达标就改→达标为止）
- [ ] 可视化：`graph.get_graph().draw_mermaid()` 看流程图

**将掌握的概念**：`StateGraph`、`State`（TypedDict）、`Node`、`Edge`、`Conditional Edge`、`END`、`compile()`、循环与终止

---

### P4 · 综合项目：Deep Research Agent 🔶 待办（核心目标）

> 目标：把 P0~P3 的能力组装成一个"会自己查资料、写报告"的智能体。
> 预计目录：`p4_deep_research/`

- [ ] 架构设计：规划 → 搜索 → 阅读 → 整合 → 写报告 的多节点状态图
- [ ] 规划节点：把用户问题拆成多个子问题 / 搜索关键词
- [ ] 搜索节点：调用搜索工具，收集多源信息
- [ ] 阅读节点：对搜索结果做摘要 / 信息抽取（复用 P1 RAG + P2 Tools）
- [ ] 整合节点：判断信息是否充分，不充分则回到搜索（条件边 + 循环）
- [ ] 写报告节点：把整合后的信息组织成结构化报告
- [ ] 状态管理：用 `State` 在节点间传递"问题/搜索结果/摘要/草稿/迭代次数"
- [ ] 流式输出：用 `astream_events` 实时展示 agent 思考与写作过程
- [ ] 命令行入口：`python main.py research "LangGraph 和 LangChain 有什么区别？"`

**将掌握的概念**：多节点图编排、条件循环、状态在节点间流转、RAG + Tools + Agent 综合应用、流式输出

---

### P5 · 进阶（可选）⬜ 待办

> 目标：往生产级方向打磨，按兴趣选做。

- [ ] 多智能体协作：多个 agent 分工（研究员 / 写手 / 审稿人），用 LangGraph 做调度
- [ ] Human-in-the-loop：agent 运行中暂停，等人确认/修改后继续（`interrupt`）
- [ ] 持久化与断点续跑：`Checkpointer`（SQLite / PostgreSQL）保存图状态
- [ ] 部署：用 FastAPI / LangServe 把 agent 包装成 HTTP API
- [ ] 前端界面：配一个简单的 Web 聊天页（Streamlit / Gradio）

---

## 当前状态

| 阶段 | 状态 | 目录 |
|------|------|------|
| P0 基础热身 | ✅ 已完成 | `p0_foundations/` |
| P1 RAG 入门 | ✅ 已完成 | `p1_rag/` |
| P2 Tools & Agents | ⬜ 待办 | `p2_tools_agents/` |
| P3 LangGraph 入门 | ⬜ 待办 | `p3_langgraph/` |
| P4 综合项目 | 🔶 待办（核心目标） | `p4_deep_research/` |
| P5 进阶（可选） | ⬜ 待办 | — |

---

## 快速开始（P0）

```bash
# 1. 进入项目目录
cd ~/Workbuddy/langchain-langGraph/p0_foundations

# 2. 配置密钥（编辑 .env 填入真实 API key）
#    .env 已从 .env.example 生成，改里面的 OPENAI_API_KEY 即可

# 3. 用项目虚拟环境运行
~/Workbuddy/langchain-langGraph/.venv/bin/python main.py demo
```

支持的子命令：
```bash
python main.py chat "你好"                        # 直接对话
python main.py translate "Hello" --to 中文         # 翻译
python main.py summarize "一段长文本"               # 总结
python main.py polish "那个事我搞定了" --tone 口语   # 润色
python main.py demo                                # 依次演示 4 个核心概念
```

> PyCharm 调试：Settings → Python Interpreter 指向 `.venv/bin/python` → 配 Run/Debug 填参数 → 打断点 Debug。
