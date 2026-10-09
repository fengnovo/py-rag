# RAG (Retrieval-Augmented Generation) 完整链路 Demo

## 📖 这个 Demo 解释了什么？

这是一个**生产级 RAG 系统**的完整实现链路，展示了从文档到答案的每一个关键环节。

## 🔗 RAG 完整链路图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG 完整链路                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ① 文档解析        ② 文本分块        ③ 向量嵌入                            │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                           │
│  │ MinerU   │ ──▶ │ 递归分块  │ ──▶ │  bge-m3  │                           │
│  │unstructured│    │ 512+10%  │     │  嵌入    │                           │
│  └──────────┘     └──────────┘     └──────────┘                           │
│                                         │                                   │
│                                         ▼                                   │
│  ⑥ LLM生成         ⑤ 重排序          ④ 向量库                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                           │
│  │Qwen/GPT  │ ◀── │bge-      │ ◀── │ Qdrant/  │                           │
│  │ via      │     │reranker  │     │ Milvus/  │                           │
│  │ LiteLLM  │     │ -v2-m3   │     │ pgvector │                           │
│  └──────────┘     └──────────┘     └──────────┘                           │
│       │                                  ▲                                  │
│       │          ⑦ 混合检索              │                                  │
│       └────────▶ ┌──────────┐ ──────────┘                                  │
│                  │BM25+向量 │                                               │
│                  │  RRF     │                                               │
│                  └──────────┘                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ⑧ 评估 + ⑨ 追踪                                                           │
│  ┌──────────────────────────────────────┐                                  │
│  │  RAGAS 评估  +  Langfuse 追踪       │                                  │
│  └──────────────────────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📚 各环节详解

### ① 文档解析 (MinerU / unstructured)

**作用**: 将各种格式的文档（PDF、Word、HTML等）转换为纯文本

**为什么需要？**
- 原始文档格式复杂，包含表格、图片、排版信息
- LLM 只能处理纯文本，需要先"清洗"

**工具选择**:
- **MinerU**: 开源文档解析工具，擅长 PDF 解析，保留文档结构
- **unstructured**: Python 库，支持多种格式，开箱即用

### ② 文本分块 (Recursive Chunking)

**作用**: 将长文档切分成适合嵌入的小块

**参数解释**:
- **512 tokens**: 每个块的目标大小（约 400-500 汉字）
- **10% 重叠**: 块与块之间有 51 tokens 的重叠，防止语义被截断

**为什么递归分块？**
- 先按段落切分，如果太大再按句子，再太大按词
- 保持语义完整性，避免在句子中间切断

### ③ 向量嵌入 (bge-m3)

**作用**: 将文本块转换为高维向量（数学表示）

**为什么用 bge-m3？**
- BAAI 开源的多语言嵌入模型
- 支持中文、英文等 100+ 语言
- 向量维度 1024，语义表达能力强

**原理**: 语义相近的文本，向量距离更近

### ④ 向量数据库 (Qdrant / Milvus / pgvector)

**作用**: 存储和高效检索向量

**选择建议**:
- **Qdrant/Milvus**: 专业向量数据库，支持多租户，适合大规模生产
- **pgvector**: PostgreSQL 插件，适合已有 PG 的项目，减少运维成本

### ⑤ 混合检索 (BM25 + 向量 RRF)

**作用**: 从向量库中找出与问题最相关的文本块

**为什么混合？**
- **BM25**: 关键词匹配（精确匹配专有名词）
- **向量检索**: 语义匹配（理解同义词、近义表达）
- **RRF (Reciprocal Rank Fusion)**: 融合两种检索结果的排序算法

### ⑥ 重排序 (bge-reranker-v2-m3)

**作用**: 对检索结果精排，提高最终输入 LLM 的质量

**为什么需要？**
- 初筛可能包含"看起来相关但实际不重要"的内容
- Reranker 对每个 (问题, 文档) 对打分，重新排序

### ⑦ LLM 生成 (Qwen/GPT via LiteLLM)

**作用**: 基于检索到的上下文生成最终答案

**LiteLLM 的作用**:
- 统一的 API 接口调用不同 LLM（OpenAI、Qwen、Claude等）
- 方便切换模型、负载均衡、失败重试

### ⑧ 评估 (RAGAS + 黄金集)

**作用**: 量化评估 RAG 系统的质量

**RAGAS 指标**:
- **Faithfulness**: 答案是否忠实于检索到的上下文
- **Answer Relevancy**: 答案是否回答了用户问题
- **Context Precision**: 检索的上下文是否精准
- **Context Recall**: 是否检索到了所有需要的信息

**黄金集**: 人工标注的 (问题, 标准答案, 参考上下文) 测试集

### ⑨ 追踪 (Langfuse)

**作用**: 监控和调试 RAG 系统的每次运行

**追踪内容**:
- 每次查询的完整链路（检索了哪些文档、用了哪些块、生成了什么答案）
- 延迟、token 消耗、成本
- 便于定位问题和优化

## 🚀 运行 Demo

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 3-1. 验证 Qdrant 连接
python -c "from qdrant_client import QdrantClient; c = QdrantClient(host='127.0.0.1', port=56333); print('✅ Qdrant OK:', c.get_collections())"

# 3-2. 逐步验证
    # 验证索引构建   ， 会从 hf-mirror.com 下载 bge-m3 模型 2.27G大小
export HF_ENDPOINT=https://hf-mirror.com
python verify_steps.py --step 1-4
    # 验证查询链路   ， 会从 hf-mirror.com 下载 bge-reranker-v2-m3 模型 2.27G大小
python verify_steps.py --step 5-7

# 4. 运行完整链路
python main.py --skip-eval

# 5. 运行单个模块测试
python test_modules.py
```

## 📁 项目结构

```
py-rag/
├── README.md              # 本文件
├── requirements.txt       # 依赖
├── config.py             # 配置
├── main.py               # 主程序（完整链路演示）
├── modules/
│   ├── __init__.py
│   ├── document_parser.py    # ① 文档解析
│   ├── text_chunker.py       # ② 文本分块
│   ├── embedder.py           # ③ 向量嵌入
│   ├── vector_store.py       # ④ 向量存储
│   ├── hybrid_retriever.py   # ⑤ 混合检索
│   ├── reranker.py           # ⑥ 重排序
│   ├── llm_generator.py      # ⑦ LLM生成
│   ├── evaluator.py          # ⑧ 评估
│   └── tracer.py             # ⑨ 追踪
├── data/
│   └── sample_docs/          # 示例文档
└── tests/
    └── test_modules.py       # 模块测试
```

## 💡 核心概念总结

| 环节 | 输入 | 输出 | 关键工具 |
|------|------|------|----------|
| 文档解析 | PDF/Word/HTML | 纯文本 | MinerU, unstructured |
| 文本分块 | 长文本 | 文本块列表 | LangChain TextSplitter |
| 向量嵌入 | 文本块 | 向量列表 | bge-m3 |
| 向量存储 | 向量+文本 | 向量数据库 | Qdrant, Milvus, pgvector |
| 混合检索 | 查询 | 候选文档 | BM25 + 向量 + RRF |
| 重排序 | 候选文档 | 精排文档 | bge-reranker-v2-m3 |
| LLM生成 | 问题+上下文 | 答案 | Qwen, GPT via LiteLLM |
| 评估 | 答案+标注 | 质量分数 | RAGAS |
| 追踪 | 运行数据 | 监控面板 | Langfuse |


## 链路对照表
**搭一个“会翻书、会查资料、再回答问题”的小机器人。**  
> **一句话总结：** 
> 1. 先把所有资料（书上的字）抠出来 → 剪成小纸条 → 发意思号码 → 放进柜子  ，对应step 1-4    # 验证索引构建
> 2. 用两种方法找 → 再精挑细排 → 让大模型照着答               ，对应step 5-7    # 验证查询链路
> 3. 考试打分 → 全程记录。这样机器人就不是瞎编，而是"先查资料，再回答"。

| 环节 | 对应文件 | 做了什么 | 通俗解释 |
|------|---------|---------|----------|
| **① 解析 MinerU/unstructured** | [document_parser.py](modules/document_parser.py) | 把 PDF/Word 等文档解析为纯文本 | 把 PDF、Word、图片里的字"抠"出来，像把书上的字抄到纸上。MinerU / unstructured 就是两种"抄字机器人"。 |
| **② 分块：递归 512 + 10% 重叠** | [text_chunker.py](modules/text_chunker.py) | 按段落→行→句递归切分，每块 512 tokens，相邻块重叠 51 tokens | 故事太长了，机器人一次看不完，就把长文章剪成小纸条。每张大约 512 个字。剪的时候，上一张结尾和下一张开头留一点点重复，像拼图留边，免得把一句话剪断。"递归"就是：太长就一层层再剪小。 |
| **③ 嵌入 bge-m3** | [embedder.py](modules/embedder.py) | 用 BAAI/bge-m3 把文本块变成 1024 维向量 | 给每张小纸条发一个"意思号码"。意思差不多的纸条，号码也差不多。bge-m3 就是那个"发号码的机器"。 |
| **④ 向量库 Qdrant/Milvus/pgvector** | [vector_store.py](modules/vector_store.py) | 存向量，支持高效相似度搜索（多租户隔离） | 把纸条和号码放进一个大柜子里，方便以后找。Qdrant / Milvus 是不同牌子的柜子。"多租户"就是：每个小朋友有自己带锁的抽屉，不能拿错别人的。pgvector 是：如果家里已经有 PostgreSQL 这个大柜子，就加一个能放"意思号码"的格子。 |
| **⑤ 混合检索 BM25 + 向量 RRF** | [hybrid_retriever.py](modules/hybrid_retriever.py) | BM25 关键词匹配 + 向量语义匹配，用 RRF 算法融合排序 | 有人问问题时，机器人去找纸条。BM25：按"字一样"找，像查关键词。向量：按"意思像"找，像找同义词。RRF：把两种找法的排行榜合起来，像两个老师都推荐谁，谁就更靠前。 |
| **⑥ 重排 bge-reranker-v2-m3** | [reranker.py](modules/reranker.py) | 交叉注意力精排，从 20 个候选中挑出最相关的 5 个 | 找到一堆纸条后，再请一个更细心的老师重新排队。最有用、最相关的放最前面。bge-reranker-v2-m3 就是这个"细心的排队老师"。 |
| **⑦ 生成 Qwen/GPT 经 LiteLLM** | [llm_generator.py](modules/llm_generator.py) | 把上下文+问题喂给 LLM 生成答案，LiteLLM 统一多模型调用 | 把问题 + 最好的几张纸条，交给会说话的大模型，让它写答案。LiteLLM 网关像一个"插线板/前台"，想换不同大模型，就从这里换。 |
| **⑧ 评估 RAGAS + 黄金集** | [evaluator.py](modules/evaluator.py) | 4 个指标量化评估：忠实度、相关性、精确度、召回率 | 准备一套标准答案，像考试卷，叫"黄金集"。再用 RAGAS 这个"评分老师"看机器人答得对不对、有没有根据、有没有瞎编。 |
| **⑨ 追踪 Langfuse** | [tracer.py](modules/tracer.py) | 记录每次查询的完整链路，便于调试和监控 | 给机器人戴一个小摄像头，或者让它写日记。每次它怎么找、怎么答、用了多久，都记下来。Langfuse 就是帮你看记录、查错误的地方。 |


每个模块文件头部都有**详细的中文注释**，解释了为什么需要这个环节、原理是什么、参数怎么调。[main.py](main.py) 串联了所有环节，运行时会在终端逐步打印每个 Step 的执行过程和结果。

解释 FAISS 和 Qdrant 的区别：FAISS 是演示模式（开箱即用、不持久化），Qdrant 是生产模式（需要启动服务、支持多租户）。帮助用户理解为什么代码里同时有这两个向量库。  
**FAISS 是给 Demo 用的，Qdrant 是给生产用的。** 它们不是同时用，而是二选一。

具体来说：

| | FAISS | Qdrant/Milvus |
|---|---|---|
| **定位** | 本地演示，开箱即用 | 生产环境 |
| **需要额外服务？** | 不需要，纯内存库 | 需要先启动 Qdrant 服务 |
| **数据持久化？** | 不持久，程序关了就没了 | 持久化存储 |
| **多租户？** | 不支持 | 支持 |

从 [config.py](config.py#L44) 可以看到：

```python
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss / qdrant / milvus / pgvector
```

默认是 `faiss`，因为 Demo 不想让用户还得先装一个 Qdrant 服务才能跑起来。通过 [vector_store.py](modules/vector_store.py#L186) 的工厂方法切换：把环境变量改成 `qdrant` 就用 Qdrant 了。

**一句话：FAISS 是"演示模式"，Qdrant 是"上线模式"，同一个代码两套配置，按需切换。**