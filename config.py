"""
RAG 完整链路 - 配置文件
========================
集中管理所有环节的参数配置
"""
import os

# 在导入任何第三方库之前设置环境变量
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "1")  # 跳过 LiteLLM GitHub 下载
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # HuggingFace 离线模式
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # Transformers 离线模式

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# ② 文本分块配置
# ============================================================
CHUNK_CONFIG = {
    "chunk_size": 512,          # 每个块的目标 token 数
    "chunk_overlap": 51,        # 重叠 token 数 = 512 * 10%
    "separator": "\n",          # 递归分隔符优先级: \n\n > \n > 。> 空格
    "length_function": "tiktoken",  # 用 tiktoken 计算 token 数
}


# ============================================================
# ③ 嵌入模型配置
# ============================================================
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-m3",           # BAAI 多语言嵌入模型
    "dimension": 1024,                      # 向量维度
    "batch_size": 32,                       # 批量嵌入大小
    "normalize": True,                      # 归一化向量（方便余弦相似度）
}


# ============================================================
# ④ 向量库配置
# ============================================================
# Demo 使用 FAISS（本地），生产环境切换为 Qdrant/Milvus/pgvector
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss / qdrant / milvus / pgvector

QDRANT_CONFIG = {
    "host": os.getenv("QDRANT_HOST", "localhost"),
    "port": int(os.getenv("QDRANT_PORT", "6333")),
    "collection_name": os.getenv("QDRANT_COLLECTION_NAME", "rag_demo"),
    # 多租户: 通过 payload filter 实现 tenant 隔离
    # 同一个 Qdrant 服务下，不同 collection 名称互不干扰
}

MILVUS_CONFIG = {
    "host": os.getenv("MILVUS_HOST", "localhost"),
    "port": int(os.getenv("MILVUS_PORT", "19530")),
    "collection_name": "rag_demo",
}

PGVECTOR_CONFIG = {
    "connection_string": os.getenv(
        "PGVECTOR_CONNECTION",
        "postgresql://user:password@localhost:5432/rag_db"
    ),
    "table_name": "rag_vectors",
}


# ============================================================
# ⑤ 混合检索配置
# ============================================================
RETRIEVAL_CONFIG = {
    "top_k": 20,              # 初筛返回的文档数
    "bm25_weight": 0.5,       # BM25 权重（RRF 融合用）
    "vector_weight": 0.5,     # 向量检索权重
    "rrf_k": 60,              # RRF 算法的常数 k（越大越平滑）
    "final_top_n": 5,         # 重排序后取前 N 个
}


# ============================================================
# ⑥ 重排序配置
# ============================================================
RERANKER_CONFIG = {
    "model_name": "BAAI/bge-reranker-v2-m3",  # 重排序模型
    "top_n": 5,                                 # 最终保留的文档数
    "batch_size": 16,
}


# ============================================================
# ⑦ LLM 生成配置
# ============================================================
# LiteLLM 统一接口，可切换 OpenAI / Qwen / Claude 等
# provider: openai / dashscope / anthropic / 等
LLM_CONFIG = {
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "provider": os.getenv("LLM_PROVIDER", None),  # dashscope / openai / None
    "temperature": 0.1,
    "max_tokens": 2048,
    "api_base": os.getenv("LLM_API_BASE", None),
    "api_key": os.getenv("LLM_API_KEY", None),
}

# ============================================================
# ⑧ 评估配置
# ============================================================
EVAL_CONFIG = {
    "metrics": [
        "faithfulness",        # 答案忠实度
        "answer_relevancy",    # 答案相关性
        "context_precision",   # 上下文精确度
        "context_recall",      # 上下文召回率
    ],
    # 黄金集路径
    "golden_set_path": "data/golden_set.json",
}


# ============================================================
# ⑨ 追踪配置 (Langfuse)
# ============================================================
LANGFUSE_CONFIG = {
    "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    "secret_key": os.getenv("LANGFUSE_SECRET_KEY", ""),
    "base_url": os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    "enabled": os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
}


# ============================================================
# 数据路径
# ============================================================
DATA_DIR = "data"
SAMPLE_DOCS_DIR = "data/sample_docs"
