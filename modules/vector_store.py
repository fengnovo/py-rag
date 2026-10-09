"""
④ 向量存储模块
===============
将嵌入向量存入向量数据库，支持高效相似度检索。

向量数据库选型:
--------------
┌─────────────┬──────────────────┬──────────────────┬──────────────────┐
│   特性       │ Qdrant           │ Milvus           │ pgvector         │
├─────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 类型         │ 专业向量DB       │ 专业向量DB       │ PG 插件          │
│ 多租户       │ ✅ payload filter│ ✅ partition key │ ✅ schema 隔离   │
│ 部署复杂度   │ 低 (单二进制)    │ 高 (依赖 etcd)   │ 低 (PG 扩展)     │
│ 适合场景     │ 中小规模         │ 大规模/高并发    │ 已有 PG 的项目   │
│ 混合检索     │ ✅ 内置 BM25     │ ❌ 需外部        │ ✅ 配合 tsvector │
│ 语言         │ Rust             │ Go/C++           │ C                │
└─────────────┴──────────────────┴──────────────────┴──────────────────┘

本 Demo 使用 FAISS 做本地演示（无需额外服务）
"""

from typing import List, Dict, Optional
import numpy as np
from rich.console import Console

console = Console()


class VectorStore:
    """向量存储基类"""

    def add(self, chunks: List[Dict]) -> None:
        raise NotImplementedError

    def search(self, query_vector: List[float], top_k: int = 20) -> List[Dict]:
        raise NotImplementedError


class FAISSStore(VectorStore):
    """
    FAISS 向量存储（本地 Demo 用）

    FAISS (Facebook AI Similarity Search):
    - Meta 开源的向量检索库
    - 支持精确搜索 (IndexFlatIP) 和近似搜索 (IndexHNSW)
    - 纯内存，适合小规模数据和原型开发
    """

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = None
        self.chunks = []  # 存储原始数据

        console.print(f"  [cyan]初始化 FAISS 索引[/cyan] (维度: {dimension})")

    def add(self, chunks: List[Dict]) -> None:
        """
        将向量和对应文本加入索引

        Args:
            chunks: 包含 embedding 和 content 的文本块列表
        """
        try:
            import faiss

            # 创建索引 (IndexFlatIP = 精确内积搜索)
            # 因为向量已归一化，内积 = 余弦相似度
            self.index = faiss.IndexFlatIP(self.dimension)

            # 提取向量矩阵
            vectors = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)

            # 加入索引
            self.index.add(vectors)
            self.chunks = chunks

            console.print(f"  [green]✅ 已索引 {len(chunks)} 个向量[/green]")

        except ImportError:
            console.print("  [yellow]⚠️  FAISS 未安装，使用 NumPy 暴力搜索[/yellow]")
            self.chunks = chunks

    def search(self, query_vector: List[float], top_k: int = 20) -> List[Dict]:
        """
        向量相似度搜索

        Args:
            query_vector: 查询向量
            top_k: 返回最相似的 K 个结果

        Returns:
            相似的文本块列表（带相似度分数）
        """
        query = np.array([query_vector], dtype=np.float32)

        if self.index is not None:
            # FAISS 搜索
            scores, indices = self.index.search(query, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                chunk["retrieval_method"] = "vector"
                results.append(chunk)
            return results
        else:
            # NumPy 暴力搜索（降级方案）
            vectors = np.array([c["embedding"] for c in self.chunks], dtype=np.float32)
            scores = vectors @ query.T
            scores = scores.flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                chunk["retrieval_method"] = "vector"
                results.append(chunk)
            return results


class QdrantStore(VectorStore):
    """
    Qdrant 向量存储（生产环境）

    多租户实现: 通过 payload filter
    每个租户的数据带有 tenant_id 标签，查询时自动过滤
    """

    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection_name: str = "rag_demo"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        console.print(f"  [cyan]连接 Qdrant[/cyan] {host}:{port}")
        # 实际实现:
        # from qdrant_client import QdrantClient
        # self.client = QdrantClient(host=host, port=port)

    def add(self, chunks: List[Dict]) -> None:
        console.print("  [yellow]Qdrant add() - 需要 qdrant-client[/yellow]")

    def search(self, query_vector: List[float], top_k: int = 20) -> List[Dict]:
        console.print("  [yellow]Qdrant search() - 需要 qdrant-client[/yellow]")
        return []


class MilvusStore(VectorStore):
    """
    Milvus 向量存储（生产环境，大规模）

    多租户实现: 通过 partition key
    """

    def __init__(self, host: str = "localhost", port: int = 19530):
        console.print(f"  [cyan]连接 Milvus[/cyan] {host}:{port}")

    def add(self, chunks: List[Dict]) -> None:
        console.print("  [yellow]Milvus add() - 需要 pymilvus[/yellow]")

    def search(self, query_vector: List[float], top_k: int = 20) -> List[Dict]:
        console.print("  [yellow]Milvus search() - 需要 pymilvus[/yellow]")
        return []


class PGVectorStore(VectorStore):
    """
    pgvector 向量存储（已有 PostgreSQL 的项目）

    优势: 不需要额外的向量数据库服务
    多租户: 通过 schema 或 tenant_id 列隔离
    """

    def __init__(self, connection_string: str = ""):
        console.print(f"  [cyan]连接 pgvector[/cyan]")

    def add(self, chunks: List[Dict]) -> None:
        console.print("  [yellow]pgvector add() - 需要 psycopg2[/yellow]")

    def search(self, query_vector: List[float], top_k: int = 20) -> List[Dict]:
        console.print("  [yellow]pgvector search() - 需要 psycopg2[/yellow]")
        return []


def create_vector_store(store_type: str = "faiss", **kwargs) -> VectorStore:
    """工厂方法: 根据配置创建向量存储"""
    stores = {
        "faiss": FAISSStore,
        "qdrant": QdrantStore,
        "milvus": MilvusStore,
        "pgvector": PGVectorStore,
    }
    store_class = stores.get(store_type, FAISSStore)
    return store_class(**kwargs)


def store_vectors(chunks: List[Dict], store_type: str = "faiss") -> VectorStore:
    """
    将嵌入后的文本块存入向量库

    Args:
        chunks: 包含 embedding 的文本块
        store_type: 向量库类型

    Returns:
        向量存储实例
    """
    console.print(f"\n[bold blue]═══ ④ 向量存储 ═══[/bold blue]")
    console.print(f"存储类型: [cyan]{store_type}[/cyan]")

    store = create_vector_store(store_type)
    store.add(chunks)

    console.print(f"[bold green]✅ 向量存储完成[/bold green]")
    return store
