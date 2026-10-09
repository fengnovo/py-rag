"""
③ 向量嵌入模块
===============
将文本块转换为高维向量（数学表示）。

什么是向量嵌入？
---------------
将文本映射到高维空间中的一个点（向量），使得:
- 语义相近的文本 → 向量距离近（余弦相似度高）
- 语义无关的文本 → 向量距离远

例如:
"猫在睡觉" 和 "小猫正在打盹" → 余弦相似度 ≈ 0.9
"猫在睡觉" 和 "股票市场下跌" → 余弦相似度 ≈ 0.1

为什么选 bge-m3？
-----------------
- BAAI（智源研究院）开源
- 多语言支持（100+ 语言），中英文混合场景表现优秀
- 向量维度 1024，表达能力强
- 最大输入 8192 tokens
- 在 MTEB 基准测试中名列前茅

 alternatives:
- text-embedding-3-small/large (OpenAI) - 英文为主
- mxbai-embed-large - 英文优秀
- bge-large-zh-v1.5 - 中文专精
"""

from typing import List, Dict
import numpy as np
from rich.console import Console

console = Console()


class Embedder:
    """向量嵌入器"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        初始化嵌入模型

        Args:
            model_name: HuggingFace 模型名称
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载嵌入模型"""
        console.print(f"  [cyan]加载嵌入模型:[/cyan] {self.model_name}")
            
        # 强制离线模式，使用本地缓存，不访问 HuggingFace
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            console.print(f"  [green]✅ 模型加载成功[/green] (维度: {self.model.get_embedding_dimension()})")
        except ImportError:
            console.print("  [yellow]️  sentence-transformers 未安装，使用随机向量模拟[/yellow]")
            self.model = None
        except Exception as e:
            console.print(f"  [yellow]⚠️  模型加载失败 ({type(e).__name__})，使用随机向量模拟[/yellow]")
            console.print(f"  [dim]提示: 设置 HF_ENDPOINT=https://hf-mirror.com 使用国内镜像[/dim]")
            self.model = None

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量嵌入文本

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            向量列表 [[0.1, 0.2, ...], ...]
        """
        if self.model is not None:
            # 使用真实模型
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,  # 归一化 → 余弦相似度 = 内积
                show_progress_bar=True,
            )
            return embeddings.tolist()
        else:
            # 模拟: 生成随机向量（仅用于 demo 演示流程）
            console.print("  [yellow]⚠️  使用随机向量模拟嵌入[/yellow]")
            dim = 1024
            return [
                (np.random.randn(dim) / np.linalg.norm(np.random.randn(dim))).tolist()
                for _ in texts
            ]

    def embed_query(self, query: str) -> List[float]:
        """
        嵌入查询（单条文本）

        注意: 查询和文档的嵌入方式可能不同
        - bge-m3: 查询前缀 "query: " vs 文档前缀 "passage: "
        - 但 bge-m3 也能自动检测，这里简化处理
        """
        if self.model is not None:
            embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
            )
            return embedding[0].tolist()
        else:
            dim = 1024
            vec = np.random.randn(dim)
            return (vec / np.linalg.norm(vec)).tolist()


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    对所有文本块进行嵌入

    Args:
        chunks: 文本块列表 [{"id": ..., "content": ..., "metadata": ...}, ...]

    Returns:
        添加了 embedding 字段的文本块列表
    """
    console.print(f"\n[bold blue]═══ ③ 向量嵌入 ═══[/bold blue]")

    embedder = Embedder()

    # 提取所有文本
    texts = [chunk["content"] for chunk in chunks]
    console.print(f"待嵌入: {len(texts)} 个文本块")

    # 批量嵌入
    embeddings = embedder.embed_texts(texts)

    # 将向量附加到每个 chunk
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    console.print(f"[bold green]✅ 嵌入完成[/bold green] - 向量维度: {len(embeddings[0])}")

    # 演示: 显示两个向量的相似度
    if len(embeddings) >= 2:
        sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        console.print(f"  示例: 块0 与 块1 的余弦相似度 = {sim:.4f}")

    return chunks


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    embedder = Embedder()

    texts = [
        "RAG 是检索增强生成的缩写",
        "Retrieval Augmented Generation  combines retrieval and generation",
        "今天天气真好",
    ]

    embeddings = embedder.embed_texts(texts)

    print(f"\n向量维度: {len(embeddings[0])}")
    print(f"\n计算余弦相似度:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = np.dot(embeddings[i], embeddings[j])
            print(f"  '{texts[i][:20]}...' vs '{texts[j][:20]}...' = {sim:.4f}")
