"""
⑤ 混合检索模块 (BM25 + 向量 + RRF)
====================================
结合关键词匹配和语义理解，用 RRF 算法融合排序。

为什么需要混合检索？
-------------------
单独使用向量检索的弱点:
  - 对专有名词、产品型号等精确匹配不够好
  - 例: 用户问 "GPT-4o 的上下文窗口是多少" → 向量可能匹配到 "GPT-4" 的内容

单独使用 BM25 的弱点:
  - 无法理解同义词和语义相似
  - 例: "如何提升模型效果" vs "怎样改善模型表现" → BM25 匹配不到

混合检索 = 两者互补:
  - BM25: 精确匹配关键词
  - 向量: 理解语义
  - RRF: 融合两种结果

什么是 RRF (Reciprocal Rank Fusion)?
------------------------------------
一种简单有效的排序融合算法:

  RRF_score(d) = Σ 1 / (k + rank_i(d))

其中:
  - d: 文档
  - k: 常数（通常 60），防止排名靠前的文档分数过高
  - rank_i(d): 文档在第 i 个检索方法中的排名

示例:
  文档A: BM25 排名 2, 向量排名 5
  → RRF = 1/(60+2) + 1/(60+5) = 0.0161 + 0.0154 = 0.0315

  文档B: BM25 排名 1, 向量排名 10
  → RRF = 1/(60+1) + 1/(60+10) = 0.0164 + 0.0143 = 0.0307

  → 文档A 排前面（因为它在两种方法中都比较靠前）
"""

from typing import List, Dict
import re
from collections import defaultdict
from rich.console import Console
from rich.table import Table

console = Console()


class BM25Retriever:
    """
    BM25 关键词检索

    BM25 (Best Matching 25) 是经典的文本检索算法:
    - 基于词频 (TF) 和逆文档频率 (IDF)
    - 词在文档中出现越多 → 分数越高
    - 词在整个语料中越罕见 → 分数越高
    """

    def __init__(self, chunks: List[Dict]):
        try:
            from rank_bm25 import BM25Okapi
            # 分词（简单按空格和标点切分）
            self.corpus = [self._tokenize(c["content"]) for c in chunks]
            self.bm25 = BM25Okapi(self.corpus)
            self.chunks = chunks
            console.print(f"  [green]✅ BM25 索引构建完成[/green] ({len(chunks)} 文档)")
        except ImportError:
            console.print("  [yellow]⚠️  rank-bm25 未安装，BM25 不可用[/yellow]")
            self.bm25 = None
            self.chunks = chunks

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（生产环境建议用 jieba 中文分词）"""
        # 中文按字切分，英文按词切分
        text = text.lower()
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)
        return tokens

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """BM25 检索"""
        if self.bm25 is None:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # 排序取 top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = float(scores[idx])
            chunk["retrieval_method"] = "bm25"
            results.append(chunk)

        return results


def reciprocal_rank_fusion(
    results_list: List[List[Dict]],
    weights: List[float] = None,
    k: int = 60,
) -> List[Dict]:
    """
    RRF (Reciprocal Rank Fusion) 融合算法

    Args:
        results_list: 多个检索结果列表 [bm25_results, vector_results, ...]
        weights: 每种检索方法的权重 [0.5, 0.5]
        k: RRF 常数（默认 60）

    Returns:
        融合后的排序结果
    """
    if weights is None:
        weights = [1.0] * len(results_list)

    # 计算每个文档的 RRF 分数
    rrf_scores = defaultdict(float)
    doc_map = {}  # id → chunk 数据

    for results, weight in zip(results_list, weights):
        for rank, chunk in enumerate(results):
            doc_id = chunk.get("id", id(chunk))
            # RRF 公式: weight / (k + rank)
            # rank 从 0 开始，所以实际是 k + rank + 1
            rrf_scores[doc_id] += weight / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = chunk

    # 按 RRF 分数排序
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        chunk = doc_map[doc_id].copy()
        chunk["rrf_score"] = rrf_scores[doc_id]
        chunk["retrieval_method"] = "hybrid_rrf"
        fused_results.append(chunk)

    return fused_results


def hybrid_retrieve(
    query: str,
    query_vector: List[float],
    vector_store,
    bm25_retriever: BM25Retriever,
    top_k: int = 20,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
) -> List[Dict]:
    """
    混合检索: BM25 + 向量 → RRF 融合

    Args:
        query: 用户查询文本
        query_vector: 查询向量
        vector_store: 向量存储实例
        bm25_retriever: BM25 检索器
        top_k: 每种方法初筛数量
        bm25_weight: BM25 权重
        vector_weight: 向量检索权重
        rrf_k: RRF 常数

    Returns:
        融合排序后的文档列表
    """
    console.print(f"\n[bold blue]═══ ⑤ 混合检索 ═══[/bold blue]")
    console.print(f"查询: [yellow]'{query}'[/yellow]")

    # 1. BM25 检索（关键词匹配）
    console.print(f"\n  [cyan]BM25 关键词检索...[/cyan]")
    bm25_results = bm25_retriever.search(query, top_k=top_k)
    console.print(f"  BM25 返回 {len(bm25_results)} 条结果")
    if bm25_results:
        console.print(f"  Top1 BM25 分数: {bm25_results[0].get('bm25_score', 'N/A'):.4f}")

    # 2. 向量检索（语义匹配）
    console.print(f"\n  [cyan]向量语义检索...[/cyan]")
    vector_results = vector_store.search(query_vector, top_k=top_k)
    console.print(f"  向量检索返回 {len(vector_results)} 条结果")
    if vector_results:
        console.print(f"  Top1 向量分数: {vector_results[0].get('score', 'N/A'):.4f}")

    # 3. RRF 融合
    console.print(f"\n  [cyan]RRF 融合排序...[/cyan]")
    console.print(f"  权重: BM25={bm25_weight}, 向量={vector_weight}, k={rrf_k}")

    fused = reciprocal_rank_fusion(
        results_list=[bm25_results, vector_results],
        weights=[bm25_weight, vector_weight],
        k=rrf_k,
    )

    console.print(f"  融合后: [green]{len(fused)}[/green] 条结果")

    # 展示融合结果
    if fused:
        table = Table(title="RRF 融合结果 Top 5")
        table.add_column("排名", style="cyan")
        table.add_column("来源", style="green")
        table.add_column("RRF分数", style="yellow")
        table.add_column("内容预览", style="white")

        for i, doc in enumerate(fused[:5]):
            table.add_row(
                str(i + 1),
                doc["metadata"].get("source", "?"),
                f"{doc.get('rrf_score', 0):.4f}",
                doc["content"][:50] + "...",
            )
        console.print(table)

    return fused


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    # 模拟 RRF
    bm25_results = [
        {"id": "A", "content": "文档A", "bm25_score": 10},
        {"id": "B", "content": "文档B", "bm25_score": 8},
        {"id": "C", "content": "文档C", "bm25_score": 5},
    ]
    vector_results = [
        {"id": "B", "content": "文档B", "score": 0.95},
        {"id": "D", "content": "文档D", "score": 0.90},
        {"id": "A", "content": "文档A", "score": 0.85},
    ]

    fused = reciprocal_rank_fusion([bm25_results, vector_results])
    print("\nRRF 融合结果:")
    for doc in fused:
        print(f"  {doc['id']}: RRF={doc['rrf_score']:.4f}")
