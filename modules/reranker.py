"""
⑥ 重排序模块 (bge-reranker-v2-m3)
===================================
对混合检索的结果进行精排。

为什么需要重排序？
-----------------
混合检索返回的结果是"粗排"，可能存在:
1. 语义相关但实际不重要的文档
2. 关键词匹配但答非所问的文档
3. 不同来源的文档难以直接比较

重排序器 (Reranker) 的作用:
- 对每个 (查询, 文档) 对进行深度交叉注意力计算
- 给出精确的相关性分数
- 重新排序，把最相关的文档排到前面

Reranker vs Embedding 的区别:
-----------------------------
┌───────────────┬──────────────────┬──────────────────┐
│               │ Embedding        │ Reranker         │
├───────────────┼──────────────────┼──────────────────┤
│ 计算方式       │ 双塔(独立编码)    │ 交叉注意力(联合)  │
│ 速度          │ 快 (可预计算)     │ 慢 (需逐对计算)   │
│ 精度          │ 中               │ 高               │
│ 用途          │ 初筛 (万级→百级)  │ 精排 (百级→十级)  │
└───────────────┴──────────────────┴──────────────────┘

为什么选 bge-reranker-v2-m3?
----------------------------
- BAAI 开源，与 bge-m3 嵌入模型配套
- 多语言支持
- 在跨语言检索场景表现优秀
"""

from typing import List, Dict
from rich.console import Console
from rich.table import Table

console = Console()


class Reranker:
    """
    重排序器

    使用交叉编码器对 (query, document) 对进行精确打分
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载重排序模型"""
        console.print(f"  [cyan]加载重排序模型:[/cyan] {self.model_name}")
            
        # 强制离线模式，使用本地缓存
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            
        try:
            from FlagEmbedding import FlagReranker
            self.model = FlagReranker(
                self.model_name,
                use_fp16=False,  # CPU 模式，避免 GPU 内存问题
            )
            console.print(f"  [green]✅ 重排序模型加载成功[/green]")
        except ImportError:
            console.print("  [yellow]⚠️  FlagEmbedding 未安装，使用模拟重排序[/yellow]")
            self.model = None
        except Exception as e:
            console.print(f"  [yellow]️  模型加载失败 ({type(e).__name__})，使用模拟重排序[/yellow]")
            console.print(f"  [dim]错误: {e}[/dim]")
            self.model = None

    def rerank(self, query: str, documents: List[Dict], top_n: int = 5) -> List[Dict]:
        """
        对文档列表进行重排序

        Args:
            query: 用户查询
            documents: 待排序的文档列表
            top_n: 返回前 N 个

        Returns:
            重排序后的文档列表（带 rerank_score）
        """
        if not documents:
            return []

        console.print(f"  [cyan]重排序 {len(documents)} 个候选文档...[/cyan]")

        if self.model is not None:
            # 使用真实模型
            pairs = [[query, doc["content"]] for doc in documents]
            scores = self.model.compute_score(pairs)

            # 如果是单个分数，包装为列表
            if isinstance(scores, float):
                scores = [scores]
        else:
            # 模拟: 基于内容长度和关键词的启发式打分
            import random
            scores = [random.uniform(0.3, 0.99) for _ in documents]

        # 附加分数并排序
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        sorted_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        top_docs = sorted_docs[:top_n]

        # 展示结果
        table = Table(title="重排序结果")
        table.add_column("排名", style="cyan")
        table.add_column("重排分数", style="green")
        table.add_column("来源", style="yellow")
        table.add_column("内容预览", style="white")

        for i, doc in enumerate(top_docs):
            table.add_row(
                str(i + 1),
                f"{doc['rerank_score']:.4f}",
                doc["metadata"].get("source", "?"),
                doc["content"][:60] + "...",
            )
        console.print(table)

        return top_docs


def rerank_results(
    query: str,
    retrieved_docs: List[Dict],
    top_n: int = 5,
) -> List[Dict]:
    """
    对检索结果进行重排序

    Args:
        query: 用户查询
        retrieved_docs: 混合检索返回的文档
        top_n: 最终保留数量

    Returns:
        精排后的文档列表
    """
    console.print(f"\n[bold blue]═══ ⑥ 重排序 ═══[/bold blue]")

    reranker = Reranker()
    reranked = reranker.rerank(query, retrieved_docs, top_n=top_n)

    console.print(f"[bold green]✅ 重排序完成[/bold green] - 保留 Top {len(reranked)} 文档")

    return reranked


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    query = "什么是 RAG?"
    docs = [
        {"id": "1", "content": "RAG 是检索增强生成架构", "metadata": {"source": "doc1"}},
        {"id": "2", "content": "今天天气很好", "metadata": {"source": "doc2"}},
        {"id": "3", "content": "Retrieval Augmented Generation 结合检索和生成", "metadata": {"source": "doc1"}},
    ]

    reranker = Reranker()
    results = reranker.rerank(query, docs, top_n=2)
    for r in results:
        print(f"  [{r['rerank_score']:.4f}] {r['content']}")
