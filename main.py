#!/usr/bin/env python3
"""
RAG 完整链路 Demo - 主程序
============================
这个脚本串联了 RAG 系统的 9 个核心环节:

  ① 文档解析 (MinerU/unstructured)
  ② 文本分块 (递归 512 + 10% 重叠)
  ③ 向量嵌入 (bge-m3)
  ④ 向量存储 (Qdrant/Milvus/pgvector)
  ⑤ 混合检索 (BM25 + 向量 RRF)
  ⑥ 重排序 (bge-reranker-v2-m3)
  ⑦ LLM 生成 (Qwen/GPT via LiteLLM)
  ⑧ 评估 (RAGAS + 黄金集)
  ⑨ 追踪 (Langfuse)

运行方式:
  python main.py                    # 运行完整链路
  python main.py --query "你的问题"  # 自定义问题
  python main.py --skip-eval        # 跳过评估（更快）
"""

import sys
import os
import json
import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.markdown import Markdown

console = Console()

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CHUNK_CONFIG, EMBEDDING_CONFIG, VECTOR_STORE_TYPE,
    RETRIEVAL_CONFIG, RERANKER_CONFIG, LLM_CONFIG,
    EVAL_CONFIG, LANGFUSE_CONFIG, SAMPLE_DOCS_DIR, DATA_DIR,
)


def print_banner():
    """打印欢迎横幅"""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]RAG 完整链路 Demo[/bold cyan]\n"
        "[dim]文档解析 → 分块 → 嵌入 → 向量库 → 混合检索 → 重排 → 生成 → 评估 → 追踪[/dim]",
        border_style="cyan",
    ))
    console.print()


def print_step_header(step_num: int, title: str):
    """打印步骤标题"""
    console.print()
    console.print(Rule(f"[bold blue]Step {step_num}: {title}[/bold blue]", style="blue"))
    console.print()


# ============================================================
# 索引阶段: 构建知识库
# ============================================================
def build_index():
    """
    构建知识库索引（离线阶段）

    这个函数完成 ①→②→③→④ 四个步骤:
    文档解析 → 文本分块 → 向量嵌入 → 向量存储
    """
    console.print(Panel("[bold]阶段一: 构建知识库索引（离线）[/bold]", border_style="green"))

    # ── ① 文档解析 ──────────────────────────────────────────
    print_step_header(1, "文档解析 (unstructured)")
    console.print("[dim]将 data/sample_docs/ 下的文件解析为纯文本[/dim]")
    console.print("[dim]支持: .txt, .pdf, .docx, .html, .md[/dim]")
    console.print()

    from modules.document_parser import parse_documents
    documents = parse_documents(SAMPLE_DOCS_DIR)

    if not documents:
        console.print("[red]❌ 没有找到文档！请在 data/sample_docs/ 目录下放置文档文件[/red]")
        sys.exit(1)

    # ── ② 文本分块 ──────────────────────────────────────────
    print_step_header(2, "文本分块 (Recursive Chunking)")
    console.print(f"[dim]参数: chunk_size={CHUNK_CONFIG['chunk_size']} tokens, "
                  f"overlap={CHUNK_CONFIG['chunk_overlap']} tokens (10%)[/dim]")
    console.print("[dim]方法: 按 \\n\\n → \\n → 。→ 空格 递归切分[/dim]")
    console.print()

    from modules.text_chunker import chunk_documents
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_CONFIG["chunk_size"],
        chunk_overlap=CHUNK_CONFIG["chunk_overlap"],
    )

    # ── ③ 向量嵌入 ──────────────────────────────────────────
    print_step_header(3, "向量嵌入 (bge-m3)")
    console.print(f"[dim]模型: {EMBEDDING_CONFIG['model_name']}[/dim]")
    console.print(f"[dim]维度: {EMBEDDING_CONFIG['dimension']}[/dim]")
    console.print()

    from modules.embedder import embed_chunks
    chunks = embed_chunks(chunks)

    # ── ④ 向量存储 ──────────────────────────────────────────
    print_step_header(4, f"向量存储 ({VECTOR_STORE_TYPE})")
    console.print(f"[dim]向量库类型: {VECTOR_STORE_TYPE}[/dim]")
    console.print("[dim]Demo 使用 FAISS (本地)，生产可切换 Qdrant/Milvus/pgvector[/dim]")
    console.print()

    from modules.vector_store import store_vectors
    vector_store = store_vectors(chunks, store_type=VECTOR_STORE_TYPE)

    console.print()
    console.print(Panel(
        f"[bold green]✅ 知识库索引构建完成![/bold green]\n"
        f"文档数: {len(documents)}\n"
        f"文本块数: {len(chunks)}\n"
        f"向量维度: {EMBEDDING_CONFIG['dimension']}",
        border_style="green",
    ))

    return documents, chunks, vector_store


# ============================================================
# 查询阶段: 回答问题
# ============================================================
def query_pipeline(query: str, chunks: list, vector_store, tracer=None):
    """
    RAG 查询管线（在线阶段）

    这个函数完成 ⑤→⑥→⑦ 三个步骤:
    混合检索 → 重排序 → LLM 生成
    """
    console.print(Panel(f"[bold]阶段二: 查询 - \"{query}\"[/bold]", border_style="yellow"))

    if tracer:
        tracer.start_trace(query)

    # ── ⑤ 混合检索 ──────────────────────────────────────────
    print_step_header(5, "混合检索 (BM25 + 向量 → RRF)")
    console.print("[dim]BM25: 关键词精确匹配（擅长专有名词）[/dim]")
    console.print("[dim]向量: 语义匹配（擅长同义词、近义表达）[/dim]")
    console.print("[dim]RRF: Reciprocal Rank Fusion 融合两种排序[/dim]")
    console.print()

    from modules.embedder import Embedder
    from modules.hybrid_retriever import BM25Retriever, hybrid_retrieve

    # 嵌入查询
    embedder = Embedder()
    query_vector = embedder.embed_query(query)

    # 构建 BM25 索引
    bm25 = BM25Retriever(chunks)

    # 混合检索
    retrieved = hybrid_retrieve(
        query=query,
        query_vector=query_vector,
        vector_store=vector_store,
        bm25_retriever=bm25,
        top_k=RETRIEVAL_CONFIG["top_k"],
        bm25_weight=RETRIEVAL_CONFIG["bm25_weight"],
        vector_weight=RETRIEVAL_CONFIG["vector_weight"],
        rrf_k=RETRIEVAL_CONFIG["rrf_k"],
    )

    if tracer:
        tracer.add_span("混合检索",
                        {"query": query, "top_k": RETRIEVAL_CONFIG["top_k"]},
                        {"results": len(retrieved)},
                        obs_type="retriever")

    # ── ⑥ 重排序 ──────────────────────────────────────────
    print_step_header(6, "重排序 (bge-reranker-v2-m3)")
    console.print("[dim]对混合检索结果进行精排[/dim]")
    console.print("[dim]Reranker 对每个 (query, doc) 对深度交叉计算，给出精确分数[/dim]")
    console.print()

    from modules.reranker import rerank_results
    reranked = rerank_results(
        query=query,
        retrieved_docs=retrieved,
        top_n=RERANKER_CONFIG["top_n"],
    )

    if tracer:
        tracer.add_span("重排序",
                        {"candidates": len(retrieved)},
                        {"results": len(reranked)},
                        obs_type="retriever")

    # ── ⑦ LLM 生成 ──────────────────────────────────────────
    print_step_header(7, "LLM 生成 (via LiteLLM)")
    console.print(f"[dim]模型: {LLM_CONFIG['model']}[/dim]")
    console.print("[dim]通过 LiteLLM 统一网关调用，可切换 OpenAI/Qwen/Claude 等[/dim]")
    console.print()

    from modules.llm_generator import generate_answer, display_answer
    result = generate_answer(
        query=query,
        retrieved_docs=reranked,
        model=LLM_CONFIG["model"],
        provider=LLM_CONFIG.get("provider"),  # 新增
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
        api_key=LLM_CONFIG.get("api_key"),
        api_base=LLM_CONFIG.get("api_base"),
    )

    display_answer(result)

    if tracer:
        tracer.add_span("LLM 生成",
                        {"model": LLM_CONFIG["model"], "query": query},
                        {"answer": result["answer"][:200]},
                        obs_type="generation")

    return {
        "query": query,
        "answer": result["answer"],
        "contexts": reranked,
        "model": result["model"],
        "usage": result.get("usage", {}),
    }


# ============================================================
# 评估阶段
# ============================================================
def run_evaluation(pipeline_fn):
    """
    运行 RAGAS 评估

    使用黄金集（人工标注的测试数据）评估系统质量
    """
    print_step_header(8, "评估 (RAGAS + 黄金集)")
    console.print("[dim]使用黄金集测试 RAG 系统的 4 个核心指标:[/dim]")
    console.print("[dim]  Faithfulness    - 答案是否忠实于上下文[/dim]")
    console.print("[dim]  Answer Relevancy - 答案是否回答了问题[/dim]")
    console.print("[dim]  Context Precision - 检索的上下文是否精准[/dim]")
    console.print("[dim]  Context Recall   - 是否检索到所有需要的信息[/dim]")
    console.print()

    from modules.evaluator import create_golden_set, run_evaluation_pipeline

    golden_set = create_golden_set()
    console.print(f"  黄金集: {len(golden_set)} 个标注问题")

    scores = run_evaluation_pipeline(golden_set, pipeline_fn)
    return scores


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="RAG 完整链路 Demo")
    parser.add_argument("--query", type=str, default="什么是 RAG？它有哪些优势？",
                        help="查询问题")
    parser.add_argument("--skip-eval", action="store_true",
                        help="跳过评估阶段")
    parser.add_argument("--skip-trace", action="store_true",
                        help="跳过追踪")
    args = parser.parse_args()

    print_banner()

    start_time = time.time()

    # ── 阶段一: 构建索引 ──
    documents, chunks, vector_store = build_index()

    # ── 初始化追踪器 ──
    from modules.tracer import create_tracer
    tracer = create_tracer(
        enabled=LANGFUSE_CONFIG["enabled"],
        public_key=LANGFUSE_CONFIG["public_key"],
        secret_key=LANGFUSE_CONFIG["secret_key"],
        base_url=LANGFUSE_CONFIG["base_url"],
    )

    # ── 阶段二: 查询 ──
    result = query_pipeline(args.query, chunks, vector_store, tracer)

    # ── ⑧ 评估 ──
    if not args.skip_eval:
        def eval_pipeline_fn(q):
            return query_pipeline(q, chunks, vector_store)

        scores = run_evaluation(eval_pipeline_fn)
    else:
        console.print("\n[dim]跳过评估阶段 (--skip-eval)[/dim]")

    # ── 结束追踪 ──
    if tracer:
        trace_result = tracer.end_trace({"answer": result["answer"]})

    # ── 总结 ──
    total_time = time.time() - start_time
    console.print()
    console.print(Panel.fit(
        f"[bold green]🎉 RAG 完整链路演示完成![/bold green]\n\n"
        f"总耗时: {total_time:.2f}s\n"
        f"查询: {args.query}\n"
        f"文档数: {len(documents)}\n"
        f"文本块数: {len(chunks)}\n"
        f"模型: {result['model']}",
        border_style="green",
    ))

    console.print()
    console.print("[bold]链路总结:[/bold]")
    console.print("  ① 文档解析 → ② 文本分块 → ③ 向量嵌入 → ④ 向量存储")
    console.print("  → ⑤ 混合检索 → ⑥ 重排序 → ⑦ LLM 生成")
    console.print("  → ⑧ RAGAS 评估 → ⑨ Langfuse 追踪")
    console.print()
    console.print("[dim]提示: 修改 config.py 调整参数，替换 data/sample_docs/ 中的文档测试效果[/dim]")


if __name__ == "__main__":
    main()
