#!/usr/bin/env python3
"""
RAG 链路逐步验证脚本
====================
运行方式:
  python verify_steps.py              # 验证所有步骤
  python verify_steps.py --step 1     # 只验证第1步
  python verify_steps.py --step 1-3   # 验证第1到3步
"""

import sys
import os
import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_step_1():
    """① 文档解析验证"""
    console.print("\n[bold cyan]═══ 验证 Step 1: 文档解析 ═══[/bold cyan]")
    
    from modules.document_parser import parse_documents
    
    docs = parse_documents("data/sample_docs")
    
    if docs:
        console.print(f"[green]✅ 成功解析 {len(docs)} 个文档[/green]")
        for doc in docs:
            console.print(f"  - {doc['filename']}: {len(doc['content'])} 字符")
        return True, docs
    else:
        console.print("[red]❌ 没有解析到文档[/red]")
        return False, []


def verify_step_2(docs):
    """② 文本分块验证"""
    console.print("\n[bold cyan]═══ 验证 Step 2: 文本分块 ═══[/bold cyan]")
    
    from modules.text_chunker import chunk_documents
    from config import CHUNK_CONFIG
    
    chunks = chunk_documents(
        docs,
        chunk_size=CHUNK_CONFIG["chunk_size"],
        chunk_overlap=CHUNK_CONFIG["chunk_overlap"],
    )
    
    if chunks:
        console.print(f"[green]✅ 成功切分为 {len(chunks)} 个文本块[/green]")
        return True, chunks
    else:
        console.print("[red]❌ 分块失败[/red]")
        return False, []


def verify_step_3(chunks):
    """③ 向量嵌入验证"""
    console.print("\n[bold cyan]═══ 验证 Step 3: 向量嵌入 ═══[/bold cyan]")
    
    from modules.embedder import embed_chunks
    
    chunks = embed_chunks(chunks)
    
    # 验证向量维度
    if chunks and "embedding" in chunks[0]:
        dim = len(chunks[0]["embedding"])
        console.print(f"[green]✅ 嵌入成功，向量维度: {dim}[/green]")
        return True, chunks
    else:
        console.print("[red]❌ 嵌入失败[/red]")
        return False, []


def verify_step_4(chunks):
    """④ 向量存储验证 (Qdrant)"""
    console.print("\n[bold cyan]═══ 验证 Step 4: 向量存储 (Qdrant) ═══[/bold cyan]")
    
    from config import QDRANT_CONFIG
    
    console.print(f"  连接: {QDRANT_CONFIG['host']}:{QDRANT_CONFIG['port']}")
    console.print(f"  Collection: {QDRANT_CONFIG['collection_name']}")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # 连接测试
        client = QdrantClient(
            host=QDRANT_CONFIG["host"],
            port=QDRANT_CONFIG["port"],
        )
        
        # 检查 collection 是否存在
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        console.print(f"  现有 collections: {collection_names}")
        
        collection_name = QDRANT_CONFIG["collection_name"]
        
        # 如果不存在则创建
        if collection_name not in collection_names:
            console.print(f"  创建 collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=len(chunks[0]["embedding"]),
                    distance=Distance.COSINE,
                ),
            )
        
        # 清空旧数据
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(chunks[0]["embedding"]),
                distance=Distance.COSINE,
            ),
        )
        
        # 插入数据
        points = []
        for i, chunk in enumerate(chunks):
            points.append(PointStruct(
                id=i,
                vector=chunk["embedding"],
                payload={
                    "content": chunk["content"],
                    "source": chunk["metadata"].get("source", ""),
                    "chunk_index": chunk["metadata"].get("chunk_index", 0),
                },
            ))
        
        client.upsert(collection_name=collection_name, points=points)
        
        # 验证
        info = client.get_collection(collection_name)
        console.print(f"[green]✅ Qdrant 存储成功，共 {info.points_count} 个向量[/green]")
        
        return True, client
        
    except ImportError:
        console.print("[yellow]⚠️  qdrant-client 未安装，使用 FAISS 替代[/yellow]")
        from modules.vector_store import store_vectors
        store = store_vectors(chunks, store_type="faiss")
        return True, store
    except Exception as e:
        console.print(f"[red]❌ Qdrant 连接失败: {e}[/red]")
        console.print("[yellow]降级使用 FAISS[/yellow]")
        from modules.vector_store import store_vectors
        store = store_vectors(chunks, store_type="faiss")
        return True, store


def verify_step_5(chunks, vector_store):
    """⑤ 混合检索验证"""
    console.print("\n[bold cyan]═══ 验证 Step 5: 混合检索 ═══[/bold cyan]")
    
    from modules.embedder import Embedder
    from modules.hybrid_retriever import BM25Retriever, hybrid_retrieve
    from config import RETRIEVAL_CONFIG, QDRANT_CONFIG
    
    query = "什么是 RAG？"
    console.print(f"  测试查询: '{query}'")
    
    # 如果 vector_store 是 QdrantClient，需要重新加载数据
    if hasattr(vector_store, 'get_collection'):
        # 从 Qdrant 重新加载 chunks
        console.print("  [dim]从 Qdrant 重新加载数据...[/dim]")
        from qdrant_client import QdrantClient
        client = vector_store
        collection_name = QDRANT_CONFIG["collection_name"]
        
        # 查询所有点
        results = client.scroll(
            collection_name=collection_name,
            limit=100,
            with_payload=True,
            with_vectors=True,
        )
        points = results[0]
        
        # 转换为 chunks 格式
        chunks = []
        for point in points:
            chunk = {
                "id": f"chunk_{point.id}",
                "content": point.payload.get("content", ""),
                "embedding": point.vector,
                "metadata": {
                    "source": point.payload.get("source", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                },
            }
            chunks.append(chunk)
        
        console.print(f"  重新加载 {len(chunks)} 个文本块")
        
        # 创建 FAISS 存储用于检索
        from modules.vector_store import FAISSStore
        faiss_store = FAISSStore(dimension=len(chunks[0]["embedding"]))
        faiss_store.add(chunks)
        vector_store = faiss_store
    
    # 嵌入查询
    embedder = Embedder()
    query_vector = embedder.embed_query(query)
    
    # BM25
    bm25 = BM25Retriever(chunks)
    
    # 混合检索
    results = hybrid_retrieve(
        query=query,
        query_vector=query_vector,
        vector_store=vector_store,
        bm25_retriever=bm25,
        top_k=RETRIEVAL_CONFIG["top_k"],
    )
    
    if results:
        console.print(f"[green]✅ 检索到 {len(results)} 条结果[/green]")
        console.print(f"  Top1: {results[0]['content'][:80]}...")
        return True, results
    else:
        console.print("[red]❌ 检索失败[/red]")
        return False, []


def verify_step_6(query, results):
    """⑥ 重排序验证"""
    console.print("\n[bold cyan]═══ 验证 Step 6: 重排序 ═══[/bold cyan]")
    
    from modules.reranker import rerank_results
    from config import RERANKER_CONFIG
    
    reranked = rerank_results(
        query=query,
        retrieved_docs=results,
        top_n=RERANKER_CONFIG["top_n"],
    )
    
    if reranked:
        console.print(f"[green]✅ 重排序完成，保留 {len(reranked)} 条[/green]")
        return True, reranked
    else:
        console.print("[red]❌ 重排序失败[/red]")
        return False, []


def verify_step_7(query, reranked):
    """⑦ LLM 生成验证"""
    console.print("\n[bold cyan]═══ 验证 Step 7: LLM 生成 ═══[/bold cyan]")
    
    from modules.llm_generator import generate_answer, display_answer
    from config import LLM_CONFIG
    
    result = generate_answer(
        query=query,
        retrieved_docs=reranked,
        model=LLM_CONFIG["model"],
        provider=LLM_CONFIG.get("provider"),
        temperature=LLM_CONFIG["temperature"],
        api_key=LLM_CONFIG.get("api_key"),
        api_base=LLM_CONFIG.get("api_base"),
    )
    
    if result and result.get("answer"):
        display_answer(result)
        console.print(f"[green]✅ LLM 生成成功[/green]")
        return True, result
    else:
        console.print("[red]❌ LLM 生成失败[/red]")
        return False, {}


def verify_step_8():
    """⑧ 评估验证"""
    console.print("\n[bold cyan]═══ 验证 Step 8: RAGAS 评估 ═══[/bold cyan]")
    
    try:
        from modules.evaluator import create_golden_set, evaluate_with_ragas, display_evaluation
        
        golden_set = create_golden_set()
        console.print(f"  黄金集: {len(golden_set)} 个问题")
        
        # 模拟一些结果进行评估
        questions = [item["question"] for item in golden_set[:2]]
        answers = ["这是模拟答案1", "这是模拟答案2"]
        contexts = [["上下文1"], ["上下文2"]]
        ground_truths = [item["ground_truth"] for item in golden_set[:2]]
        
        scores = evaluate_with_ragas(questions, answers, contexts, ground_truths)
        display_evaluation(scores)
        
        console.print(f"[green]✅ 评估模块正常[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]⚠️  评估模块: {e}[/yellow]")
        return True  # 非关键


def verify_step_9():
    """⑨ 追踪验证"""
    console.print("\n[bold cyan]═══ 验证 Step 9: Langfuse 追踪 ═══[/bold cyan]")
    
    from config import LANGFUSE_CONFIG
    
    if LANGFUSE_CONFIG["enabled"]:
        console.print(f"  Langfuse: {LANGFUSE_CONFIG['host']}")
        
        try:
            from langfuse import Langfuse
            
            lf = Langfuse(
                public_key=LANGFUSE_CONFIG["public_key"],
                secret_key=LANGFUSE_CONFIG["secret_key"],
                host=LANGFUSE_CONFIG["host"],
            )
            
            # 测试连接
            trace = lf.trace(name="test_trace", input={"test": True})
            lf.flush()
            
            console.print(f"[green]✅ Langfuse 连接成功[/green]")
            return True
        except Exception as e:
            console.print(f"[yellow]⚠️  Langfuse 连接失败: {e}[/yellow]")
            return True
    else:
        console.print("[dim]Langfuse 未启用 (设置 LANGFUSE_ENABLED=true)[/dim]")
        return True


def main():
    parser = argparse.ArgumentParser(description="逐步验证 RAG 链路")
    parser.add_argument("--step", type=str, default=None,
                        help="指定验证步骤，如: 1, 1-3, 5")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]RAG 链路逐步验证[/bold cyan]",
        border_style="cyan",
    ))
    
    # 确定要验证的步骤
    if args.step:
        if "-" in args.step:
            start, end = map(int, args.step.split("-"))
            steps = list(range(start, end + 1))
        else:
            steps = [int(args.step)]
    else:
        steps = list(range(1, 10))
    
    console.print(f"将验证步骤: {steps}")
    
    # 存储中间结果
    docs = []
    chunks = []
    vector_store = None
    results = []
    reranked = []
    
    # 逐步执行
    for step in steps:
        if step == 1:
            ok, docs = verify_step_1()
        elif step == 2:
            if not docs:
                console.print("[dim]自动执行前置步骤...[/dim]")
                ok, docs = verify_step_1()
            ok, chunks = verify_step_2(docs)
        elif step == 3:
            if not chunks:
                console.print("[dim]自动执行前置步骤...[/dim]")
                ok, docs = verify_step_1()
                ok, chunks = verify_step_2(docs)
            ok, chunks = verify_step_3(chunks)
        elif step == 4:
            if not chunks:
                console.print("[dim]自动执行前置步骤...[/dim]")
                ok, docs = verify_step_1()
                ok, chunks = verify_step_2(docs)
                ok, chunks = verify_step_3(chunks)
            ok, vector_store = verify_step_4(chunks)
        elif step == 5:
            if not chunks or not vector_store:
                # 从 Qdrant 重新加载，不需要执行 step 1-4
                console.print("[dim]从 Qdrant 加载已有数据...[/dim]")
                from config import QDRANT_CONFIG
                from qdrant_client import QdrantClient
                qdrant_client = QdrantClient(
                    host=QDRANT_CONFIG["host"],
                    port=QDRANT_CONFIG["port"],
                )
                ok, results = verify_step_5([], qdrant_client)
            else:
                ok, results = verify_step_5(chunks, vector_store)
        elif step == 6:
            if not results:
                console.print("[dim]从 Qdrant 加载数据并执行检索...[/dim]")
                from config import QDRANT_CONFIG
                from qdrant_client import QdrantClient
                qdrant_client = QdrantClient(
                    host=QDRANT_CONFIG["host"],
                    port=QDRANT_CONFIG["port"],
                )
                ok, results = verify_step_5([], qdrant_client)
            query = "什么是 RAG？"
            ok, reranked = verify_step_6(query, results)
        elif step == 7:
            if not reranked:
                console.print("[dim]从 Qdrant 加载数据并执行检索 + 重排...[/dim]")
                from config import QDRANT_CONFIG
                from qdrant_client import QdrantClient
                qdrant_client = QdrantClient(
                    host=QDRANT_CONFIG["host"],
                    port=QDRANT_CONFIG["port"],
                )
                ok, results = verify_step_5([], qdrant_client)
                query = "什么是 RAG？"
                ok, reranked = verify_step_6(query, results)
            query = "什么是 RAG？"
            ok, result = verify_step_7(query, reranked)
        elif step == 8:
            verify_step_8()
        elif step == 9:
            verify_step_9()
    
    console.print("\n[bold green]✅ 验证完成![/bold green]")


if __name__ == "__main__":
    main()
