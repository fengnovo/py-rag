"""
② 文本分块模块
===============
将长文档切分成适合嵌入的小块（chunk）。

为什么要分块？
-------------
1. 嵌入模型有 token 上限（bge-m3 最大 8192 tokens）
2. 检索精度: 小块 → 更精确的语义匹配
3. LLM 上下文窗口有限，不能塞入整篇文档

递归分块 (Recursive Chunking) 原理:
-----------------------------------
按优先级依次尝试以下分隔符:
  "\n\n" (段落) → "\n" (行) → "。" (句子) → " " (词)

流程:
1. 先用 "\n\n" 切分
2. 如果某块仍然 > chunk_size，再用 "\n" 切分该块
3. 如果还太大，用 "。" 切分
4. 递归直到所有块都 <= chunk_size

参数解释:
- chunk_size = 512 tokens: 每块的目标大小
  - 大约 400-500 个汉字（中文 1 token ≈ 1.5 字）
  - 大约 700-800 个英文单词（英文 1 token ≈ 0.75 词）
- chunk_overlap = 51 tokens (10%): 块之间的重叠
  - 防止在关键语义处截断
  - 例如: "RAG系统由检索和生成两部分组成" 不会被切成
    "RAG系统由检索和" + "生成两部分组成"
"""

from typing import List, Dict
from rich.console import Console
from rich.table import Table

console = Console()


def count_tokens(text: str) -> int:
    """
    计算文本的 token 数

    使用 tiktoken (OpenAI 的 tokenizer)
    也可以用字符数近似: 中文 ≈ 字符数 * 0.6, 英文 ≈ 词数 * 1.3
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # 降级: 粗略估算（中文约 1.5 字符 = 1 token）
        return int(len(text) * 0.7)


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 51,
    separators: List[str] = None,
) -> List[str]:
    """
    递归分块器核心实现

    Args:
        text: 待分块的文本
        chunk_size: 每块目标 token 数
        chunk_overlap: 重叠 token 数
        separators: 分隔符优先级列表

    Returns:
        文本块列表
    """
    if separators is None:
        # 分隔符优先级: 段落 > 行 > 句子 > 词
        separators = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]

    chunks = []

    def _split(text: str, sep_idx: int = 0) -> List[str]:
        """递归切分"""
        if sep_idx >= len(separators):
            return [text] if text.strip() else []

        separator = separators[sep_idx]

        if separator == "":
            # 最后的兜底: 按字符硬切
            return _hard_split(text, chunk_size)

        # 用当前分隔符切分
        parts = text.split(separator)

        result = []
        current_chunk = ""

        for part in parts:
            # 尝试把当前 part 加入 current_chunk
            test_chunk = current_chunk + separator + part if current_chunk else part
            test_tokens = count_tokens(test_chunk)

            if test_tokens <= chunk_size:
                current_chunk = test_chunk
            else:
                # current_chunk 已经够大了，保存它
                if current_chunk:
                    result.append(current_chunk.strip())

                # 如果单个 part 就超过 chunk_size，用下一级分隔符继续切
                if count_tokens(part) > chunk_size:
                    sub_chunks = _split(part, sep_idx + 1)
                    result.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk.strip():
            result.append(current_chunk.strip())

        return result

    # 第一步: 递归切分
    raw_chunks = _split(text)

    # 第二步: 添加重叠
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        chunks = _add_overlap(raw_chunks, chunk_overlap)
    else:
        chunks = raw_chunks

    return chunks


def _hard_split(text: str, max_tokens: int) -> List[str]:
    """按 token 数硬切分（兜底方案）"""
    chunks = []
    words = list(text)
    current = ""
    for char in words:
        current += char
        if count_tokens(current) >= max_tokens:
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _add_overlap(chunks: List[str], overlap_tokens: int) -> List[str]:
    """
    在相邻块之间添加重叠

    例如:
    块1: "A B C D E"
    块2: "D E F G H"  (D E 是重叠部分)
    块3: "G H I J K"  (G H 是重叠部分)
    """
    result = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        curr_chunk = chunks[i]

        # 取前一个块的尾部作为重叠
        words = prev_chunk.split()
        overlap_text = " ".join(words[-overlap_tokens:]) if len(words) > overlap_tokens else prev_chunk

        # 将重叠部分加到当前块头部
        new_chunk = overlap_text + " " + curr_chunk
        result.append(new_chunk.strip())

    return result


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 51,
) -> List[Dict]:
    """
    对多个文档进行分块

    Args:
        documents: 文档列表 [{"filename": ..., "content": ...}, ...]
        chunk_size: 块大小 (tokens)
        chunk_overlap: 重叠大小 (tokens)

    Returns:
        分块结果列表，每块带有元数据
    """
    console.print(f"\n[bold blue]═══ ② 文本分块 ═══[/bold blue]")
    console.print(f"参数: chunk_size={chunk_size} tokens, overlap={chunk_overlap} tokens (10%)")

    all_chunks = []

    for doc in documents:
        content = doc["content"]
        filename = doc["filename"]

        # 分块
        chunks = recursive_chunk(content, chunk_size, chunk_overlap)

        for idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id": f"{filename}_chunk_{idx}",
                "content": chunk_text,
                "metadata": {
                    "source": filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "token_count": count_tokens(chunk_text),
                },
            })

        console.print(
            f"  [cyan]{filename}[/cyan] → "
            f"[green]{len(chunks)}[/green] 块"
        )

    # 统计表格
    table = Table(title="分块统计")
    table.add_column("文档", style="cyan")
    table.add_column("块数", style="green")
    table.add_column("平均tokens/块", style="yellow")
    table.add_column("总tokens", style="magenta")

    for doc in documents:
        doc_chunks = [c for c in all_chunks if c["metadata"]["source"] == doc["filename"]]
        avg_tokens = sum(c["metadata"]["token_count"] for c in doc_chunks) / max(len(doc_chunks), 1)
        total_tokens = sum(c["metadata"]["token_count"] for c in doc_chunks)
        table.add_row(
            doc["filename"],
            str(len(doc_chunks)),
            f"{avg_tokens:.0f}",
            str(total_tokens),
        )

    console.print(table)
    console.print(f"\n[bold green]总计: {len(all_chunks)} 个文本块[/bold green]")

    return all_chunks


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    sample_text = """
RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI架构。

它的核心思想是：先从知识库中检索相关文档，再将这些文档作为上下文输入给大语言模型，让模型基于这些真实信息生成回答。

RAG的优势包括：减少幻觉、知识可更新、可追溯来源。

完整的RAG链路包含：文档解析 → 分块 → 嵌入 → 向量存储 → 检索 → 重排序 → 生成 → 评估。
"""
    chunks = recursive_chunk(sample_text, chunk_size=50, chunk_overlap=5)
    for i, chunk in enumerate(chunks):
        print(f"\n--- 块 {i+1} ---")
        print(chunk)
