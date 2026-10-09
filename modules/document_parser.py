"""
① 文档解析模块
===============
使用 unstructured 库将各种格式的文档转换为纯文本。

为什么需要文档解析？
-------------------
- 企业文档通常是 PDF、Word、PPT、HTML 等格式
- 这些文件包含排版、图片、表格等非结构化信息
- LLM 只能理解纯文本，所以需要先"提取"和"清洗"

MinerU vs unstructured:
-----------------------
- MinerU: 专注 PDF，能保留标题层级、表格结构，适合学术论文
- unstructured: 通用型，支持 20+ 格式，开箱即用，本 demo 选用

实际生产中的考虑:
-----------------
- OCR: 扫描件 PDF 需要 OCR 识别文字
- 表格: 需要保留表格结构（转为 Markdown 或 HTML）
- 图片: 多模态模型可以处理图片，否则需要图片描述
"""

import os
from typing import List, Dict
from rich.console import Console

console = Console()


def parse_with_unstructured(file_path: str) -> Dict[str, str]:
    """
    使用 unstructured 解析单个文件

    Args:
        file_path: 文件路径（支持 .pdf, .docx, .html, .txt 等）

    Returns:
        {"filename": "xxx.pdf", "content": "提取的纯文本", "metadata": {...}}
    """
    console.print(f"  [cyan]📄 解析文件:[/cyan] {file_path}")

    # ----------------------------------------------------------
    # unstructured 的核心思路:
    # 1. 按元素(element)解析文档: 标题、段落、表格、图片等
    # 2. 过滤掉不需要的元素（如图片）
    # 3. 拼接为纯文本
    # ----------------------------------------------------------
    try:
        from unstructured.partition.auto import partition

        # partition() 自动检测文件类型并解析
        elements = partition(filename=file_path)

        # 提取文本内容
        content = "\n\n".join([str(el) for el in elements])

        # 元数据
        metadata = {
            "filename": os.path.basename(file_path),
            "filetype": os.path.splitext(file_path)[1],
            "element_count": len(elements),
        }

        console.print(f"  [green]✅ 解析成功[/green] - 提取 {len(content)} 字符, {len(elements)} 个元素")

        return {
            "filename": os.path.basename(file_path),
            "content": content,
            "metadata": metadata,
        }

    except ImportError:
        console.print("  [yellow]⚠️  unstructured 未安装，使用简单文本读取[/yellow]")
        return _simple_parse(file_path)
    except Exception as e:
        console.print(f"  [red]❌ 解析失败:[/red] {e}")
        return _simple_parse(file_path)


def _simple_parse(file_path: str) -> Dict[str, str]:
    """降级方案: 直接读取文本文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "filename": os.path.basename(file_path),
        "content": content,
        "metadata": {"filename": os.path.basename(file_path), "simple_parse": True},
    }


def parse_documents(directory: str) -> List[Dict[str, str]]:
    """
    批量解析目录下所有文档

    Args:
        directory: 文档目录

    Returns:
        解析结果列表 [{"filename": ..., "content": ..., "metadata": ...}, ...]
    """
    console.print(f"\n[bold blue]═══ ① 文档解析 ═══[/bold blue]")
    console.print(f"扫描目录: {directory}")

    results = []
    supported_extensions = {".txt", ".pdf", ".docx", ".html", ".md", ".csv"}

    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext in supported_extensions:
            result = parse_with_unstructured(file_path)
            results.append(result)

    console.print(f"\n[bold green]共解析 {len(results)} 个文档[/bold green]")
    return results


# ============================================================
# 演示用: 如果直接运行此文件
# ============================================================
if __name__ == "__main__":
    docs = parse_documents("data/sample_docs")
    for doc in docs:
        print(f"\n{'='*60}")
        print(f"文件: {doc['filename']}")
        print(f"内容预览: {doc['content'][:200]}...")
        print(f"{'='*60}")
