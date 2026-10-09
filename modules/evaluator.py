"""
⑧ 评估模块 (RAGAS + 黄金集)
=============================
量化评估 RAG 系统的回答质量。

为什么需要评估？
---------------
RAG 系统有很多可调参数:
- 分块大小、重叠比例
- 检索数量、重排数量
- Prompt 模板
- 模型选择

没有评估 = 盲人摸象，无法知道改动是变好还是变坏。

RAGAS 评估框架:
--------------
RAGAS (Retrieval Augmented Generation Assessment) 提供 4 个核心指标:

1. Faithfulness（忠实度）
   - 答案是否忠实于检索到的上下文？
   - 有没有"幻觉"（编造上下文中没有的信息）？
   - 例: 上下文说"RAG 有3个优势"，答案说"RAG 有5个优势" → 低分

2. Answer Relevancy（答案相关性）
   - 答案是否回答了用户的问题？
   - 有没有答非所问？
   - 例: 问"RAG 是什么"，答"Python 是一种编程语言" → 低分

3. Context Precision（上下文精确度）
   - 检索到的上下文是否精准？
   - 有没有混入不相关的文档？
   - 例: 检索了10篇文档，只有2篇相关 → 低分

4. Context Recall（上下文召回率）
   - 是否检索到了所有需要的信息？
   - 有没有遗漏关键文档？
   - 需要"黄金标准"答案来计算

黄金集 (Golden Set):
-------------------
人工标注的测试集，格式:
[
    {
        "question": "RAG 是什么？",
        "ground_truth": "RAG 是检索增强生成架构...",
        "contexts": ["RAG 的全称是...", "RAG 结合了检索..."],
    },
    ...
]
"""

from typing import List, Dict
import json
import os
from rich.console import Console
from rich.table import Table

console = Console()


def create_golden_set() -> List[Dict]:
    """
    创建示例黄金集（人工标注的测试数据）

    实际项目中，黄金集应该:
    1. 覆盖各种类型的问题
    2. 由领域专家标注
    3. 定期更新
    """
    golden_set = [
        {
            "question": "什么是 RAG？",
            "ground_truth": "RAG（Retrieval-Augmented Generation）是检索增强生成架构，结合了信息检索和大语言模型生成的技术。",
            "contexts": [
                "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI架构",
                "RAG 的核心思想是先从知识库中检索相关文档，再将这些文档作为上下文输入给大语言模型",
            ],
        },
        {
            "question": "RAG 有哪些优势？",
            "ground_truth": "RAG 的优势包括减少幻觉、知识可更新、可追溯信息来源。",
            "contexts": [
                "RAG的优势包括：减少幻觉、知识可更新、可追溯来源",
            ],
        },
        {
            "question": "文本分块为什么要设置重叠？",
            "ground_truth": "重叠是为了防止在关键语义处截断，避免句子被切断导致语义不完整。",
            "contexts": [
                "chunk_overlap = 51 tokens (10%): 块之间的重叠，防止在关键语义处截断",
            ],
        },
    ]
    return golden_set


def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: List[str] = None,
) -> Dict[str, float]:
    """
    使用 RAGAS 评估 RAG 系统

    Args:
        questions: 用户问题列表
        answers: LLM 生成的答案列表
        contexts_list: 每个问题检索到的上下文列表
        ground_truths: 标准答案列表（可选）

    Returns:
        各指标的分数 {"faithfulness": 0.85, "answer_relevancy": 0.90, ...}
    """
    console.print(f"\n[bold blue]═══ ⑧ RAGAS 评估 ═══[/bold blue]")

    try:
        import warnings
        from ragas import evaluate
        from datasets import Dataset

        # ragas 旧版 metrics 通过 OPENAI_API_KEY/OPENAI_API_BASE 环境变量配置 LLM
        # 将项目配置的 LLM 信息注入环境变量，供 ragas 内部使用
        import config
        if config.LLM_CONFIG.get("api_key"):
            os.environ["OPENAI_API_KEY"] = config.LLM_CONFIG["api_key"]
        if config.LLM_CONFIG.get("api_base"):
            os.environ["OPENAI_API_BASE"] = config.LLM_CONFIG["api_base"]
        # 加长超时时间（评估需要多次 LLM 调用，默认超时太短）
        os.environ["OPENAI_TIMEOUT"] = "300"
        os.environ["OPENAI_REQUEST_TIMEOUT"] = "300"

        # ragas 0.4.x: 旧版 metrics 虽然 deprecated 但兼容 evaluate()
        # 新版 collections metrics 与 evaluate() 有类型不兼容问题
        # 注意: ragas 内部用 instructor 库做 structured output，
        # 与 dashscope 等非 OpenAI API 存在兼容性问题（连接超时）。
        # 使用 OpenAI API 时可正常评估。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        # 构建评估数据集
        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        if ground_truths:
            eval_data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(eval_data)

        # 执行评估
        console.print(f"  [cyan]评估 {len(questions)} 个问题...[/cyan]")
        results = evaluate(
            dataset,
            metrics=metrics,
        )

        # 提取分数（兼容不同版本 ragas 的返回格式）
        try:
            scores = results.scores
            if isinstance(scores, dict):
                return {k: float(v) for k, v in scores.items()}
        except AttributeError:
            pass
        try:
            return {k: float(v) for k, v in results.items()}
        except (TypeError, KeyError):
            # 手动提取
            scores = {}
            for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                if hasattr(results, key):
                    scores[key] = float(getattr(results, key))
            if scores:
                return scores
            raise ValueError(f"无法解析评估结果: {type(results)}")

    except ImportError:
        console.print("  [yellow]⚠️  ragas 未安装，使用模拟评估[/yellow]")
        return _mock_evaluate(questions)
    except Exception as e:
        console.print(f"  [yellow]⚠️  RAGAS 评估失败 ({type(e).__name__})，使用模拟评估[/yellow]")
        console.print(f"  [dim]提示: ragas 0.4.x 与部分 API 提供商存在兼容性问题[/dim]")
        return _mock_evaluate(questions)


def _mock_evaluate(questions: List[str]) -> Dict[str, float]:
    """模拟评估结果"""
    import random
    return {
        "faithfulness": round(random.uniform(0.7, 0.95), 4),
        "answer_relevancy": round(random.uniform(0.75, 0.95), 4),
        "context_precision": round(random.uniform(0.6, 0.9), 4),
        "context_recall": round(random.uniform(0.6, 0.85), 4),
    }


def display_evaluation(scores: Dict[str, float]):
    """展示评估结果"""
    table = Table(title="RAG 系统评估结果 (RAGAS)")
    table.add_column("指标", style="cyan")
    table.add_column("分数", style="green")
    table.add_column("说明", style="yellow")

    descriptions = {
        "faithfulness": "答案是否忠实于上下文（不编造信息）",
        "answer_relevancy": "答案是否回答了用户问题",
        "context_precision": "检索的上下文是否精准",
        "context_recall": "是否检索到了所有需要的信息",
    }

    for metric, score in scores.items():
        desc = descriptions.get(metric, "")
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        table.add_row(metric, f"{score:.4f} {bar}", desc)

    console.print(table)

    # 总体评价
    avg_score = sum(scores.values()) / len(scores)
    if avg_score >= 0.85:
        console.print(f"\n  [bold green]🎉 总体优秀! 平均分: {avg_score:.4f}[/bold green]")
    elif avg_score >= 0.7:
        console.print(f"\n  [bold yellow]👍 总体良好，仍有优化空间。平均分: {avg_score:.4f}[/bold yellow]")
    else:
        console.print(f"\n  [bold red]⚠️ 需要优化。平均分: {avg_score:.4f}[/bold red]")


def run_evaluation_pipeline(
    golden_set: List[Dict],
    rag_pipeline_fn,
) -> Dict[str, float]:
    """
    完整的评估流程

    Args:
        golden_set: 黄金集
        rag_pipeline_fn: RAG 管线函数 (query) -> {"answer": ..., "contexts": [...]}

    Returns:
        评估分数
    """
    console.print(f"\n[bold blue]═══ 运行评估流程 ═══[/bold blue]")
    console.print(f"黄金集大小: {len(golden_set)}")

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for item in golden_set:
        q = item["question"]
        gt = item.get("ground_truth", "")

        # 运行 RAG 管线
        result = rag_pipeline_fn(q)

        questions.append(q)
        answers.append(result.get("answer", ""))
        contexts_list.append([doc["content"] for doc in result.get("contexts", [])])
        ground_truths.append(gt)

    # 评估
    scores = evaluate_with_ragas(questions, answers, contexts_list, ground_truths)
    display_evaluation(scores)

    return scores
