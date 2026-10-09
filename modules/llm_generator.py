"""
⑦ LLM 生成模块 (Qwen/GPT via LiteLLM)
========================================
基于检索到的上下文，使用 LLM 生成最终答案。

LiteLLM 是什么？
----------------
一个统一的 LLM API 网关库，让你用同一套代码调用:
- OpenAI (GPT-4o, GPT-4o-mini)
- 阿里通义 (Qwen-Plus, Qwen-Max)
- Anthropic (Claude 3.5)
- 本地模型 (Ollama, vLLM)
- 等等 100+ 模型

好处:
- 一套代码切换多个模型
- 自动重试、超时处理
- 负载均衡（多 API key 轮询）
- 统一的 token 计数和成本计算

RAG Prompt 模板:
----------------
典型的 RAG prompt 结构:

    基于以下已知信息回答用户的问题。
    如果已知信息中没有相关内容，请说明你不知道。

    已知信息:
    {context}

    用户问题: {query}

    回答:
"""

from typing import List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


# ============================================================
# Prompt 模板
# ============================================================
RAG_PROMPT_TEMPLATE = """\
你是一个专业的知识助手。请基于以下已知信息回答用户的问题。

规则:
1. 只基于提供的已知信息回答，不要编造信息
2. 如果已知信息中没有相关内容，请明确说明"根据已有信息无法回答"
3. 回答时引用信息来源（文件名）
4. 保持回答简洁准确

---
已知信息:
{context}
---

用户问题: {query}

请回答:"""


def build_context(retrieved_docs: List[Dict]) -> str:
    """
    将检索到的文档拼接为上下文字符串

    Args:
        retrieved_docs: 重排序后的文档列表

    Returns:
        格式化的上下文文本
    """
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.get("metadata", {}).get("source", "未知")
        score = doc.get("rerank_score", 0)
        content = doc["content"].strip()

        context_parts.append(
            f"[信息{i}] 来源: {source} (相关度: {score:.2f})\n"
            f"{content}\n"
        )

    return "\n".join(context_parts)


def generate_answer(
    query: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4o-mini",
    provider: str = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    api_key: str = None,
    api_base: str = None,
) -> Dict:
    """
    使用 LLM 生成答案

    Args:
        query: 用户问题
        retrieved_docs: 重排序后的检索结果
        model: 模型名称
        provider: 提供商 (dashscope / openai / None)
        temperature: 温度参数（越低越确定）
        max_tokens: 最大输出 token 数
        api_key: API 密钥
        api_base: 自定义 API 地址

    Returns:
        {"answer": "...", "prompt": "...", "model": "...", "usage": {...}}
    """
    console.print(f"\n[bold blue]═══ ⑦ LLM 生成 ═══[/bold blue]")

    # 1. 构建上下文
    context = build_context(retrieved_docs)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)

    console.print(f"  [cyan]模型:[/cyan] {model}")
    console.print(f"  [cyan]提供商:[/cyan] {provider or 'auto'}")
    console.print(f"  [cyan]温度:[/cyan] {temperature}")
    console.print(f"  [cyan]上下文长度:[/cyan] {len(context)} 字符")
    console.print(f"  [cyan]Prompt 长度:[/cyan] {len(prompt)} 字符")

    # 2. 调用 LLM（通过 LiteLLM）
    try:
        # 在导入 LiteLLM 之前设置环境变量，跳过 GitHub 模型价格表下载
        import os
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "1")
        
        import litellm

        console.print(f"  [cyan]调用 LiteLLM...[/cyan]")

        # 构建模型标识符 (provider/model 格式)
        if provider:
            litellm_model = f"{provider}/{model}"
        else:
            litellm_model = model

        response = litellm.completion(
            model=litellm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base,
        )

        answer = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        console.print(f"  [green]✅ 生成完成[/green]")
        console.print(f"  Token 使用: prompt={usage['prompt_tokens']}, "
                       f"completion={usage['completion_tokens']}, "
                       f"total={usage['total_tokens']}")

        return {
            "answer": answer,
            "prompt": prompt,
            "model": model,
            "usage": usage,
        }

    except ImportError:
        console.print("  [yellow]⚠️  litellm 未安装，使用模拟回答[/yellow]")
        return _mock_generate(query, retrieved_docs, prompt, model)
    except Exception as e:
        console.print(f"  [red]❌ LLM 调用失败:[/red] {e}")
        return _mock_generate(query, retrieved_docs, prompt, model)


def _mock_generate(query: str, retrieved_docs: List[Dict], prompt: str, model: str) -> Dict:
    """模拟生成（无需 API key）"""
    # 基于检索内容生成简单回答
    context_summary = "\n".join([
        f"- {doc['content'][:100]}..."
        for doc in retrieved_docs[:3]
    ])

    answer = (
        f"根据检索到的信息，关于「{query}」的相关内容如下:\n\n"
        f"{context_summary}\n\n"
        f"[这是模拟回答 - 安装 litellm 并配置 API key 后将使用真实 LLM 生成]"
    )

    return {
        "answer": answer,
        "prompt": prompt,
        "model": model,
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(answer), "total_tokens": len(prompt) + len(answer)},
    }


def display_answer(result: Dict):
    """美化展示答案"""
    console.print()
    console.print(Panel(
        Markdown(result["answer"]),
        title=f"[bold green]LLM 回答[/bold green] (模型: {result['model']})",
        border_style="green",
    ))
    console.print()


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    query = "RAG 系统有哪些优势？"
    docs = [
        {
            "content": "RAG 的优势包括：减少幻觉、知识可更新、可追溯来源。",
            "rerank_score": 0.95,
            "metadata": {"source": "rag_intro.txt"},
        },
        {
            "content": "RAG 通过检索真实文档作为上下文，让 LLM 基于事实生成回答。",
            "rerank_score": 0.88,
            "metadata": {"source": "rag_intro.txt"},
        },
    ]

    result = generate_answer(query, docs)
    display_answer(result)
