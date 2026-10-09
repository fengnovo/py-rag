"""
⑨ 追踪模块 (Langfuse)
======================
监控和调试 RAG 系统的每次运行。

为什么需要追踪？
---------------
RAG 系统链路长、组件多，出了问题很难定位:
- 是检索没找对文档？
- 是重排序排错了？
- 是 LLM 幻觉了？
- 是某个文档质量差？

Langfuse 的作用:
- 记录每次查询的完整链路（trace）
- 可视化每个环节的输入/输出
- 统计延迟、成本、token 消耗
- 支持团队协作和标注

追踪内容:
---------
Trace (一次完整查询):
├── Span: 文档解析
├── Span: 文本分块
├── Span: 向量嵌入
├── Span: 混合检索
│   ├── Span: BM25 检索
│   └── Span: 向量检索
├── Span: 重排序
├── Span: LLM 生成
└── Span: 评估（可选）

如何使用 Langfuse:
-----------------
1. 注册 Langfuse Cloud (https://cloud.langfuse.com) 或自部署
2. 获取 API Key
3. 在代码中初始化 tracer
4. 在每个环节记录 span
"""

import time
import json
from contextlib import ExitStack
from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table

console = Console()


class RAGTracer:
    """
    RAG 链路追踪器

    支持 Langfuse v4（生产）和简单的本地日志（开发）
    """

    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        base_url: str = "https://cloud.langfuse.com",
        enabled: bool = False,
    ):
        self.enabled = enabled
        self.trace_id = None
        self.spans = []
        self.start_time = None
        self.root_span = None       # 根 span（对应一次完整查询）
        self.child_spans = []       # 子 span 列表（各环节）
        self._stack = None          # ExitStack 管理上下文管理器生命周期

        if enabled:
            try:
                from langfuse import get_client
                # 优先使用环境变量，显式参数作为覆盖
                import os
                if public_key:
                    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
                if secret_key:
                    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
                if base_url:
                    os.environ.setdefault("LANGFUSE_BASE_URL", base_url)

                self.langfuse = get_client()
                console.print(f"  [green]✅ Langfuse 追踪已启用[/green]")
            except Exception as e:
                console.print(f"  [yellow]⚠️  Langfuse 初始化失败 ({e.__class__.__name__}: {e})，降级为本地模式[/yellow]")
                self.langfuse = None
                self.enabled = False
        else:
            self.langfuse = None
            console.print("  [dim]追踪器: 本地模式（设置 LANGFUSE_ENABLED=true 启用 Langfuse）[/dim]")

    def start_trace(self, query: str, metadata: Dict = None):
        """开始一次查询的追踪（创建根 span + trace）"""
        self.start_time = time.time()
        self.trace_id = f"trace_{int(time.time() * 1000)}"
        self.spans = []
        self.child_spans = []

        if self.enabled and self.langfuse:
            try:
                # 使用 ExitStack 管理上下文管理器生命周期
                self._stack = ExitStack()
                # v4 推荐: start_as_current_observation 上下文管理器自动创建 trace
                root_cm = self.langfuse.start_as_current_observation(
                    name="rag_query",
                    as_type="span",
                    input={"query": query},
                    metadata=metadata or {},
                )
                self.root_span = self._stack.enter_context(root_cm)
            except Exception as e:
                console.print(f"  [yellow]⚠️  Langfuse 创建 trace 失败: {e}[/yellow]")
                self.root_span = None

        console.print(f"\n[bold blue]═══ ⑨ 追踪开始 ═══[/bold blue]")
        console.print(f"  Trace ID: [cyan]{self.trace_id}[/cyan]")
        console.print(f"  查询: [yellow]{query}[/yellow]")

    def add_span(self, name: str, input_data: Dict = None, output_data: Dict = None,
                 metadata: Dict = None, obs_type: str = "span"):
        """记录一个环节（创建子 span）

        Args:
            name: 环节名称
            input_data: 输入数据
            output_data: 输出数据
            metadata: 元数据
            obs_type: 观察类型 - "span" / "generation" / "retriever" / "embedding"
        """
        span = {
            "name": name,
            "timestamp": time.time(),
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {},
        }
        self.spans.append(span)

        if self.enabled and self.langfuse and self.root_span is not None and self._stack is not None:
            try:
                # v4 推荐: start_as_current_observation 上下文管理器，退出时自动 .end()
                child_cm = self.langfuse.start_as_current_observation(
                    name=name,
                    as_type=obs_type,
                    input=input_data,
                    output=output_data,
                    metadata=metadata or {},
                )
                child = self._stack.enter_context(child_cm)
                self.child_spans.append(child)
            except Exception as e:
                console.print(f"  [dim]️  Langfuse 记录 span '{name}' 失败: {e}[/dim]")

        elapsed = span["timestamp"] - (self.spans[-2]["timestamp"] if len(self.spans) > 1 else self.start_time)
        console.print(f"  [dim]  {name}[/dim] ({elapsed:.3f}s)")

    def end_trace(self, final_output: Dict = None):
        """结束追踪"""
        total_time = time.time() - self.start_time

        if self.enabled and self.langfuse and self.root_span is not None:
            try:
                # 更新根 span 的最终输出
                self.root_span.update(
                    output=final_output,
                    metadata={"total_time": total_time, "num_spans": len(self.spans)},
                )
                # ExitStack.close() 按 LIFO 顺序关闭所有上下文管理器：
                # 先结束所有子 span，最后结束根 span，确保数据完整上报
                if self._stack is not None:
                    self._stack.close()
                    self._stack = None
            except Exception as e:
                console.print(f"  [yellow]️  Langfuse 结束 trace 失败: {e}[/yellow]")
            finally:
                try:
                    self.langfuse.flush()
                except Exception as e:
                    console.print(f"  [dim]️  Langfuse flush 失败: {e.__class__.__name__}[/dim]")

        console.print(f"\n[bold blue]═══ ⑨ 追踪结束 ═══[/bold blue]")
        console.print(f"  总耗时: [green]{total_time:.3f}s[/green]")
        console.print(f"  环节数: [cyan]{len(self.spans)}[/cyan]")

        # 展示链路时间线
        self._display_timeline()

        return {
            "trace_id": self.trace_id,
            "total_time": total_time,
            "spans": self.spans,
        }

    def _display_timeline(self):
        """展示时间线"""
        table = Table(title="链路时间线")
        table.add_column("环节", style="cyan")
        table.add_column("耗时", style="green")
        table.add_column("累计", style="yellow")

        cumulative = 0
        for i, span in enumerate(self.spans):
            if i == 0:
                elapsed = span["timestamp"] - self.start_time
            else:
                elapsed = span["timestamp"] - self.spans[i - 1]["timestamp"]
            cumulative += elapsed

            table.add_row(
                span["name"],
                f"{elapsed:.3f}s",
                f"{cumulative:.3f}s",
            )

        console.print(table)


def create_tracer(enabled: bool = False, **kwargs) -> RAGTracer:
    """创建追踪器"""
    return RAGTracer(enabled=enabled, **kwargs)


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    tracer = RAGTracer(enabled=False)

    tracer.start_trace("什么是 RAG?")
    time.sleep(0.1)
    tracer.add_span("文档解析", {"files": 3}, {"documents": 3})
    time.sleep(0.05)
    tracer.add_span("文本分块", {"documents": 3}, {"chunks": 15})
    time.sleep(0.08)
    tracer.add_span("向量嵌入", {"chunks": 15}, {"vectors": 15})
    time.sleep(0.03)
    tracer.add_span("混合检索", {"query": "什么是 RAG?"}, {"results": 20})
    time.sleep(0.06)
    tracer.add_span("重排序", {"candidates": 20}, {"results": 5})
    time.sleep(0.2)
    tracer.add_span("LLM 生成", {"model": "gpt-4o-mini"}, {"answer": "RAG 是..."})

    tracer.end_trace({"answer": "RAG 是检索增强生成"})
