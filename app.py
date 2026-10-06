"""文献智能问答系统 —— 命令行入口。

用法::

    python app.py                      # 交互式问答
    python app.py --sync                # 同步索引后退出（不提问）
    python app.py --rebuild             # 全量重建索引后进入问答
    python app.py --status              # 查看索引状态
    python app.py -q "论文的创新点是什么"  # 单次提问，适合脚本调用
    python app.py -q "..." --stream      # 流式输出（打字机效果）
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import get_settings
from src.logger import get_logger
from src.qa import Answer, PaperRAG

logger = get_logger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="基于 RAG 的学术文献智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python app.py                          # 交互模式\n"
            "  python app.py --status                 # 查看索引状态\n"
            "  python app.py --rebuild                # 重建索引\n"
            "  python app.py -q '论文的创新点是什么'     # 单次提问\n"
            "  python app.py -q '用了什么指标' --stream  # 流式输出\n"
        ),
    )
    parser.add_argument("-q", "--question", help="单次提问，不进入交互循环")
    parser.add_argument("--rebuild", action="store_true", help="全量重建向量索引")
    parser.add_argument("--sync", action="store_true", help="只同步索引后退出")
    parser.add_argument("--status", action="store_true", help="查看索引状态后退出")
    parser.add_argument("--stream", action="store_true", help="流式输出答案")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖检索文档块数量")
    return parser.parse_args()


def print_banner(settings) -> None:
    """打印启动信息与当前配置。"""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="cyan")
    for key, value in settings.summary().items():
        style = "red" if str(value).startswith("!!") else "cyan"
        table.add_row(key, f"[{style}]{value}[/{style}]")
    console.print(
        Panel(
            table,
            title="[bold]文献 RAG 问答系统[/bold]",
            subtitle="DeepSeek + Chroma + BGE",
            border_style="blue",
        )
    )


def print_index_status(rag: PaperRAG) -> None:
    """打印向量库覆盖情况（诊断"文献没被索引"这类问题）。"""
    info = rag.manager.collection_info()
    console.print(f"\n[bold]向量库状态[/bold]  集合: {info['集合名']}")
    console.print(f"  文本块数: [green]{info['文本块数']}[/green]")
    console.print(f"  覆盖文件: [green]{info['覆盖文件数']}[/green]")
    if info["文件列表"]:
        for name in info["文件列表"]:
            console.print(f"    · {name}")


def print_answer(answer: Answer) -> None:
    """以可读格式打印答案与出处。"""
    console.print(
        Panel(
            answer.text,
            title=f"[bold green]回答[/bold green] [dim]({answer.latency_s}s，"
            f"检索 {answer.retrieved} 块)[/dim]",
            border_style="green",
        )
    )
    if answer.sources:
        table = Table(title="参考来源", show_lines=False, border_style="dim")
        table.add_column("#", style="dim", width=3)
        table.add_column("文献", style="cyan", max_width=42)
        table.add_column("页", justify="right", width=5)
        table.add_column("摘要", style="dim", max_width=60)
        for src in answer.sources:
            table.add_row(str(src.index), src.file, str(src.page), src.snippet + "...")
        console.print(table)


def stream_answer(rag: PaperRAG, question: str, top_k: int | None) -> None:
    """流式打印答案（打字机效果）。"""
    console.print("\n[bold green]回答[/bold green] [dim](流式)[/dim]")
    try:
        for piece in rag.ask_stream(question, top_k=top_k):
            console.print(piece, end="", highlight=False)
        console.print()
    except RuntimeError as exc:
        console.print(f"\n[red]{exc}[/red]")


def main() -> int:
    """程序主流程。

    Returns:
        进程退出码：0 成功，1 配置或运行错误，130 用户中断。
    """
    args = parse_args()

    try:
        settings = get_settings()
    except ValueError as exc:
        console.print(f"[red]配置错误：{exc}[/red]")
        return 1

    print_banner(settings)

    try:
        rag = PaperRAG(settings)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]初始化失败：{exc}[/red]")
        return 1

    # ---- 仅查看状态 ----
    if args.status:
        print_index_status(rag)
        return 0

    # ---- 同步索引 ----
    try:
        with console.status("[cyan]正在同步向量索引..."):
            stats = rag.ensure_index(rebuild=args.rebuild)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]索引构建失败：{exc}[/red]")
        return 1

    if stats.is_up_to_date:
        console.print(
            f"\n[green]索引已是最新[/green]：{stats.indexed_files} 个文献、"
            f"{stats.total_chunks} 个文本块"
        )
    else:
        console.print(
            f"\n[green]索引同步完成[/green]：新增 {stats.added_chunks} 块，"
            f"现有 {stats.indexed_files} 个文献、{stats.total_chunks} 个文本块"
        )
        if stats.removed_files:
            console.print(f"  已清理 {len(stats.removed_files)} 个失效文献的向量")

    if args.sync:
        return 0

    # ---- 单次提问 ----
    if args.question:
        if args.stream:
            stream_answer(rag, args.question, args.top_k)
            return 0
        try:
            print_answer(rag.ask(args.question, top_k=args.top_k))
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        return 0

    # ---- 交互模式 ----
    console.print("\n[dim]输入问题开始提问；exit / quit / 退出 结束[/dim]\n")
    while True:
        try:
            question = console.input("[bold]你的问题：[/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已退出。[/dim]")
            return 130

        if question.lower() in {"exit", "quit", "退出"}:
            console.print("[dim]已退出。[/dim]")
            return 0
        if not question:
            continue

        try:
            if args.stream:
                stream_answer(rag, question, args.top_k)
            else:
                print_answer(rag.ask(question, top_k=args.top_k))
        except (ValueError, RuntimeError) as exc:
            # 单次问答失败不应终止整个会话
            console.print(f"[red]{exc}[/red]")
        console.print()


if __name__ == "__main__":
    sys.exit(main())
