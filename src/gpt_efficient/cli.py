"""Thin Rich CLI: `gpte ask`, `gpte chat`, `gpte traces`."""

import argparse

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from gpt_efficient.cache import SemanticCache
from gpt_efficient.config import Settings
from gpt_efficient.engine import Engine
from gpt_efficient.providers import build_embedder, build_providers
from gpt_efficient.schemas import Message, TraceRow
from gpt_efficient.trace import TraceLogger

console = Console()


def _engine(settings: Settings) -> Engine:
    cache_on = settings.cache.enabled
    return Engine(
        settings,
        build_providers(settings),
        TraceLogger(settings.trace_db),
        embedder=build_embedder(settings) if cache_on else None,
        cache=SemanticCache(settings.cache.db) if cache_on else None,
    )


def _footer(row: TraceRow) -> str:
    sim = f" {row.cache_sim:.3f}" if row.cache_sim is not None else ""
    return (
        f"[dim]{row.tier}/{row.model} · cache {row.cache_status}{sim} · "
        f"in {row.tokens_in} · out {row.tokens_out} · "
        f"${row.cost_usd:.6f} · {row.latency_ms:.0f} ms[/dim]"
    )


def cmd_ask(settings: Settings, query: str) -> None:
    engine = _engine(settings)
    resp = engine.ask(query)
    console.print(Markdown(resp.text))
    console.print(_footer(engine.logger.all()[-1]))


def cmd_chat(settings: Settings) -> None:
    engine = _engine(settings)
    history: list[Message] = []
    console.print("[dim]Ctrl-D or /exit to quit.[/dim]")
    while True:
        try:
            query = console.input("[bold cyan]you> [/bold cyan]").strip()
        except EOFError:
            break
        if query in {"/exit", "/quit"}:
            break
        if not query:
            continue
        resp = engine.ask(query, history)
        console.print(Markdown(resp.text))
        console.print(_footer(engine.logger.all()[-1]))
        history += [Message(role="user", content=query), Message(role="assistant", content=resp.text)]


def cmd_traces(settings: Settings, limit: int) -> None:
    rows = TraceLogger(settings.trace_db).all()[-limit:]
    table = Table(*TraceRow.model_fields, show_lines=False)
    for r in rows:
        table.add_row(*(str(v) for v in r.model_dump(mode="json").values()))
    console.print(table)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="gpte")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="Send one query")
    ask.add_argument("query")
    sub.add_parser("chat", help="Interactive chat")
    traces = sub.add_parser("traces", help="Show recent trace rows")
    traces.add_argument("-n", type=int, default=20)
    args = parser.parse_args()

    settings = Settings()
    try:
        if args.cmd == "ask":
            cmd_ask(settings, args.query)
        elif args.cmd == "chat":
            cmd_chat(settings)
        else:
            cmd_traces(settings, args.n)
    except Exception as exc:
        console.print(f"[red]error:[/red] {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
