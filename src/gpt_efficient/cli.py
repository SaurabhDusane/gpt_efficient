"""Thin Rich CLI: `gpte ask`, `gpte chat`, `gpte traces`, `gpte eval`."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress
from rich.table import Table

from gpt_efficient.cache import SemanticCache
from gpt_efficient.config import Settings
from gpt_efficient.engine import Engine
from gpt_efficient.evals.dataset import load_dataset
from gpt_efficient.evals.judge import Judge
from gpt_efficient.evals.report import summarize, write_report
from gpt_efficient.evals.runner import apply_overrides, load_experiments, run_eval
from gpt_efficient.providers import build_embedder, build_providers
from gpt_efficient.schemas import Message, Tier, TraceRow
from gpt_efficient.trace import TraceLogger

console = Console()


def _engine(settings: Settings) -> Engine:
    cache_on = settings.cache.enabled
    return Engine(
        settings,
        build_providers(settings),
        TraceLogger(settings.trace_db),
        embedder=build_embedder(settings) if settings.needs_embedder else None,
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


def cmd_eval(settings: Settings, args: argparse.Namespace) -> None:
    from gpt_efficient.fakes import FakeEmbedder, FakeJudgeProvider, FakeProvider

    dataset = Path(args.dataset or settings.eval.dataset)
    items = load_dataset(dataset)[: args.limit]
    experiments = load_experiments(Path(args.experiments or settings.eval.experiments))
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = sorted(set(wanted) - {e.name for e in experiments})
        if unknown:
            raise ValueError(f"unknown experiments {unknown}; have {[e.name for e in experiments]}")
        experiments = [e for e in experiments if e.name in wanted]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + ("-fake" if args.fake else "")
    out = Path(args.out) if args.out else settings.eval.out_dir / stamp

    if args.fake:
        judge = Judge(settings, FakeJudgeProvider(settings))
        make_providers = lambda s: {n: FakeProvider(s) for n in _provider_names(s)}  # noqa: E731
        make_embedder = lambda s: FakeEmbedder(s.embedding_dim or 768)  # noqa: E731
    else:
        judge_provider = settings.judge.provider or settings.default_provider
        judge = Judge(settings, build_providers(settings)[judge_provider])
        make_providers, make_embedder = build_providers, build_embedder

    paid = [e.name for e in experiments if Tier.FRONTIER in apply_overrides(settings, e.overrides).active_tiers]
    console.print(
        f"{len(items)} items × {len(experiments)} experiments = {len(items) * len(experiments)} "
        f"requests (+ judge calls, {settings.judge.model}) → {out}"
    )
    if paid and not args.fake:
        console.print(f"[yellow]can reach the paid frontier tier:[/yellow] {', '.join(paid)}")

    with Progress(console=console, transient=True) as bar:
        tasks: dict[str, int] = {}

        def progress(name: str, done: int, total: int) -> None:
            if name not in tasks:
                tasks[name] = bar.add_task(name, total=total)
            bar.update(tasks[name], completed=done)

        results = run_eval(
            settings, experiments, items, out,
            make_providers=make_providers, make_embedder=make_embedder,
            judge=judge, progress=progress,
        )  # fmt: skip

    meta = {"dataset": str(dataset), "experiments": ", ".join(e.name for e in experiments)}
    if args.fake:
        meta["mode"] = "**FAKE** provider/embedder/judge — illustrative only, not results"
    paths = write_report(results, settings, out, meta)

    table = Table("experiment", "quality", "tokens/query", "$/1k queries", "cache hits (wrong)", "errors")
    for s in summarize(results, settings.eval.low_quality):
        q = "—" if s.mean_quality is None else f"{s.mean_quality:.3f}"
        tok = "—" if s.mean_tokens is None else f"{s.mean_tokens:,.0f}"
        usd = "—" if s.mean_cost_usd is None else f"${s.mean_cost_usd * 1000:.4f}"
        table.add_row(s.experiment, q, tok, usd, f"{s.cache_hits} ({s.wrong_cache_hits})", str(s.errors))
    console.print(table)
    console.print(f"report: {paths['report']}")


def _provider_names(settings: Settings) -> set[str]:
    return {settings.target(t).provider or settings.default_provider for t in settings.active_tiers}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="gpte")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="Send one query")
    ask.add_argument("query")
    sub.add_parser("chat", help="Interactive chat")
    traces = sub.add_parser("traces", help="Show recent trace rows")
    traces.add_argument("-n", type=int, default=20)
    ev = sub.add_parser("eval", help="Run the eval harness and write a report")
    ev.add_argument("--dataset", help="JSONL dataset (default: eval.dataset)")
    ev.add_argument("--experiments", help="experiments TOML (default: eval.experiments)")
    ev.add_argument("--only", help="comma-separated experiment names to run")
    ev.add_argument("--limit", type=int, help="only the first N dataset items")
    ev.add_argument("--out", help="output dir (default: eval.out_dir/<timestamp>)")
    ev.add_argument("--fake", action="store_true", help="offline fakes; illustrative only")
    args = parser.parse_args()

    settings = Settings()
    try:
        if args.cmd == "ask":
            cmd_ask(settings, args.query)
        elif args.cmd == "chat":
            cmd_chat(settings)
        elif args.cmd == "eval":
            cmd_eval(settings, args)
        else:
            cmd_traces(settings, args.n)
    except Exception as exc:
        console.print(f"[red]error:[/red] {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
