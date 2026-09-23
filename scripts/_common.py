"""Shared helpers for the scripts/*_delta.py savings scripts."""

from pathlib import Path

from rich.console import Console
from rich.table import Table

from gpt_efficient.cache import SemanticCache
from gpt_efficient.config import Settings
from gpt_efficient.engine import Engine
from gpt_efficient.fakes import FakeEmbedder, FakeProvider
from gpt_efficient.providers import build_embedder, build_providers
from gpt_efficient.schemas import CacheStatus, TraceRow
from gpt_efficient.trace import TraceLogger


def load_queries(path: Path | None, default: list[str]) -> list[str]:
    if path is None:
        return default
    return [q.strip() for q in path.read_text().splitlines() if q.strip()]


def run(settings: Settings, queries: list[str], fake: bool, workdir: Path, label: str) -> list[TraceRow]:
    """Run `queries` through a fresh Engine (own trace + cache DB) and return its rows."""
    settings = settings.model_copy(
        update={
            "trace_db": workdir / f"{label}-traces.db",
            "cache": settings.cache.model_copy(update={"db": workdir / f"{label}-cache.db"}),
        }
    )
    cache_on = settings.cache.enabled
    if fake:
        names = {settings.target(t).provider for t in settings.active_tiers}
        providers = {n: FakeProvider(settings) for n in names}
        embedder = FakeEmbedder(settings.embedding_dim or 768) if cache_on else None
    else:
        providers = build_providers(settings)
        embedder = build_embedder(settings) if cache_on else None
    logger = TraceLogger(settings.trace_db)
    cache = SemanticCache(settings.cache.db) if cache_on else None
    engine = Engine(settings, providers, logger, embedder=embedder, cache=cache)
    for q in queries:
        engine.ask(q)
    return logger.all()


def totals(rows: list[TraceRow]) -> dict[str, float]:
    return {
        "requests": len(rows),
        "cache hits": sum(r.cache_status == CacheStatus.HIT for r in rows),
        "LLM calls": sum(r.cache_status != CacheStatus.HIT for r in rows),
        "tokens_in": sum(r.tokens_in for r in rows),
        "tokens_out": sum(r.tokens_out for r in rows),
        "embed_tokens (est.)": sum(r.embed_tokens for r in rows),
        "total tokens": sum(r.tokens_in + r.tokens_out + r.embed_tokens for r in rows),
        "cost_usd": sum(r.cost_usd for r in rows),
    }


def print_delta(console: Console, title: str, base: dict[str, float], new: dict[str, float], labels: tuple[str, str]) -> None:
    table = Table("metric", *labels, "delta", title=title)
    for k in base:
        b, c = base[k], new[k]
        fmt = (lambda x: f"{x:.6f}") if k == "cost_usd" else (lambda x: f"{x:g}")
        pct = f" ({(c - b) / b:+.0%})" if b else ""
        table.add_row(k, fmt(b), fmt(c), f"{fmt(c - b)}{pct}")
    console.print(table)
