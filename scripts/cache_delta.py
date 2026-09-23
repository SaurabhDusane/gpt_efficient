"""Token/cost delta of the semantic cache vs. the no-cache baseline.

Runs the same query workload twice through the real Engine — cache disabled,
then cache enabled (fresh cache) — and prints totals from the trace rows.

    uv run python scripts/cache_delta.py                  # real config + API key
    uv run python scripts/cache_delta.py --queries q.txt  # one query per line
    uv run python scripts/cache_delta.py --fake           # offline, illustrative only

--fake swaps in a deterministic provider and a bag-of-words embedder, so it
shows the mechanism (exact repeats hit) but not real paraphrase behaviour.
"""

import argparse
import hashlib
import math
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from gpt_efficient.cache import SemanticCache
from gpt_efficient.config import Settings
from gpt_efficient.engine import Engine
from gpt_efficient.providers import build_embedder, build_providers
from gpt_efficient.schemas import CacheStatus, Completion, Message, TraceRow
from gpt_efficient.trace import TraceLogger

# Mix of unique questions, exact repeats and paraphrases.
DEFAULT_QUERIES = [
    "What is the capital of France?",
    "Explain what a hash table is in two sentences.",
    "How many bones are in the adult human body?",
    "What is the capital of France?",
    "What's the capital city of France?",
    "Explain what a hash table is in two sentences.",
    "Briefly explain hash tables in two sentences.",
    "What is the boiling point of water at sea level in Celsius?",
    "How many bones are in the adult human body?",
    "What is the capital of Spain?",
]


class FakeProvider:
    name = "fake"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, messages: list[Message], max_tokens: int, model: str) -> Completion:
        tokens_in = sum(len(m.content.split()) for m in messages) * 2
        text = f"(fake answer to: {messages[-1].content})"
        tokens_out = 60
        return Completion(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self.settings.cost_usd(model, tokens_in, tokens_out),
            latency_ms=0.0,
            model=model,
        )


class FakeEmbedder:
    name = "fake"

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def run(settings: Settings, queries: list[str], fake: bool, workdir: Path, label: str) -> list[TraceRow]:
    settings = settings.model_copy(
        update={
            "trace_db": workdir / f"{label}-traces.db",
            "cache": settings.cache.model_copy(update={"db": workdir / f"{label}-cache.db"}),
        }
    )
    if fake:
        providers = {p: FakeProvider(settings) for p in {settings.target(t).provider for t in settings.active_tiers}}
        embedder = FakeEmbedder(settings.embedding_dim or 768)
    else:
        providers = build_providers(settings)
        embedder = build_embedder(settings) if settings.cache.enabled else None
    logger = TraceLogger(settings.trace_db)
    cache = SemanticCache(settings.cache.db) if settings.cache.enabled else None
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


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, help="file with one query per line")
    parser.add_argument("--fake", action="store_true", help="offline fake provider/embedder")
    args = parser.parse_args()

    queries = (
        [q.strip() for q in args.queries.read_text().splitlines() if q.strip()]
        if args.queries
        else DEFAULT_QUERIES
    )
    settings = Settings()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        off = settings.cache.model_copy(update={"enabled": False})
        on = settings.cache.model_copy(update={"enabled": True})
        base = totals(run(settings.model_copy(update={"cache": off}), queries, args.fake, work, "base"))
        cached_rows = run(settings.model_copy(update={"cache": on}), queries, args.fake, work, "cached")
    cached = totals(cached_rows)

    console = Console()
    title = f"Semantic cache vs. baseline (threshold {settings.cache.threshold})"
    if args.fake:
        title += " — FAKE provider/embedder, illustrative only"
    table = Table("metric", "baseline", "cache", "delta", title=title)
    for k in base:
        b, c = base[k], cached[k]
        fmt = (lambda x: f"{x:.6f}") if k == "cost_usd" else (lambda x: f"{x:g}")
        pct = f" ({(c - b) / b:+.0%})" if b else ""
        table.add_row(k, fmt(b), fmt(c), f"{fmt(c - b)}{pct}")
    console.print(table)

    sims = Table("query", "status", "cache_sim", title="cached run: nearest-neighbour similarity")
    for q, r in zip(queries, cached_rows, strict=True):
        sim = f"{r.cache_sim:.4f}" if r.cache_sim is not None else "-"
        sims.add_row(q, str(r.cache_status), sim)
    console.print(sims)


if __name__ == "__main__":
    main()
