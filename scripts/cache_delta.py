"""Token/cost delta of the semantic cache vs. the no-cache baseline.

Runs the same query workload twice through the real Engine — cache disabled,
then cache enabled (fresh cache) — and prints totals from the trace rows.
The router is whatever config.toml says, identical in both runs.

    uv run python scripts/cache_delta.py                  # real config + API key
    uv run python scripts/cache_delta.py --queries q.txt  # one query per line
    uv run python scripts/cache_delta.py --fake           # offline, illustrative only

--fake swaps in a deterministic provider and a bag-of-words embedder, so it
shows the mechanism (exact repeats hit) but not real paraphrase behaviour.
"""

import argparse
import tempfile
from pathlib import Path

from _common import load_queries, print_delta, run, totals
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from gpt_efficient.config import Settings

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


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, help="file with one query per line")
    parser.add_argument("--fake", action="store_true", help="offline fake provider/embedder")
    args = parser.parse_args()

    queries = load_queries(args.queries, DEFAULT_QUERIES)
    settings = Settings()
    off = settings.model_copy(update={"cache": settings.cache.model_copy(update={"enabled": False})})
    on = settings.model_copy(update={"cache": settings.cache.model_copy(update={"enabled": True})})
    with tempfile.TemporaryDirectory() as tmp:
        base = totals(run(off, queries, args.fake, Path(tmp), "base"))
        cached_rows = run(on, queries, args.fake, Path(tmp), "cached")

    console = Console()
    title = f"Semantic cache vs. baseline (threshold {settings.cache.threshold})"
    if args.fake:
        title += " — FAKE provider/embedder, illustrative only"
    print_delta(console, title, base, totals(cached_rows), ("no cache", "cache"))

    sims = Table("query", "status", "cache_sim", title="cached run: nearest-neighbour similarity")
    for q, r in zip(queries, cached_rows, strict=True):
        sim = f"{r.cache_sim:.4f}" if r.cache_sim is not None else "-"
        sims.add_row(q, str(r.cache_status), sim)
    console.print(sims)


if __name__ == "__main__":
    main()
