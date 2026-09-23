"""Token/cost delta of the configured router vs. the fixed-tier baseline.

Runs the same workload twice with the cache OFF (to isolate routing):
router.type = "fixed" (always default_tier — the pre-router baseline), then
the router from config.toml. Prints totals and each query's routed tier.

    uv run python scripts/router_delta.py                  # real config + API key
    uv run python scripts/router_delta.py --queries q.txt  # one query per line
    uv run python scripts/router_delta.py --fake           # offline, illustrative only

Routing trades cost against quality: this script shows the cost side only;
whether answer quality held is the eval harness's job (milestone 5).
With real keys, tier_mode "three" sends hard queries to the paid Pro tier.
"""

import argparse
import tempfile
from collections import Counter
from pathlib import Path

from _common import load_queries, print_delta, run, totals
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from gpt_efficient.config import RouterConfig, Settings
from gpt_efficient.router import build_router

# Mix of easy, medium and hard queries.
DEFAULT_QUERIES = [
    "What is the capital of France?",
    "Translate 'good morning' into Spanish.",
    "Write a haiku about autumn.",
    "How many bones are in the adult human body?",
    "Compare REST and GraphQL for a mobile app backend.",
    "My tests fail with KeyError: 'id' when I call parse(); why?",
    "Design a URL shortener that handles 10k writes/sec and discuss the trade-offs.",
    "Prove step by step that the integral of x^2 from 0 to 1 equals 1/3.",
    "Explain why the sky is blue.",
    "Summarize the plot of Hamlet in three sentences.",
]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, help="file with one query per line")
    parser.add_argument("--fake", action="store_true", help="offline fake provider")
    args = parser.parse_args()

    queries = load_queries(args.queries, DEFAULT_QUERIES)
    settings = Settings()
    no_cache = settings.cache.model_copy(update={"enabled": False})
    routed = settings.model_copy(update={"cache": no_cache})
    fixed = routed.model_copy(update={"router": RouterConfig(type="fixed")})
    with tempfile.TemporaryDirectory() as tmp:
        base_rows = run(fixed, queries, args.fake, Path(tmp), "fixed")
        new_rows = run(routed, queries, args.fake, Path(tmp), "routed")

    console = Console()
    title = (
        f"Router '{settings.router.type}' vs. fixed '{settings.default_tier}' "
        f"(tier_mode {settings.tier_mode}, cache off)"
    )
    if args.fake:
        title += " — FAKE provider, illustrative only"
    print_delta(console, title, totals(base_rows), totals(new_rows), (f"fixed {settings.default_tier}", settings.router.type))

    mix = Counter(r.tier.value for r in new_rows)
    console.print("tier mix: " + ", ".join(f"{t} {n}" for t, n in sorted(mix.items())))

    router = build_router(routed)
    per_query = Table("query", "tier", "score", "reasons", "cost_usd", title="routed run: per query")
    for q, r in zip(queries, new_rows, strict=True):
        d = router.route(q, [])
        score = "-" if d.score is None else f"{d.score:g}"
        per_query.add_row(q, r.tier.value, score, "; ".join(d.reasons), f"{r.cost_usd:.6f}")
    console.print(per_query)


if __name__ == "__main__":
    main()
