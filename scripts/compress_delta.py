"""Token/cost delta of each context-compression strategy vs. sending full history.

Runs every conversation in evals/conversations.jsonl (history + final query)
once per strategy, with routing fixed to default_tier and the cache off, and
prints totals from the trace rows.

    uv run python scripts/compress_delta.py            # real config + API key
    uv run python scripts/compress_delta.py --fake     # offline, illustrative only
    uv run python scripts/compress_delta.py --dataset my_conversations.jsonl

Compression trades tokens against quality: this script shows the token/cost
side only. Whether answers held up is the eval harness's job:
    gpte eval --dataset evals/conversations.jsonl --experiments evals/experiments_compression.toml
"""

import argparse
import tempfile
from pathlib import Path

from _common import print_delta, run, totals
from dotenv import load_dotenv
from rich.console import Console

from gpt_efficient.config import RouterConfig, Settings
from gpt_efficient.evals.dataset import load_dataset

STRATEGIES = ["truncate", "summary", "retrieval", "summary+retrieval"]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("evals/conversations.jsonl"))
    parser.add_argument("--fake", action="store_true", help="offline fake provider/embedder")
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    queries = [i.query for i in items]
    histories = [i.history for i in items]
    settings = Settings()
    fixed = settings.model_copy(
        update={
            "router": RouterConfig(type="fixed"),
            "cache": settings.cache.model_copy(update={"enabled": False}),
        }
    )

    def with_strategy(strategy: str) -> Settings:
        return fixed.model_copy(
            update={"compressor": fixed.compressor.model_copy(update={"strategy": strategy})}
        )

    console = Console()
    with tempfile.TemporaryDirectory() as tmp:
        base_rows = run(with_strategy("none"), queries, args.fake, Path(tmp), "none", histories)
        base = totals(base_rows)
        for strategy in STRATEGIES:
            rows = run(with_strategy(strategy), queries, args.fake, Path(tmp), strategy, histories)
            t = totals(rows)
            t["compressed"] = sum(r.compressed for r in rows)
            t["history tokens saved (gross)"] = sum(r.tokens_saved for r in rows)
            b = {**base, "compressed": 0, "history tokens saved (gross)": 0}
            title = (
                f"'{strategy}' vs. full history ({len(items)} conversations, "
                f"trigger {fixed.compressor.trigger_tokens}, keep {fixed.compressor.keep_recent_turns} turns)"
            )
            if args.fake:
                title += " — FAKE, illustrative only"
            print_delta(console, title, b, t, ("full history", strategy))


if __name__ == "__main__":
    main()
