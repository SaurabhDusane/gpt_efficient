"""Eval dataset: JSONL, one item per line, run in file order."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from gpt_efficient.schemas import Message

Difficulty = Literal["easy", "medium", "hard"]
Category = Literal["factual", "reasoning", "math", "code", "writing", "conversation"]


class EvalItem(BaseModel):
    id: str
    query: str
    # What a good answer contains. For open-ended items, the key points the
    # judge should look for rather than a single canonical text.
    reference: str
    difficulty: Difficulty
    category: Category
    # Short canonical answer for a deterministic match (judge sanity check).
    exact: str | None = None
    # Id of an earlier item this one paraphrases (same answer) — a cache-hit probe.
    paraphrase_of: str | None = None
    # e.g. "near-miss:<id>": similar wording, different answer — a wrong-hit probe.
    tags: list[str] = []
    # Prior turns sent with the query (multi-turn items; exercises the compressor).
    history: list[Message] = []


def load_dataset(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = EvalItem.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path}: line {n}: {exc}") from None
        if item.id in seen:
            raise ValueError(f"{path}: line {n}: duplicate id {item.id!r}")
        # Paraphrases must follow their original so the cache can have seen it.
        if item.paraphrase_of is not None and item.paraphrase_of not in seen:
            raise ValueError(
                f"{path}: line {n}: paraphrase_of {item.paraphrase_of!r} is not an earlier item"
            )
        seen.add(item.id)
        items.append(item)
    return items
