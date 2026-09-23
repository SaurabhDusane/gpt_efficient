"""Deterministic offline stand-ins for providers, embedder and judge.

Used by `gpte eval --fake` and the scripts/*_delta.py `--fake` modes to show a
mechanism working without API keys. Numbers they produce are illustrative only:
token counts and fake judge scores simply scale with the tier ladder.
"""

import hashlib
import json
import math

from gpt_efficient.config import Settings
from gpt_efficient.schemas import Completion, Message, Tier

# Per tier rank (cheapest first): fake output tokens (a stand-in for thinking
# growing with model size) and the fake judge's score for that tier's answers.
_FAKE_TOKENS_OUT = [60, 150, 400]
_FAKE_SCORES = [5, 7, 9]


def _rank(settings: Settings, model: str) -> int:
    ladder = [t for t in Tier if t in settings.tiers]
    for i, t in enumerate(ladder):
        if settings.tiers[t].model == model:
            return min(i, len(_FAKE_TOKENS_OUT) - 1)
    return 1


class FakeProvider:
    """Answers name the model; cost still comes from configured per-model prices."""

    name = "fake"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(
        self, messages: list[Message], max_tokens: int, model: str, temperature: float | None = None
    ) -> Completion:
        tokens_in = sum(len(m.content.split()) for m in messages) * 2
        tokens_out = _FAKE_TOKENS_OUT[_rank(self.settings, model)]
        return Completion(
            text=f"(fake {model} answer to: {messages[-1].content})",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self.settings.cost_usd(model, tokens_in, tokens_out),
            latency_ms=0.0,
            model=model,
        )


class FakeEmbedder:
    """Bag-of-words hashing embedder: exact repeats match, paraphrases only loosely."""

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


class FakeJudgeProvider:
    """Scores a FakeProvider answer by the tier of the model named in it.

    A real judge is blind to the model; this fake peeks at the fake answer's
    text purely so an offline run draws a non-trivial frontier.
    """

    name = "fake-judge"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(
        self, messages: list[Message], max_tokens: int, model: str, temperature: float | None = None
    ) -> Completion:
        candidate = messages[-1].content.split("CANDIDATE ANSWER")[-1]
        score = 5
        for t, target in self.settings.tiers.items():
            if f"fake {target.model} answer" in candidate:
                score = _FAKE_SCORES[_rank(self.settings, target.model)]
        try:
            cost = self.settings.cost_usd(model, 300, 50)
        except ValueError:
            cost = 0.0
        return Completion(
            text=json.dumps({"rationale": "fake judge", "score": score}),
            tokens_in=300,
            tokens_out=50,
            cost_usd=cost,
            latency_ms=0.0,
            model=model,
        )
