"""Routers pick the tier that answers a query.

Every router implements `Router.route(query, history) -> RouteDecision`; the
type is chosen by `router.type` in config. The heuristic router's logic is
pure functions (features -> score -> tier) so it's testable without an engine.
"""

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from gpt_efficient.cache import estimate_tokens
from gpt_efficient.config import HeuristicRouterConfig, Settings
from gpt_efficient.providers.base import Embedder
from gpt_efficient.schemas import Message, Tier


class RouteDecision(BaseModel):
    tier: Tier
    score: float | None = None
    confidence: float | None = None  # heuristic: None; learned router: model confidence
    escalated: bool = False  # routed one tier above the prediction (low confidence)
    embed_tokens: int = 0  # tokens the router itself embedded (learned router)
    reasons: list[str] = []


@runtime_checkable
class Router(Protocol):
    name: str

    def route(self, query: str, history: list[Message]) -> RouteDecision: ...


# --- heuristic features ---------------------------------------------------------

# Signals that a query contains code (or asks about a stack trace / SQL).
_CODE_PATTERNS = [
    r"```",
    r"^\s*(def|class)\s+\w+",
    r"^\s*from\s+[\w.]+\s+import\b",
    r"^\s*import\s+[\w.]+\s*$",
    r"#include\s*<",
    r"\bfunction\s+\w+\s*\(",
    r"\w+\([^()\n]*\)\s*[{;]",
    r"\b\w+\.\w+\([^()\n]*\)",
    r"\b(SELECT|INSERT|UPDATE|DELETE)\b.+\b(FROM|INTO|SET|WHERE)\b",
    r"Traceback \(most recent call last\)",
    r"\b\w+(Error|Exception):",
]
# Signals that a query contains math notation or asks for math.
_MATH_PATTERNS = [
    r"\$(?=[^\s$\d])[^$\n]*\$",  # inline LaTeX; not "$5 and $10"
    r"\\(frac|sqrt|int|sum|prod|lim|partial|infty)\b",
    r"[∫∑∏√∞≤≥≠±∂π]",
    r"\b\w+\s*\^\s*\w+",
    r"[\w)]\s*[-+*/]\s*[\w(]+\s*=\s*[-\w(]",
    r"\b(integral|derivative|differentiate|eigen\w*|matrix|matrices|theorem|equation|polynomial|logarithm)s?\b",
]
_CODE_RE = [re.compile(p, re.MULTILINE) for p in _CODE_PATTERNS]
_MATH_RE = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in _MATH_PATTERNS]


class HeuristicFeatures(BaseModel):
    words: int
    has_code: bool
    has_math: bool
    keyword_hits: list[str]


def heuristic_features(query: str, cfg: HeuristicRouterConfig) -> HeuristicFeatures:
    lowered = query.lower()
    return HeuristicFeatures(
        words=len(query.split()),
        has_code=any(r.search(query) for r in _CODE_RE),
        has_math=any(r.search(query) for r in _MATH_RE),
        keyword_hits=[
            kw
            for kw in cfg.keywords
            if re.search(rf"\b{re.escape(kw.lower())}(s|es)?\b", lowered)
        ],
    )


def _scored(f: HeuristicFeatures, cfg: HeuristicRouterConfig) -> tuple[float, list[str]]:
    parts: list[tuple[float, str]] = []
    if f.words >= cfg.long_query_words:
        parts.append((cfg.length_points, f"long query ({f.words} words)"))
    if f.words >= cfg.very_long_query_words:
        parts.append((cfg.length_points, "very long query"))
    if f.keyword_hits:
        pts = min(len(f.keyword_hits) * cfg.keyword_points, cfg.max_keyword_points)
        parts.append((pts, f"keywords {f.keyword_hits}"))
    if f.has_code:
        parts.append((cfg.code_points, "code"))
    if f.has_math:
        parts.append((cfg.math_points, "math"))
    return sum(p for p, _ in parts), [f"{why} +{p:g}" for p, why in parts]


def heuristic_score(f: HeuristicFeatures, cfg: HeuristicRouterConfig) -> float:
    return _scored(f, cfg)[0]


def rank(tiers: list[Tier]) -> list[Tier]:
    """Cheapest first, in Tier declaration order (LOCAL < MID < FRONTIER)."""
    order = list(Tier)
    return sorted(tiers, key=order.index)


def pick_tier(score: float, cutoffs: dict[Tier, float], active: list[Tier]) -> Tier:
    """Highest active tier whose cutoff `score` meets; the cheapest active tier is the floor."""
    ranked = rank(active)
    chosen = ranked[0]
    for tier in ranked[1:]:
        if tier in cutoffs and score >= cutoffs[tier]:
            chosen = tier
    return chosen


# --- routers --------------------------------------------------------------------


class FixedRouter:
    """Always the configured default tier: the no-router baseline."""

    name = "fixed"

    def __init__(self, settings: Settings) -> None:
        self._tier = settings.default_tier

    def route(self, query: str, history: list[Message]) -> RouteDecision:
        return RouteDecision(tier=self._tier, reasons=["fixed default_tier"])


class HeuristicRouter:
    """Length + complexity keywords + code/math presence -> tier. Ignores history."""

    name = "heuristic"

    def __init__(self, settings: Settings) -> None:
        self._cfg = settings.router.heuristic
        self._active = settings.active_tiers
        missing = [t.value for t in rank(self._active)[1:] if t not in self._cfg.tier_cutoffs]
        if missing:
            raise ValueError(f"router.heuristic.tier_cutoffs has no cutoff for active tiers {missing}")

    def route(self, query: str, history: list[Message]) -> RouteDecision:
        features = heuristic_features(query, self._cfg)
        score, reasons = _scored(features, self._cfg)
        return RouteDecision(
            tier=pick_tier(score, self._cfg.tier_cutoffs, self._active),
            score=score,
            reasons=reasons,
        )


class LearnedRouter:
    """Embedding classifier (learned_router.RouterModel) -> tier + confidence.

    Probabilities are renormalized over the active tiers; below the confidence
    threshold the router escalates one active tier up (if `escalate`).
    """

    name = "learned"

    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        from gpt_efficient.learned_router import RouterModel  # avoid an import cycle

        cfg = settings.router.learned
        if not cfg.model_path.exists():
            raise FileNotFoundError(
                f"no learned router model at {cfg.model_path}; "
                "label and train one with `gpte router label` then `gpte router train`"
            )
        self.model = RouterModel.model_validate_json(cfg.model_path.read_text())
        if self.model.embedding_model != settings.embedding_model or (
            settings.embedding_dim is not None and self.model.embedding_dim != settings.embedding_dim
        ):
            raise ValueError(
                f"router model was trained on embeddings from {self.model.embedding_model} "
                f"({self.model.embedding_dim}-d), but config uses {settings.embedding_model} "
                f"({settings.embedding_dim}-d); retrain it"
            )
        self.cfg = cfg
        self.embedder = embedder
        self.active = rank(settings.active_tiers)
        self.chars_per_token = settings.embedding_chars_per_token

    def route(self, query: str, history: list[Message]) -> RouteDecision:
        [vector] = self.embedder.embed([query])
        probs = {t: p for t, p in self.model.probabilities(vector).items() if t in self.active}
        if not probs:
            raise ValueError(f"router model classes {self.model.classes} are all inactive")
        total = sum(probs.values())
        probs = {t: p / total for t, p in probs.items()}
        predicted = max(probs, key=lambda t: probs[t])
        confidence = probs[predicted]
        tier, escalated = predicted, False
        higher = self.active[self.active.index(predicted) + 1 :]
        if self.cfg.escalate and confidence < self.cfg.confidence_threshold and higher:
            tier, escalated = higher[0], True
        reasons = [", ".join(f"p({t.value})={p:.2f}" for t, p in probs.items())]
        if escalated:
            reasons.append(f"confidence {confidence:.2f} < {self.cfg.confidence_threshold}: escalated")
        return RouteDecision(
            tier=tier,
            confidence=confidence,
            escalated=escalated,
            embed_tokens=estimate_tokens(query, self.chars_per_token),
            reasons=reasons,
        )


def build_router(settings: Settings, embedder: Embedder | None = None) -> Router:
    match settings.router.type:
        case "fixed":
            return FixedRouter(settings)
        case "heuristic":
            return HeuristicRouter(settings)
        case "learned":
            if embedder is None:
                raise ValueError("the learned router needs an embedder")
            return LearnedRouter(settings, embedder)
