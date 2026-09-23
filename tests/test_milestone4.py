"""Milestone 4: heuristic router. Easy vs. hard query pick different tiers.

Routing logic is pure, so most tests need no engine at all; the engine tests
use a fake provider that records which model it was asked for.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from gpt_efficient.cache import SemanticCache, cache_namespace
from gpt_efficient.config import (
    CacheConfig,
    HeuristicRouterConfig,
    ModelPrice,
    RouterConfig,
    Settings,
    TierTarget,
)
from gpt_efficient.engine import Engine
from gpt_efficient.router import (
    FixedRouter,
    HeuristicRouter,
    RouteDecision,
    build_router,
    heuristic_features,
    heuristic_score,
    pick_tier,
)
from gpt_efficient.schemas import CacheStatus, Completion, Message, Tier
from gpt_efficient.trace import TraceLogger

EASY = "What is the capital of France?"
HARD = (
    "Prove step by step that the integral of x^2 from 0 to 1 equals 1/3, "
    "then derive the general formula for the integral of x^n."
)
CODE = """Why does this crash?

```python
def f(xs):
    return xs[len(xs)]
```"""

MODELS = {
    Tier.LOCAL: "gemini-2.5-flash-lite",
    Tier.MID: "gemini-2.5-flash",
    Tier.FRONTIER: "gemini-3.1-pro-preview",
}


class RecordingProvider:
    name = "gemini"

    def __init__(self) -> None:
        self.models: list[str] = []

    def complete(self, messages: list[Message], max_tokens: int, model: str) -> Completion:
        self.models.append(model)
        return Completion(
            text=f"answer from {model}",
            tokens_in=10,
            tokens_out=10,
            cost_usd=0.0,
            latency_ms=1.0,
            model=model,
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        default_provider="gemini",
        default_tier=Tier.MID,
        tier_mode="three",
        tier_modes={
            "two": [Tier.LOCAL, Tier.MID],
            "three": [Tier.LOCAL, Tier.MID, Tier.FRONTIER],
        },
        tiers={t: TierTarget(model=m) for t, m in MODELS.items()},
        pricing={m: ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0) for m in MODELS.values()},
        router=RouterConfig(type="heuristic"),
        trace_db=tmp_path / "traces.db",
    )


def make_engine(settings: Settings, **kwargs):
    provider = RecordingProvider()
    logger = TraceLogger(settings.trace_db)
    return Engine(settings, {"gemini": provider}, logger, **kwargs), provider, logger


# --- the milestone test ------------------------------------------------------


def test_easy_and_hard_queries_pick_different_tiers(settings: Settings) -> None:
    engine, provider, logger = make_engine(settings)

    easy = engine.ask(EASY)
    hard = engine.ask(HARD)

    assert easy.tier == Tier.LOCAL
    assert hard.tier == Tier.FRONTIER
    # the routed tier's model is what actually got called and logged
    assert provider.models == [MODELS[Tier.LOCAL], MODELS[Tier.FRONTIER]]
    easy_row, hard_row = logger.all()
    assert (easy_row.tier, easy_row.model) == (Tier.LOCAL, MODELS[Tier.LOCAL])
    assert (hard_row.tier, hard_row.model) == (Tier.FRONTIER, MODELS[Tier.FRONTIER])


def test_two_tier_mode_caps_hard_queries_at_highest_active_tier(settings: Settings) -> None:
    two = settings.model_copy(update={"tier_mode": "two"})
    engine, provider, _ = make_engine(two)

    assert engine.ask(EASY).tier == Tier.LOCAL
    assert engine.ask(HARD).tier == Tier.MID
    assert MODELS[Tier.FRONTIER] not in provider.models


# --- pure feature / scoring / tier-picking functions ---------------------------


def test_features_detect_code_math_keywords_and_length() -> None:
    cfg = HeuristicRouterConfig()
    easy = heuristic_features(EASY, cfg)
    assert (easy.has_code, easy.has_math, easy.keyword_hits) == (False, False, [])

    hard = heuristic_features(HARD, cfg)
    assert hard.has_math
    assert {"prove", "step by step", "derive"} <= set(hard.keyword_hits)

    code = heuristic_features(CODE, cfg)
    assert code.has_code and not code.has_math

    assert heuristic_features("word " * 200, cfg).words == 200


@pytest.mark.parametrize(
    "text",
    [
        "```js\nconsole.log(1)\n```",
        "def parse(line):\n    return line.split()",
        "Traceback (most recent call last):\n  ValueError: bad",
        "SELECT name FROM users WHERE id = 3",
        "#include <stdio.h>",
    ],
)
def test_code_detection(text: str) -> None:
    assert heuristic_features(text, HeuristicRouterConfig()).has_code


@pytest.mark.parametrize(
    "text",
    ["Solve $x^2 - 4 = 0$", r"Evaluate \frac{1}{2} + \sqrt{2}", "∫ sin(x) dx", "if 3x + 2 = 11, find x"],
)
def test_math_detection(text: str) -> None:
    assert heuristic_features(text, HeuristicRouterConfig()).has_math


def test_prose_is_not_code_or_math() -> None:
    f = heuristic_features("Tell me a fun fact about otters; keep it short.", HeuristicRouterConfig())
    assert not f.has_code and not f.has_math


def test_score_uses_config_weights() -> None:
    cfg = HeuristicRouterConfig()
    f = heuristic_features(CODE, cfg)
    base = heuristic_score(f, cfg)
    heavier = cfg.model_copy(update={"code_points": cfg.code_points + 3})
    assert heuristic_score(f, heavier) == base + 3
    # keyword points are capped
    many = heuristic_features("prove derive analyze compare optimize design", cfg)
    assert heuristic_score(many, cfg) == cfg.max_keyword_points


def test_pick_tier() -> None:
    cutoffs = {Tier.MID: 1.0, Tier.FRONTIER: 3.0}
    three = [Tier.LOCAL, Tier.MID, Tier.FRONTIER]
    assert pick_tier(0, cutoffs, three) == Tier.LOCAL
    assert pick_tier(1, cutoffs, three) == Tier.MID
    assert pick_tier(2.9, cutoffs, three) == Tier.MID
    assert pick_tier(3, cutoffs, three) == Tier.FRONTIER
    assert pick_tier(99, cutoffs, [Tier.LOCAL, Tier.MID]) == Tier.MID
    # order of the active list doesn't matter; tiers are ranked cheapest first
    assert pick_tier(0, cutoffs, [Tier.FRONTIER, Tier.MID]) == Tier.MID


def test_heuristic_router_requires_cutoffs_for_active_upper_tiers(settings: Settings) -> None:
    bad = settings.model_copy(
        update={
            "router": RouterConfig(
                type="heuristic", heuristic=HeuristicRouterConfig(tier_cutoffs={Tier.MID: 1.0})
            )
        }
    )
    with pytest.raises(ValueError, match="frontier"):
        HeuristicRouter(bad)


def test_route_decision_explains_itself(settings: Settings) -> None:
    decision = HeuristicRouter(settings).route(HARD, [])
    assert isinstance(decision, RouteDecision)
    assert decision.tier == Tier.FRONTIER
    assert decision.score is not None and decision.score >= 3
    assert any("math" in r for r in decision.reasons)


# --- config / wiring -----------------------------------------------------------------


def test_router_type_from_config(settings: Settings) -> None:
    assert isinstance(build_router(settings), HeuristicRouter)
    fixed = settings.model_copy(update={"router": RouterConfig(type="fixed")})
    router = build_router(fixed)
    assert isinstance(router, FixedRouter)
    assert router.route(HARD, []).tier == Tier.MID  # always default_tier
    with pytest.raises(ValidationError):
        RouterConfig(type="magic")


def test_default_router_is_fixed_so_earlier_behaviour_is_unchanged() -> None:
    assert RouterConfig().type == "fixed"


def test_router_config_is_part_of_cache_key(settings: Settings) -> None:
    fixed = settings.model_copy(update={"router": RouterConfig(type="fixed")})
    tweaked = settings.model_copy(
        update={
            "router": RouterConfig(
                type="heuristic", heuristic=HeuristicRouterConfig(code_points=5)
            )
        }
    )
    ns = cache_namespace(settings)
    assert cache_namespace(fixed) != ns
    assert cache_namespace(tweaked) != ns


def test_cache_hit_skips_router(settings: Settings, tmp_path: Path) -> None:
    class SpyRouter:
        name = "spy"
        calls = 0

        def route(self, query: str, history: list[Message]) -> RouteDecision:
            SpyRouter.calls += 1
            return RouteDecision(tier=Tier.MID)

    class ConstEmbedder:
        name = "const"

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    cached = settings.model_copy(
        update={"cache": CacheConfig(enabled=True, db=tmp_path / "cache.db")}
    )
    engine, provider, logger = make_engine(
        cached,
        embedder=ConstEmbedder(),
        cache=SemanticCache(cached.cache.db),
        router=SpyRouter(),
    )
    engine.ask(EASY)
    engine.ask(EASY)

    assert SpyRouter.calls == 1
    assert [r.cache_status for r in logger.all()] == [CacheStatus.MISS, CacheStatus.HIT]


def test_dollar_amounts_are_not_latex() -> None:
    f = heuristic_features("I paid $5 and $10 for lunch, what's the total?", HeuristicRouterConfig())
    assert not f.has_math


def test_keywords_match_plurals() -> None:
    f = heuristic_features("Discuss the trade-offs.", HeuristicRouterConfig())
    assert f.keyword_hits == ["trade-off"]
