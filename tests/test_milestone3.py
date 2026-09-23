"""Milestone 3: semantic cache. Repeat/paraphrased query -> cache hit.

Provider and embedder are fakes with hand-built vectors of known cosine
similarity, so no API key or network is needed.
"""

import math
from pathlib import Path

import pytest

from gpt_efficient.cache import CacheEntry, SemanticCache, cache_namespace, estimate_tokens
from gpt_efficient.config import CacheConfig, ModelPrice, Settings, TierTarget
from gpt_efficient.engine import Engine
from gpt_efficient.schemas import CacheStatus, Completion, Message, Tier
from gpt_efficient.trace import TraceLogger


def unit(cos: float) -> list[float]:
    """A 4-d unit vector whose cosine with [1, 0, 0, 0] is `cos`."""
    return [cos, math.sqrt(1 - cos**2), 0.0, 0.0]


QUERY = "What is the capital of France?"
VECTORS = {
    QUERY: unit(1.0),
    "what's france's capital city": unit(0.97),  # paraphrase
    "What is the population of France?": unit(0.80),  # same topic, different question
    "Explain quicksort.": [0.0, 0.0, 1.0, 0.0],  # unrelated
}


class FakeEmbedder:
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [VECTORS[t] for t in texts]


class FakeProvider:
    name = "gemini"

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def complete(self, messages: list[Message], max_tokens: int, model: str) -> Completion:
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider down")
        return Completion(
            text=f"answer #{self.calls} to: {messages[-1].content}",
            tokens_in=20,
            tokens_out=40,
            cost_usd=0.0001,
            latency_ms=5.0,
            model=model,
        )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        default_provider="gemini",
        default_tier=Tier.MID,
        tiers={Tier.MID: TierTarget(model="gemini-2.5-flash")},
        pricing={"gemini-2.5-flash": ModelPrice(input_per_mtok=0.30, output_per_mtok=2.50)},
        embedding_model="gemini-embedding-2",
        embedding_dim=4,
        embedding_price_per_mtok=0.20,
        embedding_chars_per_token=4.0,
        trace_db=tmp_path / "traces.db",
        cache=CacheConfig(enabled=True, threshold=0.95, db=tmp_path / "cache.db"),
    )


def make_engine(settings: Settings, provider: FakeProvider | None = None):
    provider = provider or FakeProvider()
    embedder = FakeEmbedder()
    logger = TraceLogger(settings.trace_db)
    engine = Engine(
        settings,
        providers={"gemini": provider},
        logger=logger,
        embedder=embedder,
        cache=SemanticCache(settings.cache.db),
    )
    return engine, provider, embedder, logger


# --- the milestone test ------------------------------------------------------


def test_repeat_query_hits_cache(settings: Settings) -> None:
    engine, provider, _, logger = make_engine(settings)

    first = engine.ask(QUERY)
    second = engine.ask(QUERY)

    assert second.text == first.text
    assert provider.calls == 1  # the repeat never reached the LLM
    miss, hit = logger.all()
    assert miss.cache_status == CacheStatus.MISS
    assert miss.cache_sim is None  # cache was empty
    assert hit.cache_status == CacheStatus.HIT
    assert hit.cache_sim == pytest.approx(1.0, abs=1e-5)
    assert (hit.tokens_in, hit.tokens_out) == (0, 0)
    assert hit.response_len == len(first.text)
    # a hit is served as the tier/model that produced the cached answer
    assert (hit.tier, hit.provider, hit.model) == (Tier.MID, "gemini", "gemini-2.5-flash")


def test_paraphrase_hits_cache(settings: Settings) -> None:
    engine, provider, _, logger = make_engine(settings)

    first = engine.ask(QUERY)
    para = engine.ask("what's france's capital city")

    assert para.text == first.text
    assert provider.calls == 1
    hit = logger.all()[-1]
    assert hit.cache_status == CacheStatus.HIT
    assert hit.cache_sim == pytest.approx(0.97, abs=1e-4)


def test_different_question_misses_and_logs_best_sim(settings: Settings) -> None:
    engine, provider, _, logger = make_engine(settings)

    engine.ask(QUERY)
    engine.ask("What is the population of France?")

    assert provider.calls == 2
    row = logger.all()[-1]
    assert row.cache_status == CacheStatus.MISS
    # misses record the nearest neighbour's similarity too (threshold analysis)
    assert row.cache_sim == pytest.approx(0.80, abs=1e-4)


# --- threshold, key and policy ------------------------------------------------


def test_threshold_comes_from_config(settings: Settings) -> None:
    strict = settings.model_copy(
        update={"cache": settings.cache.model_copy(update={"threshold": 0.98})}
    )
    engine, provider, _, logger = make_engine(strict)

    engine.ask(QUERY)
    engine.ask("what's france's capital city")  # sim 0.97 < 0.98

    assert provider.calls == 2
    assert logger.all()[-1].cache_status == CacheStatus.MISS


def test_config_change_that_alters_answers_changes_cache_key(settings: Settings) -> None:
    engine, provider, _, _ = make_engine(settings)
    engine.ask(QUERY)

    changed = settings.model_copy(update={"system_prompt": "Answer in French."})
    assert cache_namespace(changed) != cache_namespace(settings)
    engine2, provider2, _, logger2 = make_engine(changed)
    engine2.ask(QUERY)

    assert provider2.calls == 1  # no stale hit across system prompts
    assert logger2.all()[-1].cache_status == CacheStatus.MISS


def test_namespace_ignores_irrelevant_settings(settings: Settings) -> None:
    moved = settings.model_copy(update={"trace_db": Path("elsewhere.db")})
    assert cache_namespace(moved) == cache_namespace(settings)


def test_history_bypasses_cache(settings: Settings) -> None:
    engine, provider, embedder, logger = make_engine(settings)
    engine.ask(QUERY)
    history = [Message(role="user", content="Hi"), Message(role="assistant", content="Hello")]

    engine.ask(QUERY, history)

    assert provider.calls == 2
    assert len(embedder.calls) == 1  # no embed for the bypassed request
    row = logger.all()[-1]
    assert row.cache_status == CacheStatus.BYPASS
    assert row.embed_tokens == 0


def test_cache_disabled(settings: Settings) -> None:
    off = settings.model_copy(update={"cache": settings.cache.model_copy(update={"enabled": False})})
    engine, provider, embedder, logger = make_engine(off)

    engine.ask(QUERY)
    engine.ask(QUERY)

    assert provider.calls == 2
    assert embedder.calls == []
    assert {r.cache_status for r in logger.all()} == {CacheStatus.DISABLED}


def test_failed_request_is_not_cached(settings: Settings) -> None:
    engine, _, _, logger = make_engine(settings, FakeProvider(fail=True))
    with pytest.raises(RuntimeError):
        engine.ask(QUERY)
    assert logger.all()[-1].error == "RuntimeError: provider down"

    engine2, provider2, _, _ = make_engine(settings)
    engine2.ask(QUERY)
    assert provider2.calls == 1  # the failure left nothing in the cache


# --- embedding cost accounting --------------------------------------------------


def test_embed_tokens_and_cost_are_logged(settings: Settings) -> None:
    engine, _, _, logger = make_engine(settings)

    engine.ask(QUERY)
    engine.ask(QUERY)

    miss, hit = logger.all()
    embed_tokens = estimate_tokens(QUERY, 4.0)
    embed_cost = embed_tokens * 0.20 / 1e6
    assert miss.embed_tokens == hit.embed_tokens == embed_tokens
    assert miss.cost_usd == pytest.approx(0.0001 + embed_cost)  # LLM + embedding
    assert hit.cost_usd == pytest.approx(embed_cost)  # a hit still pays for the lookup


def test_estimate_tokens() -> None:
    assert estimate_tokens("", 4.0) == 0
    assert estimate_tokens("abc", 4.0) == 1
    assert estimate_tokens("a" * 9, 4.0) == 3
    assert estimate_tokens("a" * 9, 3.0) == 3


# --- the store itself -----------------------------------------------------------


def test_store_nearest_and_namespace_isolation(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path / "c.db")
    entry = CacheEntry(
        query="q", response="r", tier=Tier.MID, provider="gemini", model="gemini-2.5-flash"
    )
    assert cache.nearest(unit(1.0), "ns-a") is None

    cache.store(unit(1.0), "ns-a", entry)

    match = cache.nearest(unit(0.97), "ns-a")
    assert match is not None
    assert match.entry == entry
    assert match.similarity == pytest.approx(0.97, abs=1e-4)
    assert cache.nearest(unit(1.0), "ns-b") is None  # other namespace sees nothing
    # persists across instances
    assert SemanticCache(tmp_path / "c.db").nearest(unit(1.0), "ns-a") is not None


def test_trace_db_from_earlier_milestone_gains_new_columns(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:  # schema as of milestone 2 (no embed_tokens)
        conn.execute(
            "CREATE TABLE traces (id PRIMARY KEY, ts, query_hash, cache_status, cache_sim, tier, "
            "provider, model, tokens_in, tokens_out, cost_usd, latency_ms, compressed, "
            "tokens_saved, escalated, response_len, error)"
        )
    settings = Settings(
        tiers={Tier.MID: TierTarget(model="m")},
        pricing={"m": ModelPrice(input_per_mtok=1, output_per_mtok=1)},
        trace_db=db,
    )
    logger = TraceLogger(db)
    Engine(settings, {"gemini": FakeProvider()}, logger).ask("hi")
    [row] = logger.all()
    assert row.embed_tokens == 0


def test_embedding_failure_logs_one_miss_row(settings: Settings) -> None:
    engine, provider, embedder, logger = make_engine(settings)

    def boom(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("no key")

    embedder.embed = boom
    with pytest.raises(RuntimeError):
        engine.ask(QUERY)

    [row] = logger.all()
    assert row.cache_status == CacheStatus.MISS
    assert row.error == "RuntimeError: no key"
    assert row.embed_tokens == 0 and row.cost_usd == 0
    assert provider.calls == 0
