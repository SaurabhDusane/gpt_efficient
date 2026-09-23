"""Milestone 2: Gemini provider + embeddings behind provider-neutral interfaces.

The Gemini client is faked, so no API key or network is needed. The one live
test at the bottom runs only when GEMINI_API_KEY is set.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gpt_efficient.config import ModelPrice, Settings, TierTarget
from gpt_efficient.engine import Engine
from gpt_efficient.providers import build_embedder, build_providers
from gpt_efficient.providers.base import Embedder, LLMProvider
from gpt_efficient.providers.gemini_provider import GeminiEmbedder, GeminiProvider
from gpt_efficient.schemas import CacheStatus, Message, Tier
from gpt_efficient.trace import TraceLogger

REPLY = "Paris is the capital of France."


class FakeModels:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.generate_calls: list[dict] = []
        self.embed_calls: list[dict] = []

    def generate_content(self, *, model, contents, config=None):
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            text=REPLY,
            model_version=model,
            usage_metadata=SimpleNamespace(
                prompt_token_count=20,
                candidates_token_count=10,
                thoughts_token_count=30,
                cached_content_token_count=None,
            ),
        )

    def embed_content(self, *, model, contents, config=None):
        self.embed_calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1] * self.dim) for _ in contents]
        )


class FakeGeminiClient:
    def __init__(self, dim: int = 8) -> None:
        self.models = FakeModels(dim)


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
        tiers={
            Tier.LOCAL: TierTarget(model="gemini-2.5-flash-lite"),
            Tier.MID: TierTarget(model="gemini-2.5-flash"),
            Tier.FRONTIER: TierTarget(model="gemini-3.1-pro-preview"),
        },
        pricing={
            "gemini-2.5-flash-lite": ModelPrice(input_per_mtok=0.10, output_per_mtok=0.40),
            "gemini-2.5-flash": ModelPrice(input_per_mtok=0.30, output_per_mtok=2.50),
            "gemini-3.1-pro-preview": ModelPrice(
                input_per_mtok=2.00,
                output_per_mtok=12.00,
                long_context_threshold=200_000,
                input_per_mtok_long=4.00,
                output_per_mtok_long=18.00,
            ),
        },
        embedding_model="gemini-embedding-2",
        embedding_dim=8,
        embedding_price_per_mtok=0.20,
        trace_db=tmp_path / "traces.db",
        system_prompt="Be concise.",
    )


def test_gemini_adapter_satisfies_interface(settings: Settings) -> None:
    assert isinstance(GeminiProvider(settings, client=FakeGeminiClient()), LLMProvider)


def test_gemini_maps_completion(settings: Settings) -> None:
    client = FakeGeminiClient()
    provider = GeminiProvider(settings, client=client)
    completion = provider.complete(
        [
            Message(role="system", content="Be concise."),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
            Message(role="user", content="Capital of France?"),
        ],
        max_tokens=64,
        model="gemini-2.5-flash",
    )

    assert completion.text == REPLY
    assert completion.model == "gemini-2.5-flash"
    assert completion.tokens_in == 20
    # thinking tokens are billed as output, so they count toward tokens_out
    assert completion.tokens_out == 10 + 30
    assert completion.cost_usd == pytest.approx((20 * 0.30 + 40 * 2.50) / 1e6)
    assert completion.latency_ms >= 0

    call = client.models.generate_calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["config"].system_instruction == "Be concise."
    assert call["config"].max_output_tokens == 64
    # system message is lifted out; assistant maps to Gemini's "model" role
    assert [(c.role, c.parts[0].text) for c in call["contents"]] == [
        ("user", "Hi"),
        ("model", "Hello!"),
        ("user", "Capital of France?"),
    ]


def test_gemini_types_do_not_escape(settings: Settings) -> None:
    completion = GeminiProvider(settings, client=FakeGeminiClient()).complete(
        [Message(role="user", content="x")], max_tokens=8, model="gemini-2.5-flash"
    )
    vectors = GeminiEmbedder(settings, client=FakeGeminiClient()).embed(["x"])
    assert type(completion).__module__.startswith("gpt_efficient")
    assert all(type(v) is list and all(type(x) is float for x in v) for v in vectors)


def test_embedder_returns_vectors(settings: Settings) -> None:
    client = FakeGeminiClient(dim=8)
    embedder = GeminiEmbedder(settings, client=client)
    assert isinstance(embedder, Embedder)

    vectors = embedder.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)
    call = client.models.embed_calls[0]
    assert call["model"] == "gemini-embedding-2"
    # gemini-embedding-2 aggregates a bare list of strings into ONE embedding;
    # each text must be its own Content to get one vector per text.
    assert [c.parts[0].text for c in call["contents"]] == ["a", "b", "c"]
    assert all(len(c.parts) == 1 for c in call["contents"])
    assert call["config"].output_dimensionality == 8


def test_embedder_rejects_wrong_dimensionality(settings: Settings) -> None:
    embedder = GeminiEmbedder(settings, client=FakeGeminiClient(dim=5))
    with pytest.raises(ValueError, match="dimension"):
        embedder.embed(["a"])


def test_cost_uses_config_pricing(settings: Settings) -> None:
    # Same token counts, different configured rates -> different cost.
    assert settings.cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(2.80)
    settings.pricing["gemini-2.5-flash"] = ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0)
    assert settings.cost_usd("gemini-2.5-flash", 1_000_000, 1_000_000) == pytest.approx(2.00)
    # Long-context rates apply once the prompt exceeds the configured threshold.
    assert settings.cost_usd("gemini-3.1-pro-preview", 200_000, 0) == pytest.approx(0.40)
    assert settings.cost_usd("gemini-3.1-pro-preview", 200_001, 1_000_000) == pytest.approx(
        200_001 * 4.00 / 1e6 + 18.00
    )
    assert settings.embedding_cost_usd(1_000_000) == pytest.approx(0.20)
    with pytest.raises(ValueError, match="pricing"):
        settings.cost_usd("unpriced-model", 1, 1)


def test_tier_mode_reads_active_tiers_from_config(settings: Settings) -> None:
    assert settings.active_tiers == [Tier.LOCAL, Tier.MID, Tier.FRONTIER]
    two = settings.model_copy(update={"tier_mode": "two"})
    assert two.active_tiers == [Tier.LOCAL, Tier.MID]
    # the provider falls back to default_provider when a tier doesn't name one
    assert settings.target(Tier.MID).provider == "gemini"


def test_tier_mode_validation(settings: Settings) -> None:
    base = settings.model_dump()
    with pytest.raises(ValidationError, match="tier_mode"):
        Settings(**{**base, "tier_mode": "four"})
    with pytest.raises(ValidationError, match="default_tier"):
        Settings(**{**base, "tier_mode": "two", "default_tier": Tier.FRONTIER})


def test_engine_routes_to_gemini_and_logs_one_trace(settings: Settings) -> None:
    logger = TraceLogger(settings.trace_db)
    engine = Engine(
        settings,
        providers={"gemini": GeminiProvider(settings, client=FakeGeminiClient())},
        logger=logger,
    )

    engine.ask("Capital of France?")

    [row] = logger.all()
    assert (row.tier, row.provider, row.model) == (Tier.MID, "gemini", "gemini-2.5-flash")
    assert row.cache_status == CacheStatus.DISABLED
    assert (row.tokens_in, row.tokens_out) == (20, 40)
    assert row.cost_usd == pytest.approx((20 * 0.30 + 40 * 2.50) / 1e6)
    assert row.response_len == len(REPLY)
    assert row.error is None


def test_registry_builds_gemini_without_credentials(settings: Settings) -> None:
    assert isinstance(build_providers(settings)["gemini"], GeminiProvider)
    assert isinstance(build_embedder(settings), GeminiEmbedder)


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_live_gemini_round_trip(settings: Settings) -> None:
    completion = GeminiProvider(settings).complete(
        [Message(role="user", content="Reply with the single word: ok")],
        max_tokens=16,
        model=settings.target(Tier.LOCAL).model,
    )
    assert completion.tokens_in > 0
    assert completion.cost_usd > 0
    live = settings.model_copy(update={"embedding_dim": 768})
    [vector] = GeminiEmbedder(live).embed(["hello"])
    assert len(vector) == 768


def test_repo_config_toml_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_config = Path(__file__).resolve().parents[1] / "config.toml"
    monkeypatch.setenv("GPTE_CONFIG", str(repo_config))
    s = Settings()
    assert s.default_provider == "gemini"
    for tier in s.active_tiers:  # every active tier has a priced model
        s.cost_usd(s.target(tier).model, 1, 1)
    assert set(s.model_copy(update={"tier_mode": "two"}).active_tiers) < set(s.active_tiers)
