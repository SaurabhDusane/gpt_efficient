"""Milestone 1: one query end-to-end, one logged trace row.

The Anthropic client is faked, so no API key or network is needed.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gpt_efficient.config import Settings, TierTarget
from gpt_efficient.engine import Engine
from gpt_efficient.providers.anthropic_provider import AnthropicProvider
from gpt_efficient.providers.base import LLMProvider
from gpt_efficient.schemas import CacheStatus, Message, Tier
from gpt_efficient.trace import TraceLogger, query_hash


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=kwargs["model"],
            content=[SimpleNamespace(type="text", text="Paris is the capital of France.")],
            usage=SimpleNamespace(
                input_tokens=20,
                output_tokens=10,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )


class FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = FakeMessages()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        default_tier=Tier.MID,
        tiers={Tier.MID: TierTarget(provider="anthropic", model="claude-haiku-4-5")},
        pricing={"claude-haiku-4-5": {"input_per_mtok": 1.0, "output_per_mtok": 5.0}},
        trace_db=tmp_path / "traces.db",
        system_prompt="Be concise.",
    )


def test_adapter_satisfies_interface(settings: Settings) -> None:
    provider = AnthropicProvider(settings, client=FakeAnthropicClient())
    assert isinstance(provider, LLMProvider)


def test_adapter_maps_completion(settings: Settings) -> None:
    client = FakeAnthropicClient()
    provider = AnthropicProvider(settings, client=client)
    completion = provider.complete(
        [
            Message(role="system", content="Be concise."),
            Message(role="user", content="Capital of France?"),
        ],
        max_tokens=64,
        model="claude-haiku-4-5",
    )
    assert completion.text == "Paris is the capital of France."
    assert (completion.tokens_in, completion.tokens_out) == (20, 10)
    # 20 * $1/M + 10 * $5/M
    assert completion.cost_usd == pytest.approx(20e-6 + 50e-6)
    assert completion.model == "claude-haiku-4-5"
    # system prompt goes to the top-level `system` param, not the messages list
    call = client.messages.calls[0]
    assert call["system"] == "Be concise."
    assert call["messages"] == [{"role": "user", "content": "Capital of France?"}]


def test_one_query_end_to_end_logs_one_trace(settings: Settings) -> None:
    client = FakeAnthropicClient()
    logger = TraceLogger(settings.trace_db)
    engine = Engine(
        settings,
        providers={"anthropic": AnthropicProvider(settings, client=client)},
        logger=logger,
    )

    response = engine.ask("Capital of France?")

    assert response.text == "Paris is the capital of France."
    rows = logger.all()
    assert len(rows) == 1
    row = rows[0]
    assert row.id == response.trace_id
    assert row.query_hash == query_hash("Capital of France?")
    assert row.cache_status == CacheStatus.DISABLED
    assert row.cache_sim is None
    assert row.tier == Tier.MID
    assert row.provider == "anthropic"
    assert row.model == "claude-haiku-4-5"
    assert (row.tokens_in, row.tokens_out) == (20, 10)
    assert row.cost_usd == pytest.approx(70e-6)
    assert row.latency_ms >= 0
    assert row.compressed is False and row.tokens_saved == 0
    assert row.escalated is False
    assert row.response_len == len("Paris is the capital of France.")


def test_failed_request_still_logs_one_trace(settings: Settings) -> None:
    class Boom(FakeMessages):
        def create(self, **kwargs):
            raise RuntimeError("provider down")

    client = FakeAnthropicClient()
    client.messages = Boom()
    logger = TraceLogger(settings.trace_db)
    engine = Engine(
        settings,
        providers={"anthropic": AnthropicProvider(settings, client=client)},
        logger=logger,
    )

    with pytest.raises(RuntimeError):
        engine.ask("Capital of France?")

    rows = logger.all()
    assert len(rows) == 1
    assert rows[0].error == "RuntimeError: provider down"
    assert rows[0].tokens_out == 0
