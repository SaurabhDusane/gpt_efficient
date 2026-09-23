"""Milestone 6: context compressor. Long history -> fewer tokens, quality held.

Offline, "quality held" is checked by its precondition: the fact the final
question depends on (the needle, planted early in a long history) still
reaches the model. Real answer quality is measured by the eval harness on
evals/conversations.jsonl.
"""

import math
from pathlib import Path

import pytest

from gpt_efficient.compressor import (
    SUMMARY_PROMPT,
    Compressor,
    SummaryStore,
    estimate_messages_tokens,
    split_exchanges,
    top_k_indices,
)
from gpt_efficient.config import CompressorConfig, ModelPrice, Settings, TierTarget
from gpt_efficient.engine import Engine
from gpt_efficient.evals.dataset import load_dataset
from gpt_efficient.evals.runner import ItemResult, apply_overrides, load_experiments
from gpt_efficient.schemas import CacheStatus, Completion, Message, Tier
from gpt_efficient.trace import TraceLogger

REPO = Path(__file__).resolve().parents[1]
MODELS = {Tier.LOCAL: "lite", Tier.MID: "flash", Tier.FRONTIER: "pro"}
NEEDLE = "By the way, my flight number is QX-481. Please remember it."
QUERY = "What was my flight number again?"
FILLER = "Here is a long and detailed explanation of the topic, " * 7


def exchange(user: str, assistant: str) -> list[Message]:
    return [Message(role="user", content=user), Message(role="assistant", content=assistant)]


def long_history(n: int = 20, needle_at: int = 1) -> list[Message]:
    msgs: list[Message] = []
    for i in range(n):
        if i == needle_at:
            msgs += exchange(NEEDLE, "Noted: your flight number is QX-481.")
        else:
            msgs += exchange(f"Tell me about topic {i}.", f"Topic {i}: {FILLER}")
    return msgs


class FakeProvider:
    """Answers and summaries; token counts track the text it was actually sent."""

    name = "gemini"

    def __init__(self, fail_summary: bool = False) -> None:
        self.calls: list[tuple[str, list[Message]]] = []
        self.fail_summary = fail_summary

    def complete(self, messages, max_tokens, model, temperature=None) -> Completion:
        self.calls.append((model, messages))
        tokens_in = estimate_messages_tokens(messages, 4.0)
        if messages[0].content == SUMMARY_PROMPT:
            if self.fail_summary:
                raise RuntimeError("summarizer down")
            # keep every user line, so planted facts survive the summary
            body = messages[-1].content
            users = [ln for ln in body.splitlines() if ln.startswith("User:")]
            text = "SUMMARY: " + " | ".join(users)
        else:
            text = "answer"
        return Completion(text=text, tokens_in=tokens_in, tokens_out=len(text) // 4,
                          cost_usd=tokens_in * 1e-6, latency_ms=1.0, model=model)  # fmt: skip

    def answer_calls(self) -> list[tuple[str, list[Message]]]:
        return [c for c in self.calls if c[1][0].content != SUMMARY_PROMPT]

    def summary_calls(self) -> list[tuple[str, list[Message]]]:
        return [c for c in self.calls if c[1][0].content == SUMMARY_PROMPT]


class KeywordEmbedder:
    """Vectors over two keywords: the needle exchange and the query share 'flight'."""

    name = "kw"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts += texts
        out = []
        for t in texts:
            v = [t.lower().count("flight"), t.lower().count("topic"), 0.1]
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v])
        return out


def settings_for(tmp_path: Path, strategy: str, **cfg) -> Settings:
    return Settings(
        default_provider="gemini",
        default_tier=Tier.MID,
        tier_modes={"three": [Tier.LOCAL, Tier.MID, Tier.FRONTIER]},
        tiers={t: TierTarget(model=m) for t, m in MODELS.items()},
        pricing={m: ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0) for m in MODELS.values()},
        embedding_price_per_mtok=1.0,
        compressor=CompressorConfig(strategy=strategy, **cfg),
        trace_db=tmp_path / f"{strategy}.db",
    )


def ask(tmp_path: Path, strategy: str, history: list[Message] | None = None, **cfg):
    s = settings_for(tmp_path, strategy, **cfg)
    provider, embedder, logger = FakeProvider(), KeywordEmbedder(), TraceLogger(s.trace_db)
    Engine(s, {"gemini": provider}, logger, embedder=embedder).ask(QUERY, history or long_history())
    [row] = logger.all()
    return row, provider, embedder


def sent_text(provider: FakeProvider) -> str:
    [(_, msgs)] = provider.answer_calls()
    return "\n".join(m.content for m in msgs)


# --- the milestone test ------------------------------------------------------


@pytest.mark.parametrize("strategy", ["truncate", "summary", "retrieval", "summary+retrieval"])
def test_long_history_fewer_tokens_quality_held(tmp_path: Path, strategy: str) -> None:
    base, base_provider, _ = ask(tmp_path, "none")
    row, provider, _ = ask(tmp_path, strategy)

    assert base.compressed is False and base.tokens_saved == 0
    assert row.compressed is True
    assert row.tokens_in < base.tokens_in  # fewer tokens reached the answering model
    assert row.tokens_saved > 0
    # quality held (precondition): the needle still reaches the model...
    if strategy == "truncate":
        # ...except for plain truncation, the baseline the smart strategies must beat
        assert "QX-481" not in sent_text(provider)
    else:
        assert "QX-481" in sent_text(provider)


# --- behaviour --------------------------------------------------------------------


def test_short_history_is_left_alone(tmp_path: Path) -> None:
    short = long_history(n=3)
    row, provider, embedder = ask(tmp_path, "summary+retrieval", history=short)
    assert row.compressed is False and row.tokens_saved == 0 and row.summary_tokens == 0
    assert provider.summary_calls() == [] and embedder.texts == []


def test_recent_turns_kept_verbatim_in_order(tmp_path: Path) -> None:
    history = long_history()
    _, provider, _ = ask(tmp_path, "summary+retrieval", history=history, keep_recent_turns=4)
    [(_, msgs)] = provider.answer_calls()
    tail = msgs[-9:-1]  # 4 exchanges before the final query
    assert tail == history[-8:]
    assert msgs[-1] == Message(role="user", content=QUERY)
    # context blocks are system messages placed before the verbatim turns
    non_system = [m for m in msgs if m.role != "system"]
    assert [m.role for m in non_system] == ["user", "assistant"] * 4 + ["user"]


def test_tokens_saved_is_history_estimate_delta(tmp_path: Path) -> None:
    history = long_history()
    row, provider, _ = ask(tmp_path, "truncate", history=history, keep_recent_turns=4)
    kept = history[-8:]
    assert row.tokens_saved == estimate_messages_tokens(history, 4.0) - estimate_messages_tokens(kept, 4.0)


def test_summary_is_written_by_budget_tier_and_costed(tmp_path: Path) -> None:
    row, provider, embedder = ask(tmp_path, "summary")
    [(model, _)] = provider.summary_calls()
    assert model == MODELS[Tier.LOCAL]  # the configured summary_tier (default: local)
    assert row.model == MODELS[Tier.MID]  # the answer still comes from the routed tier
    assert row.summary_tokens > 0
    assert embedder.texts == [] and row.embed_tokens == 0  # summary-only needs no embeddings
    # cost = answer + summary (both priced at $1/M in, $1/M out here)
    answer_cost = row.tokens_in * 1e-6
    assert row.cost_usd > answer_cost


def test_retrieval_logs_embedding_tokens(tmp_path: Path) -> None:
    row, provider, embedder = ask(tmp_path, "retrieval", retrieve_k=1)
    assert row.embed_tokens > 0 and row.summary_tokens == 0
    assert provider.summary_calls() == []
    # only the top-1 older exchange (the needle) comes back verbatim
    text = sent_text(provider)
    assert "QX-481" in text and "Topic 5:" not in text


def test_rolling_summary_only_summarizes_new_exchanges(tmp_path: Path) -> None:
    s = settings_for(tmp_path, "summary", keep_recent_turns=4)
    provider, store = FakeProvider(), SummaryStore()
    comp = Compressor(s, {"gemini": provider}, embedder=None, store=store)
    history = long_history(n=20)
    comp.compress(QUERY, history)
    comp.compress(QUERY, history + exchange("Tell me about topic 99.", f"Topic 99: {FILLER}"))

    first, second = provider.summary_calls()
    assert "QX-481" in first[1][-1].content
    # the second call builds on the stored summary and sends only the one new exchange
    assert "SUMMARY:" in second[1][-1].content
    assert second[1][-1].content.split("NEW EXCHANGES:")[1].count("User:") == 1


def test_summary_failure_still_logs_one_row(tmp_path: Path) -> None:
    s = settings_for(tmp_path, "summary")
    logger = TraceLogger(s.trace_db)
    engine = Engine(s, {"gemini": FakeProvider(fail_summary=True)}, logger)
    with pytest.raises(RuntimeError):
        engine.ask(QUERY, long_history())
    [row] = logger.all()
    assert row.error == "RuntimeError: summarizer down"


def test_retrieval_requires_an_embedder(tmp_path: Path) -> None:
    s = settings_for(tmp_path, "retrieval")
    with pytest.raises(ValueError, match="embedder"):
        Engine(s, {"gemini": FakeProvider()}, TraceLogger(s.trace_db))


def test_history_still_bypasses_cache(tmp_path: Path) -> None:
    from gpt_efficient.cache import SemanticCache
    from gpt_efficient.config import CacheConfig

    s = settings_for(tmp_path, "summary").model_copy(
        update={"cache": CacheConfig(enabled=True, db=tmp_path / "c.db")}
    )
    logger = TraceLogger(s.trace_db)
    Engine(s, {"gemini": FakeProvider()}, logger, embedder=KeywordEmbedder(),
           cache=SemanticCache(s.cache.db)).ask(QUERY, long_history())  # fmt: skip
    [row] = logger.all()
    assert row.cache_status == CacheStatus.BYPASS and row.compressed


# --- pure helpers -----------------------------------------------------------------------


def test_split_exchanges() -> None:
    h = exchange("a", "b") + exchange("c", "d") + [Message(role="user", content="e")]
    assert [[m.content for m in ex] for ex in split_exchanges(h)] == [["a", "b"], ["c", "d"], ["e"]]
    lead = [Message(role="assistant", content="hi")] + exchange("a", "b")
    assert [[m.content for m in ex] for ex in split_exchanges(lead)] == [["hi"], ["a", "b"]]


def test_top_k_indices_keeps_conversation_order() -> None:
    q = [1.0, 0.0]
    vecs = [[0.0, 1.0], [0.9, 0.1], [0.2, 0.8], [1.0, 0.0]]
    assert top_k_indices(q, vecs, 2) == [1, 3]
    assert top_k_indices(q, vecs, 10) == [0, 1, 2, 3]
    assert top_k_indices(q, [], 3) == []


def test_estimate_messages_tokens() -> None:
    assert estimate_messages_tokens([Message(role="user", content="a" * 9)], 4.0) == 3
    assert estimate_messages_tokens([], 4.0) == 0


def test_compressor_config_defaults() -> None:
    c = CompressorConfig()
    assert (c.strategy, c.trigger_tokens, c.keep_recent_turns, c.retrieve_k) == ("none", 1500, 4, 3)
    assert c.summary_tier == Tier.LOCAL


# --- eval integration -------------------------------------------------------------------


def test_item_result_total_tokens_includes_summary() -> None:
    r = ItemResult(experiment="e", item_id="i", difficulty="easy", category="conversation",
                   tier=Tier.MID, provider="p", model="m", cache_status=CacheStatus.DISABLED,
                   tokens_in=10, tokens_out=5, embed_tokens=3, summary_tokens=7)  # fmt: skip
    assert r.total_tokens == 25


def test_conversations_dataset_is_valid_and_long(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    trigger = Settings().compressor.trigger_tokens
    items = load_dataset(REPO / "evals" / "conversations.jsonl")
    assert len(items) >= 10
    for item in items:
        roles = [m.role for m in item.history]
        assert roles == ["user", "assistant"] * (len(roles) // 2), item.id
        assert len(item.history) // 2 >= 10, item.id  # at least 10 exchanges
        assert estimate_messages_tokens(item.history, 4.0) > trigger, item.id
        assert item.category == "conversation"
    tags = {t for i in items for t in i.tags}
    assert {"needle:early", "needle:middle", "recent-only", "aggregate"} <= tags


def test_compression_experiments_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    base = Settings()
    exps = load_experiments(REPO / "evals" / "experiments_compression.toml")
    strategies = {apply_overrides(base, e.overrides).compressor.strategy for e in exps}
    assert strategies == {"none", "truncate", "summary", "retrieval", "summary+retrieval"}


def test_runner_sends_item_history(tmp_path: Path) -> None:
    from gpt_efficient.config import EvalConfig, JudgeConfig
    from gpt_efficient.evals.dataset import EvalItem
    from gpt_efficient.evals.judge import Judge
    from gpt_efficient.evals.report import summarize
    from gpt_efficient.evals.runner import Experiment, run_eval

    class Judge10:
        name = "j"

        def complete(self, messages, max_tokens, model, temperature=None) -> Completion:
            return Completion(text='{"score": 10}', tokens_in=1, tokens_out=1, cost_usd=0.0,
                              latency_ms=0.0, model=model)  # fmt: skip

    base = settings_for(tmp_path, "none").model_copy(
        update={"judge": JudgeConfig(model="j"), "eval": EvalConfig(max_retries=0)}
    )
    item = EvalItem(id="c1", query=QUERY, reference="QX-481", difficulty="medium",
                    category="conversation", history=long_history(), tags=["needle:early"])  # fmt: skip
    provider = FakeProvider()
    exps = [Experiment(name="none"), Experiment(name="summary", overrides={"compressor": {"strategy": "summary"}})]
    results = run_eval(base, exps, [item], tmp_path / "run", make_providers=lambda s: {"gemini": provider},
                       make_embedder=lambda s: KeywordEmbedder(), judge=Judge(base, Judge10()),
                       sleep=lambda s: None)  # fmt: skip
    none, summ = results
    assert (none.compressed, summ.compressed) == (False, True)
    assert summ.tokens_saved > 0 and summ.summary_tokens > 0
    assert summ.tokens_in < none.tokens_in
    s_none, s_summ = summarize(results, low_quality=0.5)
    assert (s_none.compressed, s_summ.compressed) == (0, 1)
    assert s_summ.mean_tokens_saved == pytest.approx(summ.tokens_saved)


def test_conversation_probes_sit_where_their_tags_say(monkeypatch: pytest.MonkeyPatch) -> None:
    """Needles must be outside the verbatim window; recent-only answers inside it."""
    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    keep = Settings().compressor.keep_recent_turns
    for item in load_dataset(REPO / "evals" / "conversations.jsonl"):
        if item.exact is None:
            continue
        exchanges = split_exchanges(item.history)
        recent = " ".join(m.content for ex in exchanges[-keep:] for m in ex)
        older = " ".join(m.content for ex in exchanges[:-keep] for m in ex)
        if "recent-only" in item.tags:
            assert item.exact in recent, item.id
        else:
            assert item.exact not in recent, item.id
            if "aggregate" not in item.tags:
                assert item.exact in older, item.id


def test_retrieval_embeds_each_exchange_once_per_conversation(tmp_path: Path) -> None:
    s = settings_for(tmp_path, "retrieval", keep_recent_turns=4)
    embedder = KeywordEmbedder()
    comp = Compressor(s, {"gemini": FakeProvider()}, embedder=embedder)
    history = long_history(n=20)
    first = comp.compress(QUERY, history)
    embedder.texts.clear()
    second = comp.compress(QUERY, history + exchange("Tell me about topic 99.", f"Topic 99: {FILLER}"))

    # 16 older exchanges + query the first time; then only the query and the one new older exchange
    assert len(embedder.texts) == 2
    assert 0 < second.embed_tokens < first.embed_tokens
