"""Amortized compression cost (milestone 6 follow-up, option 3).

Eval items are one-shot requests, so they pay the whole cost of summarizing /
embedding their older history at once. The runner also replays the compressor
turn by turn (rolling stores shared, no answer calls) and reports the average
compressor overhead per request of the conversation alongside the one-shot view.
"""

from pathlib import Path

import pytest
from test_milestone6 import QUERY, FakeProvider, KeywordEmbedder, long_history, settings_for

from gpt_efficient.compressor import Compressor, replay_overhead
from gpt_efficient.config import EvalConfig, JudgeConfig
from gpt_efficient.evals.dataset import EvalItem
from gpt_efficient.evals.judge import Judge
from gpt_efficient.evals.report import summarize, write_report
from gpt_efficient.evals.runner import Experiment, run_eval
from gpt_efficient.schemas import Completion, Message


def one_shot(strategy: str, tmp_path: Path, history: list[Message]):
    s = settings_for(tmp_path, strategy)
    return Compressor(s, {"gemini": FakeProvider()}, KeywordEmbedder()).compress(QUERY, history)


def test_replay_counts_every_request_and_summarizes_incrementally(tmp_path: Path) -> None:
    history = long_history(n=20)
    s = settings_for(tmp_path, "summary")
    provider = FakeProvider()
    overhead = replay_overhead(s, {"gemini": provider}, None, QUERY, history)

    assert overhead.requests == 21  # one per user turn, plus the final query
    calls = provider.summary_calls()
    assert calls, "some turns must have crossed the trigger"
    # every incremental call summarizes exactly the one exchange that newly aged out,
    # except the first, which covers everything older than the window at that point
    for _, msgs in calls[1:]:
        assert msgs[-1].content.split("NEW EXCHANGES:")[1].count("User:") == 1
    once = one_shot("summary", tmp_path, history)
    assert overhead.summary_tokens / overhead.requests < once.summary_tokens


def test_replay_embeds_each_exchange_once(tmp_path: Path) -> None:
    history = long_history(n=20)
    s = settings_for(tmp_path, "retrieval")
    overhead = replay_overhead(s, {"gemini": FakeProvider()}, KeywordEmbedder(), QUERY, history)
    once = one_shot("retrieval", tmp_path, history)
    assert overhead.summary_tokens == 0
    assert 0 < overhead.embed_tokens / overhead.requests < once.embed_tokens


class Judge10:
    name = "j"

    def complete(self, messages, max_tokens, model, temperature=None) -> Completion:
        return Completion(text='{"score": 10}', tokens_in=1, tokens_out=1, cost_usd=0.0,
                          latency_ms=0.0, model=model)  # fmt: skip


def run(tmp_path: Path, amortize: bool = True, items=None):
    base = settings_for(tmp_path, "none").model_copy(
        update={"judge": JudgeConfig(model="j"),
                "eval": EvalConfig(max_retries=0, amortize_compression=amortize)}  # fmt: skip
    )
    items = items or [
        EvalItem(id="c1", query=QUERY, reference="QX-481", difficulty="medium",
                 category="conversation", history=long_history(), tags=["needle:early"])  # fmt: skip
    ]
    exps = [
        Experiment(name=s, overrides={"compressor": {"strategy": s}})
        for s in ("none", "truncate", "summary", "summary+retrieval")
    ]
    results = run_eval(base, exps, items, tmp_path / "run",
                       make_providers=lambda s: {"gemini": FakeProvider()},
                       make_embedder=lambda s: KeywordEmbedder(),
                       judge=Judge(base, Judge10()), sleep=lambda s: None)  # fmt: skip
    return base, results


def test_runner_reports_amortized_alongside_one_shot(tmp_path: Path) -> None:
    _, results = run(tmp_path)
    by = {r.experiment: r for r in results}
    # no compressor overhead to amortize: amortized == one-shot
    for name in ("none", "truncate"):
        assert by[name].amortized_tokens == pytest.approx(by[name].total_tokens)
        assert by[name].amortized_cost_usd == pytest.approx(by[name].cost_usd)
    # summarizing strategies: overhead spread over the conversation's requests
    for name in ("summary", "summary+retrieval"):
        r = by[name]
        assert r.amortized_tokens < r.total_tokens
        assert r.amortized_cost_usd < r.cost_usd
        assert r.amortized_tokens > r.tokens_in + r.tokens_out  # overhead is not zero


def test_amortization_can_be_disabled(tmp_path: Path) -> None:
    _, results = run(tmp_path, amortize=False)
    for r in results:
        assert r.amortized_tokens == pytest.approx(r.total_tokens)
        assert r.amortized_cost_usd == pytest.approx(r.cost_usd)


def test_report_shows_both_views(tmp_path: Path) -> None:
    base, results = run(tmp_path)
    paths = write_report(results, base, tmp_path / "run")
    assert paths["frontier_tokens_amortized"].exists()
    assert paths["frontier_cost_amortized"].exists()
    md = paths["report"].read_text()
    assert "amortized" in md
    s = {x.experiment: x for x in summarize(results, low_quality=0.5)}
    assert s["summary"].mean_tokens_amortized < s["summary"].mean_tokens
    assert s["none"].mean_tokens_amortized == pytest.approx(s["none"].mean_tokens)


def test_single_turn_runs_have_no_amortized_plots(tmp_path: Path) -> None:
    item = EvalItem(id="s1", query="hi", reference="hello", difficulty="easy", category="factual")
    base, results = run(tmp_path, items=[item])
    paths = write_report(results, base, tmp_path / "run")
    assert "frontier_tokens_amortized" not in paths
