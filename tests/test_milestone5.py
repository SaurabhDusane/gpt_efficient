"""Milestone 5: eval harness. Report generated over a small dataset.

System provider, embedder and judge are all fakes, so the whole harness runs
offline. Token counts depend on the model and the judge's score depends on
which model wrote the answer, so every summary number is checkable by hand.
"""

import json
from pathlib import Path

import pytest

from gpt_efficient.cache import estimate_tokens
from gpt_efficient.config import (
    CacheConfig,
    EvalConfig,
    JudgeConfig,
    ModelPrice,
    RouterConfig,
    Settings,
    TierTarget,
)
from gpt_efficient.evals.dataset import EvalItem, load_dataset
from gpt_efficient.evals.judge import (
    Judge,
    JudgeVerdict,
    build_judge_messages,
    exact_match,
    parse_verdict,
    quality_from_score,
)
from gpt_efficient.evals.report import summarize, write_report
from gpt_efficient.evals.runner import (
    Experiment,
    ItemResult,
    apply_overrides,
    load_experiments,
    run_eval,
)
from gpt_efficient.schemas import CacheStatus, Completion, Message, Tier
from gpt_efficient.trace import TraceLogger

REPO = Path(__file__).resolve().parents[1]

MODELS = {
    Tier.LOCAL: "lite",
    Tier.MID: "flash",
    Tier.FRONTIER: "pro",
}
# model -> (tokens_in, tokens_out) the fake system provider reports
TOKENS = {"lite": (10, 10), "flash": (20, 40), "pro": (30, 200)}
# model that wrote the answer -> judge score (1-10)
SCORES = {"lite": 6, "flash": 8, "pro": 10}

ITEMS = [
    EvalItem(id="q1", query="What is the capital of France?", reference="Paris.",
             difficulty="easy", category="factual", exact="Paris"),
    EvalItem(id="q2", query="What's the capital city of France?", reference="Paris.",
             difficulty="easy", category="factual", exact="Paris", paraphrase_of="q1"),
    EvalItem(id="q3", query="Prove step by step that the integral of x^2 from 0 to 1 equals 1/3.",
             reference="Antiderivative x^3/3; evaluate 1/3 - 0 = 1/3.",
             difficulty="hard", category="math"),
    EvalItem(id="q4", query="Compare REST and GraphQL.", reference="Endpoints vs single schema...",
             difficulty="medium", category="code"),
]  # fmt: skip


class FakeSystem:
    """The system under test: answers mention Paris and name the model that wrote them."""

    name = "gemini"

    def __init__(self, fail_times: int = 0) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def complete(self, messages, max_tokens, model, temperature=None) -> Completion:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("429 rate limited")
        tin, tout = TOKENS[model]
        return Completion(
            text=f"[{model}] Paris is the answer.",
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=(tin * 1.0 + tout * 2.0) / 1e6,
            latency_ms=1.0,
            model=model,
        )


class FakeJudge:
    """Scores by which model wrote the candidate; records what it was shown."""

    name = "judge"

    def __init__(self, reply: str | None = None) -> None:
        self.seen: list[list[Message]] = []
        self.temperatures: list[float | None] = []
        self.reply = reply

    def complete(self, messages, max_tokens, model, temperature=None) -> Completion:
        self.seen.append(messages)
        self.temperatures.append(temperature)
        text = self.reply
        if text is None:
            candidate = messages[-1].content.split("CANDIDATE ANSWER")[-1]
            score = next(s for m, s in SCORES.items() if f"[{m}]" in candidate)
            text = json.dumps({"score": score, "rationale": "fake"})
        return Completion(text=text, tokens_in=100, tokens_out=5, cost_usd=0.001,
                          latency_ms=1.0, model=model)  # fmt: skip


class FakeEmbedder:
    name = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        # q1 and its paraphrase q2 share a vector; everything else is orthogonal
        vecs = {ITEMS[0].query: [1, 0, 0, 0], ITEMS[1].query: [1, 0, 0, 0],
                ITEMS[2].query: [0, 1, 0, 0], ITEMS[3].query: [0, 0, 1, 0]}  # fmt: skip
        return [[float(x) for x in vecs[t]] for t in texts]


@pytest.fixture
def base(tmp_path: Path) -> Settings:
    return Settings(
        default_provider="gemini",
        default_tier=Tier.MID,
        tier_modes={"three": [Tier.LOCAL, Tier.MID, Tier.FRONTIER]},
        tiers={t: TierTarget(model=m) for t, m in MODELS.items()},
        pricing={m: ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0) for m in MODELS.values()},
        embedding_price_per_mtok=0.0,
        judge=JudgeConfig(provider="judge", model="judge-model", max_tokens=512, temperature=0.0),
        eval=EvalConfig(max_retries=0, retry_backoff_s=0.0, low_quality=0.5),
        trace_db=tmp_path / "unused.db",
    )


EXPERIMENTS = [
    Experiment(name="fixed-mid", overrides={"router": {"type": "fixed"}, "cache": {"enabled": False}}),
    Experiment(name="heuristic+cache", overrides={"router": {"type": "heuristic"}, "cache": {"enabled": True}}),
]


def run(base: Settings, out: Path, system: FakeSystem | None = None, judge: FakeJudge | None = None,
        experiments: list[Experiment] = EXPERIMENTS, items: list[EvalItem] = ITEMS) -> list[ItemResult]:  # fmt: skip
    system = system or FakeSystem()
    return run_eval(
        base,
        experiments,
        items,
        out,
        make_providers=lambda s: {"gemini": system},
        make_embedder=lambda s: FakeEmbedder(),
        judge=Judge(base, judge or FakeJudge()),
        sleep=lambda s: None,
    )


# --- the milestone test ------------------------------------------------------


def test_report_generated_over_small_dataset(base: Settings, tmp_path: Path) -> None:
    out = tmp_path / "run"
    results = run(base, out)
    paths = write_report(results, base, out)

    for key in ("report", "summary_csv", "results", "frontier_tokens", "frontier_cost"):
        assert paths[key].exists() and paths[key].stat().st_size > 0, key
    md = paths["report"].read_text()
    assert "fixed-mid" in md and "heuristic+cache" in md
    assert "quality / 1k tokens" in md and "judge-model" in md
    csv_lines = paths["summary_csv"].read_text().strip().splitlines()
    assert len(csv_lines) == 1 + len(EXPERIMENTS)
    assert len(paths["results"].read_text().strip().splitlines()) == len(ITEMS) * len(EXPERIMENTS)

    fixed, routed = summarize(results, low_quality=0.5)
    # fixed-mid: every item on flash (20 in / 40 out), judged 8/10 -> (8-1)/9
    assert fixed.experiment == "fixed-mid"
    assert (fixed.n, fixed.answered, fixed.errors, fixed.cache_hits) == (4, 4, 0, 0)
    assert fixed.mean_quality == pytest.approx(7 / 9)
    assert fixed.mean_tokens == pytest.approx(60)
    assert fixed.mean_cost_usd == pytest.approx((20 + 80) / 1e6)
    assert fixed.q_per_1k_tokens == pytest.approx((7 / 9) / 0.060)
    assert fixed.q_per_usd == pytest.approx((7 / 9) / 1e-4)
    assert fixed.tier_mix == {"mid": 4}
    # heuristic+cache: q1 lite, q2 cache hit on q1's lite answer, q3 pro, q4 mid
    assert routed.tier_mix == {"local": 2, "frontier": 1, "mid": 1}
    assert routed.cache_hits == 1
    assert routed.mean_quality == pytest.approx(((6 - 1) + (6 - 1) + (10 - 1) + (8 - 1)) / 9 / 4)
    # the cache hit spent no LLM tokens; every lookup still embedded its query
    embed = sum(estimate_tokens(i.query, base.embedding_chars_per_token) for i in ITEMS)
    assert routed.mean_tokens == pytest.approx((20 + 0 + 230 + 60 + embed) / 4)
    # judge cost is reported separately, never charged to the system
    assert fixed.judge_cost_usd == pytest.approx(0.004)
    assert fixed.mean_cost_usd < 0.001
    # breakdown by difficulty
    assert set(routed.by_difficulty) == {"easy", "medium", "hard"}
    assert routed.by_difficulty["hard"].mean_quality == pytest.approx(1.0)


# --- runner ------------------------------------------------------------------------


def test_each_experiment_gets_its_own_trace_and_cache_db(base: Settings, tmp_path: Path) -> None:
    out = tmp_path / "run"
    results = run(base, out)
    for exp in EXPERIMENTS:
        rows = TraceLogger(out / exp.name / "traces.db").all()
        assert len(rows) == len(ITEMS)  # one trace row per request
    assert (out / "heuristic+cache" / "cache.db").exists()
    r = next(r for r in results if r.experiment == "heuristic+cache" and r.item_id == "q2")
    assert r.cache_status == CacheStatus.HIT and r.cache_sim == pytest.approx(1.0)


def test_judge_is_blind_and_deterministic(base: Settings, tmp_path: Path) -> None:
    judge = FakeJudge()
    run(base, tmp_path / "run", judge=judge)
    assert set(judge.temperatures) == {0.0}
    shown = " ".join(m.content for msgs in judge.seen for m in msgs)
    for leak in ("fixed-mid", "heuristic", "tier", "cache"):
        assert leak not in shown


def test_transient_errors_are_retried(base: Settings, tmp_path: Path) -> None:
    retrying = base.model_copy(update={"eval": base.eval.model_copy(update={"max_retries": 1})})
    system = FakeSystem(fail_times=1)
    results = run(retrying, tmp_path / "run", system=system, experiments=EXPERIMENTS[:1], items=ITEMS[:1])
    assert results[0].error is None and results[0].quality is not None
    # both attempts were real requests, so both logged a trace row
    assert len(TraceLogger(tmp_path / "run" / "fixed-mid" / "traces.db").all()) == 2


def test_failed_items_are_excluded_from_quality(base: Settings, tmp_path: Path) -> None:
    system = FakeSystem(fail_times=1)  # first item fails, retries disabled
    results = run(base, tmp_path / "run", system=system, experiments=EXPERIMENTS[:1])
    [s] = summarize(results, low_quality=0.5)
    assert (s.errors, s.answered, s.judged) == (1, 3, 3)
    assert s.mean_quality == pytest.approx(7 / 9)
    failed = next(r for r in results if r.error)
    assert "429" in failed.error and failed.judge_score is None


def test_unparseable_verdict_is_recorded_not_scored(base: Settings, tmp_path: Path) -> None:
    results = run(base, tmp_path / "run", judge=FakeJudge(reply="looks good to me"),
                  experiments=EXPERIMENTS[:1], items=ITEMS[:1])  # fmt: skip
    [r] = results
    assert r.judge_score is None and r.quality is None and r.judge_error
    [s] = summarize(results, low_quality=0.5)
    assert s.judged == 0 and s.mean_quality is None


def test_wrong_cache_hits_are_counted(base: Settings, tmp_path: Path) -> None:
    results = run(base, tmp_path / "run")
    # the cache hit on q2 is lite's answer, judged 6 -> quality 5/9 ~ 0.556
    assert summarize(results, low_quality=0.5)[1].wrong_cache_hits == 0
    assert summarize(results, low_quality=0.6)[1].wrong_cache_hits == 1


def test_exact_match_is_a_judge_sanity_check(base: Settings, tmp_path: Path) -> None:
    results = run(base, tmp_path / "run", experiments=EXPERIMENTS[:1])
    [s] = summarize(results, low_quality=0.5)
    assert s.exact_items == 2 and s.exact_acc == pytest.approx(1.0)
    assert s.judge_exact_disagreements == 0


def test_overrides_are_deep_merged(base: Settings) -> None:
    s = apply_overrides(base, {"cache": {"enabled": True}, "router": {"type": "heuristic"}})
    assert s.cache.enabled and s.router.type == "heuristic"
    assert s.cache.threshold == base.cache.threshold  # untouched siblings survive
    assert s.tiers == base.tiers


# --- judge ------------------------------------------------------------------------


def test_judge_prompt_contains_question_reference_and_answer() -> None:
    msgs = build_judge_messages(ITEMS[0], "It's Paris.")
    assert msgs[0].role == "system" and "1" in msgs[0].content and "10" in msgs[0].content
    user = msgs[-1].content
    assert ITEMS[0].query in user and ITEMS[0].reference in user and "It's Paris." in user


@pytest.mark.parametrize(
    "text,score",
    [
        ('{"score": 7, "rationale": "ok"}', 7),
        ('```json\n{"score": 10, "rationale": "perfect"}\n```', 10),
        ('Here you go: {"rationale": "meh", "score": 3}', 3),
    ],
)
def test_parse_verdict(text: str, score: int) -> None:
    assert parse_verdict(text).score == score


@pytest.mark.parametrize("text", ["", "no json here", '{"score": 11, "rationale": "x"}', '{"score": 0}'])
def test_parse_verdict_rejects_bad_output(text: str) -> None:
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_quality_from_score() -> None:
    assert quality_from_score(1) == 0.0
    assert quality_from_score(10) == 1.0
    assert JudgeVerdict(score=5, rationale="").score == 5


@pytest.mark.parametrize(
    "answer,expected,ok",
    [
        ("The capital is Paris.", "Paris", True),
        ("Paris, France", "paris", True),
        ("Lyon", "Paris", False),
        ("Comparison shopping", "paris", False),
        ("There are 206 bones.", "206", True),
        ("about 2060", "206", False),
        ("It is 1,000 meters", "1000", True),
        ("The ball costs $0.05.", "0.05", True),
        ("x = 2.50 hours", "2.5", True),
        ("-40 degrees", "-40", True),
        ("The probability is 671/1296 ≈ 0.518", "671/1296", True),
        ("Binary search is O(log n).", "O(log n)", True),
    ],
)
def test_exact_match(answer: str, expected: str, ok: bool) -> None:
    assert exact_match(answer, expected) is ok


# --- dataset / experiments files -------------------------------------------------------


def test_load_dataset_validates(tmp_path: Path) -> None:
    def write(lines: list[dict]) -> Path:
        p = tmp_path / "d.jsonl"
        p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        return p

    item = {"id": "a", "query": "q", "reference": "r", "difficulty": "easy", "category": "factual"}
    assert [i.id for i in load_dataset(write([item]))] == ["a"]
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(write([item, item]))
    with pytest.raises(ValueError, match="paraphrase_of"):
        load_dataset(write([{**item, "paraphrase_of": "zzz"}]))
    with pytest.raises(ValueError, match="line 1"):
        load_dataset(write([{**item, "difficulty": "trivial"}]))


def test_seed_dataset_is_valid_and_balanced() -> None:
    items = load_dataset(REPO / "evals" / "seed.jsonl")
    assert len(items) >= 30
    assert {i.difficulty for i in items} == {"easy", "medium", "hard"}
    assert {i.category for i in items} == {"factual", "reasoning", "math", "code", "writing"}
    assert sum(i.paraphrase_of is not None for i in items) >= 3
    assert any(t.startswith("near-miss") for i in items for t in i.tags)
    assert sum(i.exact is not None for i in items) >= 10
    for i in items:  # every exact answer must be found in its own reference
        if i.exact:
            assert exact_match(i.reference, i.exact), i.id


def test_repo_experiments_apply_to_repo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    base = Settings()
    experiments = load_experiments(REPO / "evals" / "experiments.toml")
    names = [e.name for e in experiments]
    assert len(names) == len(set(names)) >= 4
    assert {"fixed-mid", "heuristic"} <= set(names)
    for e in experiments:
        apply_overrides(base, e.overrides)  # raises if an override is invalid


# --- provider temperature pass-through ---------------------------------------------------


def test_providers_pass_temperature_through(tmp_path: Path) -> None:
    from test_milestone1 import FakeAnthropicClient
    from test_milestone2 import FakeGeminiClient

    from gpt_efficient.providers.anthropic_provider import AnthropicProvider
    from gpt_efficient.providers.gemini_provider import GeminiProvider

    s = Settings(pricing={"m": ModelPrice(input_per_mtok=1, output_per_mtok=1)})
    msgs = [Message(role="user", content="hi")]

    gemini = FakeGeminiClient()
    GeminiProvider(s, client=gemini).complete(msgs, max_tokens=8, model="m", temperature=0.0)
    GeminiProvider(s, client=gemini).complete(msgs, max_tokens=8, model="m")
    first, second = gemini.models.generate_calls
    assert first["config"].temperature == 0.0 and second["config"].temperature is None

    anthropic = FakeAnthropicClient()
    AnthropicProvider(s, client=anthropic).complete(msgs, max_tokens=8, model="m", temperature=0.0)
    AnthropicProvider(s, client=anthropic).complete(msgs, max_tokens=8, model="m")
    first, second = anthropic.messages.calls
    assert first["temperature"] == 0.0 and "temperature" not in second


def test_trace_logger_get(tmp_path: Path, base: Settings) -> None:
    from gpt_efficient.engine import Engine

    logger = TraceLogger(tmp_path / "t.db")
    resp = Engine(base.model_copy(update={"router": RouterConfig(), "cache": CacheConfig()}),
                  {"gemini": FakeSystem()}, logger).ask("hi")  # fmt: skip
    assert logger.get(resp.trace_id).id == resp.trace_id
    assert logger.get("nope") is None


# --- CLI end-to-end (offline) --------------------------------------------------------


def test_gpte_eval_fake_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    from gpt_efficient import cli

    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    out = tmp_path / "evalrun"
    monkeypatch.setattr(
        "sys.argv",
        ["gpte", "eval", "--fake", "--limit", "6", "--only", "fixed-mid,heuristic+cache",
         "--dataset", str(REPO / "evals" / "seed.jsonl"),
         "--experiments", str(REPO / "evals" / "experiments.toml"), "--out", str(out)],
    )  # fmt: skip
    cli.main()

    md = (out / "report.md").read_text()
    assert "FAKE" in md and "fixed-mid" in md and "heuristic+cache" in md
    assert len((out / "results.jsonl").read_text().splitlines()) == 12
    assert "report.md" in capsys.readouterr().out


def test_gpte_eval_rejects_unknown_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gpt_efficient import cli

    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    monkeypatch.setattr("sys.argv", ["gpte", "eval", "--fake", "--only", "nope",
                                     "--experiments", str(REPO / "evals" / "experiments.toml"),
                                     "--out", str(tmp_path / "x")])  # fmt: skip
    with pytest.raises(SystemExit):
        cli.main()
