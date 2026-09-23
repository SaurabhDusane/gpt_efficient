"""Milestone 7: learned router, benchmarked against the heuristic on the harness.

Labelling, training and routing all run offline here: fakes stand in for the
answering models, the judge and the embedder.
"""

import json
import re
from pathlib import Path

import pytest

from gpt_efficient.cache import cache_namespace
from gpt_efficient.config import (
    EvalConfig,
    JudgeConfig,
    LearnedRouterConfig,
    ModelPrice,
    RouterConfig,
    Settings,
    TierTarget,
)
from gpt_efficient.engine import Engine
from gpt_efficient.evals.dataset import EvalItem, load_dataset
from gpt_efficient.evals.judge import Judge
from gpt_efficient.evals.report import summarize, write_report
from gpt_efficient.evals.runner import Experiment, apply_overrides, load_experiments, run_eval
from gpt_efficient.learned_router import (
    RouterModel,
    TrainQuery,
    label_from_scores,
    label_query,
    load_train_queries,
    train_router,
)
from gpt_efficient.router import LearnedRouter, build_router
from gpt_efficient.schemas import Completion, Message, Tier
from gpt_efficient.trace import TraceLogger

REPO = Path(__file__).resolve().parents[1]
MODELS = {Tier.LOCAL: "lite", Tier.MID: "flash", Tier.FRONTIER: "pro"}
THREE = [Tier.LOCAL, Tier.MID, Tier.FRONTIER]

# 3-d "embeddings": one axis per tier, so a query's tier is easy to control.
AXES = {"easy": [1.0, 0.0, 0.0], "medium": [0.0, 1.0, 0.0], "hard": [0.0, 0.0, 1.0]}


class AxisEmbedder:
    """Embeds by the first difficulty word in the text; 'vague' sits between axes."""

    name = "axis"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for t in texts:
            word = next((w for w in ("easy", "medium", "hard", "vague") if w in t.lower()), "easy")
            out.append([0.5, 0.48, 0.0] if word == "vague" else AXES[word])
        return out


def model_for(classes: list[Tier]) -> RouterModel:
    """A hand-built model: each class's weight vector points along its axis."""
    coef = [[8.0 if i == j else 0.0 for j in range(3)] for i in range(len(classes))]
    return RouterModel(classes=classes, coef=coef, intercept=[0.0] * len(classes),
                       embedding_model="axis", embedding_dim=3)  # fmt: skip


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    model_path = tmp_path / "router.json"
    model_path.write_text(model_for(THREE).model_dump_json())
    return Settings(
        default_provider="gemini",
        default_tier=Tier.MID,
        tier_modes={"two": [Tier.LOCAL, Tier.MID], "three": THREE},
        tiers={t: TierTarget(model=m) for t, m in MODELS.items()},
        pricing={m: ModelPrice(input_per_mtok=1.0, output_per_mtok=1.0) for m in [*MODELS.values(), "judge"]},
        embedding_model="axis",
        embedding_dim=3,
        embedding_price_per_mtok=1.0,
        router=RouterConfig(type="learned", learned=LearnedRouterConfig(model_path=model_path)),
        judge=JudgeConfig(model="judge", temperature=0.0),
        eval=EvalConfig(max_retries=0, amortize_compression=False),
        trace_db=tmp_path / "traces.db",
    )


class RecordingProvider:
    name = "gemini"

    def __init__(self) -> None:
        self.models: list[str] = []

    def complete(self, messages, max_tokens, model, temperature=None) -> Completion:
        self.models.append(model)
        return Completion(text=f"[{model}] answer", tokens_in=10, tokens_out=10,
                          cost_usd=1e-5, latency_ms=1.0, model=model)  # fmt: skip


# --- the milestone test ------------------------------------------------------


def test_frontier_plot_compares_both_routers(settings: Settings, tmp_path: Path) -> None:
    class ScoreByModel:
        name = "judge"

        def complete(self, messages, max_tokens, model, temperature=None) -> Completion:
            cand = messages[-1].content.split("CANDIDATE ANSWER")[-1]
            score = 10 if "[pro]" in cand else 8 if "[flash]" in cand else 6
            return Completion(text=json.dumps({"score": score}), tokens_in=1, tokens_out=1,
                              cost_usd=0.0, latency_ms=0.0, model=model)  # fmt: skip

    items = [
        EvalItem(id="e1", query="an easy one: capital of Peru?", reference="Lima",
                 difficulty="easy", category="factual"),
        EvalItem(id="m1", query="a medium one: compare two sorting methods", reference="...",
                 difficulty="medium", category="code"),
        EvalItem(id="h1", query="a hard one: prove there are infinitely many primes", reference="...",
                 difficulty="hard", category="math"),
    ]  # fmt: skip
    exps = [
        Experiment(name="heuristic", overrides={"router": {"type": "heuristic"}}),
        Experiment(name="learned", overrides={"router": {"type": "learned"}}),
    ]
    out = tmp_path / "run"
    results = run_eval(settings, exps, items, out,
                       make_providers=lambda s: {"gemini": RecordingProvider()},
                       make_embedder=lambda s: AxisEmbedder(),
                       judge=Judge(settings, ScoreByModel()), sleep=lambda s: None)  # fmt: skip
    paths = write_report(results, settings, out)

    heur, learned = summarize(results, low_quality=0.5)
    # both routers land on the frontier plots (quality and x-axis values present)
    for s in (heur, learned):
        assert s.mean_quality is not None and s.mean_tokens is not None and s.mean_cost_usd is not None
    assert paths["frontier_tokens"].stat().st_size > 0 and paths["frontier_cost"].stat().st_size > 0
    # the learned router follows its embedding: easy->local, medium->mid, hard->frontier
    assert learned.tier_mix == {"local": 1, "mid": 1, "frontier": 1}
    assert learned.mean_route_confidence is not None and learned.mean_route_confidence > 0.9
    assert heur.mean_route_confidence is None  # heuristic has no confidence
    md = paths["report"].read_text()
    assert "## Routing" in md and "learned" in md and "heuristic" in md


# --- routing ----------------------------------------------------------------------


def test_learned_router_routes_by_embedding(settings: Settings) -> None:
    router = LearnedRouter(settings, AxisEmbedder())
    for text, tier in (("easy", Tier.LOCAL), ("medium", Tier.MID), ("hard", Tier.FRONTIER)):
        d = router.route(f"a {text} question", [])
        assert d.tier == tier and d.confidence is not None and d.confidence > 0.9
        assert not d.escalated and d.embed_tokens > 0


def test_low_confidence_escalates_one_tier(settings: Settings) -> None:
    d = LearnedRouter(settings, AxisEmbedder()).route("a vague question", [])
    # between the local and mid axes: predicted local at ~0.56 confidence (< 0.6) -> one tier up
    assert d.confidence is not None and d.confidence < settings.router.learned.confidence_threshold
    assert d.escalated and d.tier == Tier.MID


def test_escalation_can_be_disabled(settings: Settings) -> None:
    off = settings.model_copy(update={"router": settings.router.model_copy(
        update={"learned": settings.router.learned.model_copy(update={"escalate": False})})})  # fmt: skip
    d = LearnedRouter(off, AxisEmbedder()).route("a vague question", [])
    assert d.tier == Tier.LOCAL and not d.escalated


def test_escalation_is_capped_at_highest_active_tier(settings: Settings) -> None:
    strict = settings.model_copy(update={"router": settings.router.model_copy(
        update={"learned": settings.router.learned.model_copy(update={"confidence_threshold": 0.99999})})})  # fmt: skip
    d = LearnedRouter(strict, AxisEmbedder()).route("a hard question", [])
    assert d.tier == Tier.FRONTIER  # nothing above frontier


def test_two_tier_mode_renormalizes_over_active_tiers(settings: Settings) -> None:
    two = settings.model_copy(update={"tier_mode": "two"})
    d = LearnedRouter(two, AxisEmbedder()).route("a hard question", [])
    assert d.tier in (Tier.LOCAL, Tier.MID)
    assert d.confidence is not None and d.confidence <= 1.0


def test_engine_logs_confidence_escalation_and_router_embedding(settings: Settings) -> None:
    logger = TraceLogger(settings.trace_db)
    engine = Engine(settings, {"gemini": RecordingProvider()}, logger, embedder=AxisEmbedder())
    engine.ask("a hard question")
    engine.ask("a vague question")
    hard, vague = logger.all()
    assert (hard.tier, hard.escalated) == (Tier.FRONTIER, False)
    assert hard.route_confidence is not None and hard.route_confidence > 0.9
    assert hard.embed_tokens > 0  # the router's query embedding is paid for
    assert (vague.tier, vague.escalated) == (Tier.MID, True)


def test_learned_router_needs_embedder_and_model(settings: Settings, tmp_path: Path) -> None:
    assert settings.needs_embedder
    with pytest.raises(ValueError, match="embedder"):
        build_router(settings, None)
    missing = settings.model_copy(update={"router": settings.router.model_copy(
        update={"learned": LearnedRouterConfig(model_path=tmp_path / "nope.json")})})  # fmt: skip
    with pytest.raises(FileNotFoundError, match="gpte router train"):
        build_router(missing, AxisEmbedder())


def test_model_embedding_space_must_match_config(settings: Settings) -> None:
    other = settings.model_copy(update={"embedding_dim": 768})
    with pytest.raises(ValueError, match="embedding"):
        LearnedRouter(other, AxisEmbedder())


def test_cache_key_tracks_the_model_file(settings: Settings) -> None:
    before = cache_namespace(settings)
    path = settings.router.learned.model_path
    path.write_text(model_for([Tier.LOCAL, Tier.MID, Tier.FRONTIER]).model_copy(
        update={"intercept": [0.1, 0.0, 0.0]}).model_dump_json())  # fmt: skip
    assert cache_namespace(settings) != before  # retraining invalidates cached answers


# --- labelling -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "scores,label",
    [
        ({Tier.LOCAL: 9, Tier.MID: 9}, Tier.LOCAL),
        ({Tier.LOCAL: 6, Tier.MID: 8}, Tier.MID),
        ({Tier.LOCAL: 6, Tier.MID: 7}, Tier.FRONTIER),
        ({Tier.LOCAL: None, Tier.MID: 8}, Tier.MID),  # unparseable verdict never qualifies
    ],
)
def test_label_from_scores(scores, label) -> None:
    assert label_from_scores(scores, THREE, min_score=8) == label


def test_label_query_uses_top_tier_answer_as_reference(settings: Settings) -> None:
    seen: list[str] = []

    class Judge:
        name = "judge"

        def complete(self, messages, max_tokens, model, temperature=None) -> Completion:
            seen.append(messages[-1].content)
            cand = messages[-1].content.split("CANDIDATE ANSWER")[-1]
            return Completion(text=json.dumps({"score": 8 if "[flash]" in cand else 5}),
                              tokens_in=5, tokens_out=5, cost_usd=1e-6, latency_ms=0.0, model=model)  # fmt: skip

    from gpt_efficient.evals.judge import Judge as J

    provider = RecordingProvider()
    rec = label_query(TrainQuery(id="t1", query="q?", category="factual"), settings,
                      {"gemini": provider}, J(settings, Judge()))  # fmt: skip
    assert provider.models == ["lite", "flash", "pro"]  # every active tier answered
    assert len(seen) == 2  # only the cheaper tiers are judged
    assert all("REFERENCE ANSWER:\n[pro] answer" in s for s in seen)
    assert rec.scores == {Tier.LOCAL: 5, Tier.MID: 8}
    assert rec.label == Tier.MID
    assert rec.cost_usd > 0


# --- training --------------------------------------------------------------------------


def test_train_router_learns_separable_labels() -> None:
    xs, ys = [], []
    for tier, axis in zip(THREE, AXES.values(), strict=True):
        for k in range(6):
            v = [a + 0.05 * ((k * 7 + i) % 3 - 1) for i, a in enumerate(axis)]
            xs.append(v)
            ys.append(tier)
    model = train_router(xs, ys, embedding_model="axis", C=10.0)
    assert model.classes == THREE
    assert model.label_counts == {"local": 6, "mid": 6, "frontier": 6}
    assert model.cv_accuracy is not None and model.cv_accuracy > 0.9
    probs = model.probabilities([0.0, 0.0, 1.0])
    assert max(probs, key=probs.get) == Tier.FRONTIER
    assert sum(probs.values()) == pytest.approx(1.0)


def test_router_model_round_trips_as_json(tmp_path: Path) -> None:
    m = model_for(THREE)
    p = tmp_path / "m.json"
    p.write_text(m.model_dump_json())
    assert RouterModel.model_validate_json(p.read_text()) == m


# --- data files ----------------------------------------------------------------------------


def _norm(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def test_training_queries_are_disjoint_from_eval_sets() -> None:
    train = load_train_queries(REPO / "evals" / "router_train.jsonl")
    assert len(train) >= 150
    assert len({t.id for t in train}) == len(train)
    assert {t.category for t in train} == {"factual", "reasoning", "math", "code", "writing"}
    eval_queries = [i.query for f in ("seed.jsonl", "conversations.jsonl")
                    for i in load_dataset(REPO / "evals" / f)]  # fmt: skip
    for t in train:
        a = _norm(t.query)
        for q in eval_queries:
            b = _norm(q)
            jaccard = len(a & b) / len(a | b)
            assert jaccard < 0.6, (t.id, q)


def test_repo_experiments_include_learned_router(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    base = Settings()
    exps = {e.name: e for e in load_experiments(REPO / "evals" / "experiments.toml")}
    assert {"learned", "learned-two"} <= set(exps)
    assert apply_overrides(base, exps["learned"].overrides).router.type == "learned"


# --- CLI end to end (offline) -------------------------------------------------------------


def test_gpte_router_label_train_eval_fake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gpt_efficient import cli

    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    labels, model = tmp_path / "labels.jsonl", tmp_path / "router-fake.json"
    for argv in (
        ["gpte", "router", "label", "--fake", "--limit", "30", "--out", str(labels)],
        ["gpte", "router", "train", "--fake", "--labels", str(labels), "--out", str(model)],
    ):
        monkeypatch.setattr("sys.argv", argv)
        cli.main()
    assert len(labels.read_text().splitlines()) == 30
    trained = RouterModel.model_validate_json(model.read_text())
    assert trained.n_train == 30 and trained.fake

    monkeypatch.setenv("GPTE_ROUTER__LEARNED__MODEL_PATH", str(model))
    out = tmp_path / "eval"
    monkeypatch.setattr("sys.argv", ["gpte", "eval", "--fake", "--limit", "6", "--only",
                                     "heuristic,learned", "--out", str(out)])  # fmt: skip
    cli.main()
    assert "learned" in (out / "report.md").read_text()
