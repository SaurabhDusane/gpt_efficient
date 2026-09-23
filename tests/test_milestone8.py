"""Milestone 8: results notebook — efficiency-frontier plots and writeup.

The statistics are pure functions checked exactly. The research-question tables
run on synthetic results with known answers. The milestone test builds fake
eval runs with the real CLI, then executes notebooks/results.ipynb end to end.
"""

import json
from pathlib import Path

import pytest

from gpt_efficient.analysis import (
    CI,
    Run,
    RunMeta,
    bootstrap_ci,
    cache_analysis,
    cheapest_held,
    choose_reference,
    compression_analysis,
    config_table,
    load_run,
    non_inferior,
    paired_diff_ci,
    router_comparison,
    write_findings,
)
from gpt_efficient.config import AnalysisConfig, Settings
from gpt_efficient.evals.runner import ItemResult
from gpt_efficient.schemas import CacheStatus, Tier

REPO = Path(__file__).resolve().parents[1]
CFG = AnalysisConfig(n_boot=500)


def result(exp: str, item: str, quality: float | None, cost: float = 1e-4, tokens: int = 100,
           tier: Tier = Tier.MID, **kw) -> ItemResult:  # fmt: skip
    return ItemResult(experiment=exp, item_id=item, difficulty="easy", category="factual",
                      tier=tier, provider="gemini", model="m", cache_status=kw.pop("cache", CacheStatus.DISABLED),
                      tokens_in=tokens, cost_usd=cost, quality=quality,
                      judge_score=None if quality is None else round(quality * 9 + 1), **kw)  # fmt: skip


def run_of(results: list[ItemResult], fake: bool = False) -> Run:
    return Run(path=Path("x"), meta=RunMeta(fake=fake), results=results)


# --- statistics ------------------------------------------------------------------


def test_bootstrap_ci_basics() -> None:
    assert bootstrap_ci([], CFG) is None
    c = bootstrap_ci([0.5] * 10, CFG)
    assert (c.mean, c.lo, c.hi, c.n) == (0.5, 0.5, 0.5, 10)
    vals = [0.1, 0.4, 0.5, 0.9, 0.6, 0.3, 0.8, 0.2]
    a, b = bootstrap_ci(vals, CFG), bootstrap_ci(vals, CFG)
    assert a == b  # seeded: reproducible
    assert a.lo < a.mean < a.hi and a.mean == pytest.approx(sum(vals) / len(vals))


def test_paired_diff_ci_pairs_by_item() -> None:
    a = {"i1": 0.6, "i2": 0.7, "i3": 0.8, "only_a": 0.0}
    b = {"i1": 0.5, "i2": 0.6, "i3": 0.7, "only_b": 1.0}
    d = paired_diff_ci(a, b, CFG)
    assert d.n == 3  # only items present in both
    assert (d.mean, d.lo, d.hi) == pytest.approx((0.1, 0.1, 0.1))
    assert paired_diff_ci({"x": 1.0}, {"y": 1.0}, CFG) is None


def test_non_inferiority_uses_the_margin() -> None:
    assert non_inferior(CI(mean=0.0, lo=-0.04, hi=0.02, n=10), 0.05) is True
    assert non_inferior(CI(mean=0.0, lo=-0.06, hi=0.02, n=10), 0.05) is False
    assert non_inferior(None, 0.05) is None


def test_choose_reference() -> None:
    order = CFG.reference_order
    assert choose_reference(["heuristic", "fixed-mid", "fixed-frontier"], order) == "fixed-frontier"
    assert choose_reference(["heuristic", "fixed-mid"], order) == "fixed-mid"
    assert choose_reference(["heuristic"], order) is None


# --- RQ1: cost cut before quality degrades -------------------------------------------


def rq1_run() -> Run:
    rows = []
    for i in range(20):
        item = f"i{i}"
        rows += [
            result("fixed-frontier", item, 0.9, cost=1e-3, tokens=500, tier=Tier.FRONTIER),
            result("heuristic", item, 0.89, cost=2e-4, tokens=150),  # tiny drop, much cheaper
            result("fixed-local", item, 0.6, cost=5e-5, tokens=80, tier=Tier.LOCAL),  # clear drop
        ]
    return run_of(rows)


def test_config_table_and_cheapest_held() -> None:
    ref, table = config_table(rq1_run(), CFG)
    assert ref == "fixed-frontier"
    by = {r.experiment: r for r in table}
    assert by["fixed-frontier"].held is None  # the reference itself
    assert by["heuristic"].held is True and by["fixed-local"].held is False
    assert by["heuristic"].dq_vs_ref.mean == pytest.approx(-0.01)
    assert by["heuristic"].cost_vs_ref == pytest.approx(-0.8)
    best = cheapest_held(table, by="cost")
    assert best is not None and best.experiment == "heuristic"


def test_failed_and_unjudged_items_are_excluded() -> None:
    rows = [result("fixed-mid", "a", 0.5), result("fixed-mid", "b", None),
            result("fixed-mid", "c", 0.7, error="429")]  # fmt: skip
    _, [row] = config_table(run_of(rows), CFG)
    assert row.quality.n == 1 and row.cost.n == 2


# --- RQ2: learned vs heuristic ---------------------------------------------------------


def test_router_comparison_verdicts() -> None:
    rows = []
    for i in range(15):
        rows += [
            result("heuristic", f"i{i}", 0.6, cost=2e-4),
            result("learned", f"i{i}", 0.7, cost=3e-4),
        ]
    [pair] = router_comparison(run_of(rows), CFG)
    assert (pair.a, pair.b) == ("learned", "heuristic")
    assert pair.dq.mean == pytest.approx(0.1)
    assert "higher quality" in pair.verdict and "more expensive" in pair.verdict


def test_router_comparison_skips_missing_pairs() -> None:
    assert router_comparison(run_of([result("heuristic", "a", 0.5)]), CFG) == []


# --- RQ3: caching -------------------------------------------------------------------------


def test_cache_analysis_counts_hits_wrong_hits_and_probes() -> None:
    rows = []
    for i, (tags, para, hit, q) in enumerate([
        ([], None, False, 0.8),
        ([], "i0", True, 0.8),  # paraphrase probe hit, fine
        (["near-miss:i0"], None, True, 0.1),  # near-miss probe served a wrong cached answer
        ([], None, False, 0.8),
    ]):  # fmt: skip
        item = f"i{i}"
        rows.append(result("fixed-mid", item, 0.8, cost=2e-4))
        rows.append(result("fixed-mid+cache", item, q, cost=1e-5 if hit else 2e-4, tags=tags,
                           paraphrase_of=para, cache=CacheStatus.HIT if hit else CacheStatus.MISS,
                           cache_sim=0.97 if hit else 0.5))  # fmt: skip
    [row], wrong = cache_analysis(run_of(rows), CFG, low_quality=0.5)
    assert (row.experiment, row.counterpart) == ("fixed-mid+cache", "fixed-mid")
    assert (row.hits, row.answered, row.wrong_hits) == (2, 4, 1)
    assert (row.paraphrase_probes, row.paraphrase_hits) == (1, 1)
    assert (row.near_miss_probes, row.near_miss_hits) == (1, 1)
    assert row.dq_vs_counterpart.mean == pytest.approx((0 + 0 - 0.7 + 0) / 4)
    assert [w.item_id for w in wrong] == ["i2"]


# --- RQ4: compression ----------------------------------------------------------------------


def test_compression_analysis_one_shot_and_amortized() -> None:
    rows = []
    for i, tag in enumerate(["needle:early", "recent-only"]):
        item = f"c{i}"
        rows.append(result("no-compression", item, 0.9, cost=1e-3, tokens=2000, tags=[tag],
                           amortized_tokens=2000, amortized_cost_usd=1e-3))  # fmt: skip
        rows.append(result("summary", item, 0.8, cost=1.2e-3, tokens=3000, tags=[tag], compressed=True,
                           amortized_tokens=1200, amortized_cost_usd=6e-4))  # fmt: skip
    table = compression_analysis(run_of(rows), CFG)
    by = {r.experiment: r for r in table}
    s = by["summary"]
    assert s.dq.mean == pytest.approx(-0.1)
    assert s.tokens.mean == pytest.approx(3000) and s.tokens_amortized.mean == pytest.approx(1200)
    assert s.cost_amortized.mean == pytest.approx(6e-4)
    assert s.quality_by_tag == {"needle:early": pytest.approx(0.8), "recent-only": pytest.approx(0.8)}


# --- run loading / fake guard -------------------------------------------------------------


def test_load_run_reads_run_json_and_infers_fake_for_legacy_runs(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "results.jsonl").write_text(result("fixed-mid", "a", 0.5, answer="Paris.").model_dump_json() + "\n")
    (real / "run.json").write_text(RunMeta(fake=False, dataset="d.jsonl").model_dump_json())
    assert load_run(real).meta.fake is False

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "results.jsonl").write_text(
        result("fixed-mid", "a", 0.5, answer="(fake flash answer to: q)").model_dump_json() + "\n"
    )
    assert load_run(legacy).meta.fake is True


def test_findings_flag_fake_runs_and_leave_interpretation_open(tmp_path: Path) -> None:
    paths = write_findings({"routing": run_of(rq1_run().results, fake=True)}, Settings(analysis=CFG), tmp_path)
    md = paths["findings"].read_text()
    assert "ILLUSTRATIVE" in md
    assert "heuristic" in md and "fixed-frontier" in md
    assert "TODO" in md  # interpretation is never auto-written
    assert "no compression run" in md.lower()
    assert paths["frontier_cost_ci"].exists() and paths["frontier_tokens_ci"].exists()


def test_real_runs_are_not_flagged(tmp_path: Path) -> None:
    paths = write_findings({"routing": rq1_run()}, Settings(analysis=CFG), tmp_path)
    assert "ILLUSTRATIVE" not in paths["findings"].read_text()


def test_eval_writes_run_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gpt_efficient import cli

    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    out = tmp_path / "r"
    monkeypatch.setattr("sys.argv", ["gpte", "eval", "--fake", "--limit", "2", "--only", "fixed-mid", "--out", str(out)])
    cli.main()
    meta = json.loads((out / "run.json").read_text())
    assert meta["fake"] is True and meta["experiments"] == ["fixed-mid"]
    assert meta["judge_model"] and meta["rubric_version"]


# --- the milestone test -----------------------------------------------------------------------


def test_results_notebook_runs_end_to_end_on_fake_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import nbclient
    import nbformat

    from gpt_efficient import cli

    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    routing, compression, findings = tmp_path / "routing", tmp_path / "compression", tmp_path / "findings"
    for argv in (
        ["gpte", "eval", "--fake", "--limit", "10", "--only",
         "fixed-local,fixed-mid,fixed-frontier,heuristic,fixed-mid+cache", "--out", str(routing)],
        ["gpte", "eval", "--fake", "--limit", "3", "--dataset", str(REPO / "evals" / "conversations.jsonl"),
         "--experiments", str(REPO / "evals" / "experiments_compression.toml"), "--out", str(compression)],
    ):  # fmt: skip
        monkeypatch.setattr("sys.argv", argv)
        cli.main()
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(f'routing = "{routing}"\ncompression = "{compression}"\n')

    monkeypatch.setenv("GPTE_ANALYSIS__MANIFEST", str(manifest))
    monkeypatch.setenv("GPTE_ANALYSIS__OUT_DIR", str(findings))
    nb = nbformat.read(REPO / "notebooks" / "results.ipynb", as_version=4)
    assert all(c.get("outputs", []) == [] for c in nb.cells if c.cell_type == "code")  # committed clean
    nbclient.NotebookClient(nb, kernel_name="python3", timeout=300, resources={"metadata": {"path": str(REPO)}}).execute()

    md = (findings / "findings.md").read_text()
    assert "ILLUSTRATIVE" in md
    for section in ("RQ1", "RQ2", "RQ3", "RQ4"):
        assert section in md
    for fig in ("frontier_cost_ci.png", "frontier_tokens_ci.png", "compression_amortized.png"):
        assert (findings / fig).stat().st_size > 0
    images = sum(1 for c in nb.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    assert images >= 2  # plots render inline in the notebook too


def test_gpte_findings_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gpt_efficient import cli

    monkeypatch.setenv("GPTE_CONFIG", str(REPO / "config.toml"))
    routing = tmp_path / "routing"
    monkeypatch.setattr("sys.argv", ["gpte", "eval", "--fake", "--limit", "4", "--only",
                                     "fixed-mid,heuristic", "--out", str(routing)])  # fmt: skip
    cli.main()
    manifest = tmp_path / "m.toml"
    manifest.write_text(f'routing = "{routing}"\n')
    monkeypatch.setattr("sys.argv", ["gpte", "findings", "--manifest", str(manifest), "--out", str(tmp_path / "f")])
    cli.main()
    assert "fixed-mid" in (tmp_path / "f" / "findings.md").read_text()


def test_ci_formatting_puts_sign_before_currency() -> None:
    assert CI(mean=0.12, lo=-0.01, hi=0.2, n=3).fmt("+.2f", prefix="$") == "+$0.12 [-$0.01, +$0.20]"
    assert CI(mean=0.5, lo=0.4, hi=0.6, n=3).fmt(".1f") == "0.5 [0.4, 0.6]"
