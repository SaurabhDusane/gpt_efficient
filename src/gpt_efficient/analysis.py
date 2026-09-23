"""Results analysis: turns eval runs into answers to the research questions (SPEC §1).

Everything here is a pure function over `ItemResult`s so it can be tested
exactly; notebooks/results.ipynb and `gpte findings` only call into it.

Statistics: 95% (configurable) percentile bootstrap over items; comparisons
between configs are *paired* (same items) so item difficulty cancels out.
"Quality held" is non-inferiority: the lower CI bound of (config - reference)
quality must be above -margin. Interpretation is never auto-written: the
findings file only states what the numbers say, and marks runs made with fake
providers as ILLUSTRATIVE.
"""

import subprocess
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from gpt_efficient.config import AnalysisConfig, Settings
from gpt_efficient.evals.runner import ItemResult
from gpt_efficient.schemas import CacheStatus

# --- statistics -------------------------------------------------------------------


class CI(BaseModel):
    mean: float
    lo: float
    hi: float
    n: int

    def fmt(self, spec: str = ".3f", scale: float = 1.0, prefix: str = "") -> str:
        def one(v: float) -> str:
            s = format(v * scale, spec)
            return s[0] + prefix + s[1:] if prefix and s[:1] in "+-" else prefix + s  # "+$0.12"

        return f"{one(self.mean)} [{one(self.lo)}, {one(self.hi)}]"


def bootstrap_ci(values: list[float], cfg: AnalysisConfig) -> CI | None:
    """Mean with a percentile-bootstrap confidence interval (seeded, reproducible)."""
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(cfg.seed)
    means = x[rng.integers(0, len(x), size=(cfg.n_boot, len(x)))].mean(axis=1)
    alpha = (1 - cfg.confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    mean = float(x.mean())
    # guard float noise so a constant sample gives lo == mean == hi exactly
    return CI(mean=mean, lo=min(float(lo), mean), hi=max(float(hi), mean), n=len(x))


def paired_diff_ci(a: dict[str, float], b: dict[str, float], cfg: AnalysisConfig) -> CI | None:
    """Bootstrap CI of mean(a - b) over the items present in both."""
    common = [k for k in a if k in b]
    return bootstrap_ci([a[k] - b[k] for k in common], cfg)


def non_inferior(diff: CI | None, margin: float) -> bool | None:
    return None if diff is None else diff.lo > -margin


# --- runs ---------------------------------------------------------------------------


class RunMeta(BaseModel):
    fake: bool = False
    dataset: str = ""
    experiments: list[str] = []
    judge_model: str = ""
    rubric_version: str = ""
    git_commit: str | None = None
    generated: str = ""


class Run(BaseModel):
    path: Path
    meta: RunMeta
    results: list[ItemResult]

    @property
    def experiments(self) -> list[str]:
        return list(dict.fromkeys(r.experiment for r in self.results))

    def rows(self, experiment: str) -> list[ItemResult]:
        return [r for r in self.results if r.experiment == experiment]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def load_run(path: Path) -> Run:
    path = Path(path)
    results = [
        ItemResult.model_validate_json(line)
        for line in (path / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    meta_file = path / "run.json"
    if meta_file.exists():
        meta = RunMeta.model_validate_json(meta_file.read_text())
    else:  # runs from before run.json existed: recognise fake answers by their text
        meta = RunMeta(fake=any((r.answer or "").startswith("(fake ") for r in results))
    return Run(path=path, meta=meta, results=results)


def load_manifest(path: Path) -> dict[str, Path]:
    """`role = "path/to/run"` pairs; roles used: routing (RQ1-3), compression (RQ4)."""
    data = tomllib.loads(Path(path).read_text())
    return {role: Path(p) for role, p in data.items()}


def choose_reference(experiments: list[str], order: list[str]) -> str | None:
    return next((name for name in order if name in experiments), None)


def _answered(rows: list[ItemResult]) -> list[ItemResult]:
    return [r for r in rows if r.error is None]


def _by_item(rows: list[ItemResult], fn: Callable[[ItemResult], float | None]) -> dict[str, float]:
    out = {}
    for r in _answered(rows):
        v = fn(r)
        if v is not None:
            out[r.item_id] = v
    return out


def _quality(r: ItemResult) -> float | None:
    return r.quality


def _cost(r: ItemResult) -> float:
    return r.cost_usd


def _tokens(r: ItemResult) -> float:
    return float(r.total_tokens)


def _ci(rows: list[ItemResult], fn, cfg: AnalysisConfig) -> CI | None:
    return bootstrap_ci(list(_by_item(rows, fn).values()), cfg)


def _rel(new: float | None, base: float | None) -> float | None:
    return None if new is None or not base else new / base - 1


# --- RQ1: how much cost can be cut before quality degrades? ---------------------------


class ConfigRow(BaseModel):
    experiment: str
    n: int
    quality: CI | None
    tokens: CI | None
    cost: CI | None  # USD per query
    dq_vs_ref: CI | None = None  # paired quality difference vs. the reference
    held: bool | None = None  # non-inferior to the reference (None for the reference itself)
    cost_vs_ref: float | None = None  # relative change in mean cost/query
    tokens_vs_ref: float | None = None


def config_table(run: Run, cfg: AnalysisConfig, reference: str | None = None) -> tuple[str | None, list[ConfigRow]]:
    ref = reference or choose_reference(run.experiments, cfg.reference_order)
    ref_rows = run.rows(ref) if ref else []
    ref_q = _by_item(ref_rows, _quality)
    ref_cost, ref_tok = _ci(ref_rows, _cost, cfg), _ci(ref_rows, _tokens, cfg)
    table = []
    for name in run.experiments:
        rows = run.rows(name)
        row = ConfigRow(
            experiment=name,
            n=len(rows),
            quality=_ci(rows, _quality, cfg),
            tokens=_ci(rows, _tokens, cfg),
            cost=_ci(rows, _cost, cfg),
        )
        if ref and name != ref:
            row.dq_vs_ref = paired_diff_ci(_by_item(rows, _quality), ref_q, cfg)
            row.held = non_inferior(row.dq_vs_ref, cfg.margin)
            row.cost_vs_ref = _rel(row.cost.mean if row.cost else None, ref_cost.mean if ref_cost else None)
            row.tokens_vs_ref = _rel(row.tokens.mean if row.tokens else None, ref_tok.mean if ref_tok else None)
        table.append(row)
    return ref, table


def cheapest_held(table: list[ConfigRow], by: str = "cost") -> ConfigRow | None:
    """The cheapest config (by mean cost or tokens) that held quality and is cheaper than the reference."""
    rel = "cost_vs_ref" if by == "cost" else "tokens_vs_ref"
    held = [r for r in table if r.held and getattr(r, rel) is not None and getattr(r, rel) < 0]
    return min(held, key=lambda r: getattr(r, rel), default=None)


# --- RQ2: learned vs. heuristic router ----------------------------------------------------


class PairRow(BaseModel):
    a: str
    b: str
    dq: CI | None  # quality a - b
    dcost: CI | None  # cost/query a - b (USD)
    verdict: str


def _direction(ci: CI | None, up: str, down: str, same: str) -> str:
    if ci is None:
        return "not comparable"
    if ci.lo > 0:
        return up
    if ci.hi < 0:
        return down
    return same


def router_comparison(run: Run, cfg: AnalysisConfig) -> list[PairRow]:
    out = []
    for a, b in cfg.router_pairs:
        if a not in run.experiments or b not in run.experiments:
            continue
        ra, rb = run.rows(a), run.rows(b)
        dq = paired_diff_ci(_by_item(ra, _quality), _by_item(rb, _quality), cfg)
        dcost = paired_diff_ci(_by_item(ra, _cost), _by_item(rb, _cost), cfg)
        q = _direction(dq, "higher quality", "lower quality", "no significant quality difference")
        c = _direction(dcost, "more expensive", "cheaper", "no significant cost difference")
        out.append(PairRow(a=a, b=b, dq=dq, dcost=dcost, verdict=f"{a} vs {b}: {q}; {c}"))
    return out


# --- RQ3: where does caching help, and where does it serve wrong answers? --------------------


class CacheRow(BaseModel):
    experiment: str
    counterpart: str | None
    answered: int
    hits: int
    wrong_hits: int
    paraphrase_probes: int
    paraphrase_hits: int
    near_miss_probes: int
    near_miss_hits: int  # a hit on a near-miss probe served an answer to a different question
    dq_vs_counterpart: CI | None
    cost_vs_counterpart: float | None
    hit_sim: CI | None


class WrongHit(BaseModel):
    experiment: str
    item_id: str
    tags: list[str]
    cache_sim: float | None
    judge_score: int | None


def _is_near_miss(r: ItemResult) -> bool:
    return any(t.startswith("near-miss") for t in r.tags)


def cache_analysis(run: Run, cfg: AnalysisConfig, low_quality: float) -> tuple[list[CacheRow], list[WrongHit]]:
    rows_out, wrong = [], []
    for name in run.experiments:
        rows = _answered(run.rows(name))
        if not any(r.cache_status in (CacheStatus.HIT, CacheStatus.MISS) for r in rows):
            continue  # cache off in this experiment
        counterpart = name[: -len(cfg.cache_suffix)] if name.endswith(cfg.cache_suffix) else None
        if counterpart not in run.experiments:
            counterpart = None
        hits = [r for r in rows if r.cache_status == CacheStatus.HIT]
        bad = [r for r in hits if r.quality is not None and r.quality < low_quality]
        wrong += [WrongHit(experiment=name, item_id=r.item_id, tags=r.tags, cache_sim=r.cache_sim,
                           judge_score=r.judge_score) for r in bad]  # fmt: skip
        para = [r for r in rows if r.paraphrase_of]
        near = [r for r in rows if _is_near_miss(r)]
        dq = cost_rel = None
        if counterpart:
            base = run.rows(counterpart)
            dq = paired_diff_ci(_by_item(rows, _quality), _by_item(base, _quality), cfg)
            cost_ci, base_cost = _ci(rows, _cost, cfg), _ci(base, _cost, cfg)
            cost_rel = _rel(cost_ci.mean if cost_ci else None, base_cost.mean if base_cost else None)
        rows_out.append(CacheRow(
            experiment=name, counterpart=counterpart, answered=len(rows), hits=len(hits),
            wrong_hits=len(bad), paraphrase_probes=len(para),
            paraphrase_hits=sum(r.cache_status == CacheStatus.HIT for r in para),
            near_miss_probes=len(near), near_miss_hits=sum(r.cache_status == CacheStatus.HIT for r in near),
            dq_vs_counterpart=dq, cost_vs_counterpart=cost_rel,
            hit_sim=bootstrap_ci([r.cache_sim for r in hits if r.cache_sim is not None], cfg),
        ))  # fmt: skip
    return rows_out, wrong


# --- RQ4: what does compression cost in quality per token saved? ----------------------------


class CompressionRow(BaseModel):
    experiment: str
    compressed: int
    answered: int
    quality: CI | None
    dq: CI | None  # paired vs. the no-compression baseline
    tokens: CI | None  # one-shot
    tokens_amortized: CI | None
    cost: CI | None  # one-shot
    cost_amortized: CI | None
    quality_by_tag: dict[str, float]


def _amortized_tokens(r: ItemResult) -> float:
    return r.amortized_tokens if r.amortized_tokens is not None else float(r.total_tokens)


def _amortized_cost(r: ItemResult) -> float:
    return r.amortized_cost_usd if r.amortized_cost_usd is not None else r.cost_usd


def compression_analysis(run: Run, cfg: AnalysisConfig) -> list[CompressionRow]:
    base = run.rows(cfg.compression_baseline) if cfg.compression_baseline in run.experiments else []
    base_q = _by_item(base, _quality)
    out = []
    for name in run.experiments:
        rows = run.rows(name)
        answered = _answered(rows)
        tags = sorted({t for r in answered for t in r.tags})
        by_tag = {}
        for t in tags:
            qs = [r.quality for r in answered if t in r.tags and r.quality is not None]
            if qs:
                by_tag[t] = sum(qs) / len(qs)
        out.append(CompressionRow(
            experiment=name, compressed=sum(r.compressed for r in answered), answered=len(answered),
            quality=_ci(rows, _quality, cfg),
            dq=paired_diff_ci(_by_item(rows, _quality), base_q, cfg) if base and name != cfg.compression_baseline else None,
            tokens=_ci(rows, _tokens, cfg), tokens_amortized=_ci(rows, _amortized_tokens, cfg),
            cost=_ci(rows, _cost, cfg), cost_amortized=_ci(rows, _amortized_cost, cfg),
            quality_by_tag=by_tag,
        ))  # fmt: skip
    return out


# --- plots ------------------------------------------------------------------------------------


def frontier_ci_plot(points: list[tuple[str, CI, CI]], xlabel: str, xfmt, title: str, path: Path) -> None:
    """Quality (y) vs. a cost measure (x), each with its bootstrap CI as thin whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    from gpt_efficient.evals import report as style

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    fig.patch.set_facecolor(style._SURFACE)
    ax.set_facecolor(style._SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(style._GRID)
    ax.grid(True, color=style._GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=style._TEXT_2, labelsize=9)

    flat = [(n, x.mean, y.mean) for n, x, y in points]
    if points:
        front = style._pareto(flat)
        fp = sorted((x, y) for n, x, y in flat if n in front)
        ax.plot([p[0] for p in fp], [p[1] for p in fp], color=style._FRONTIER, linewidth=2, zorder=2)
        for name, x, y in points:
            on = name in front
            ax.errorbar([x.mean], [y.mean], xerr=[[x.mean - x.lo], [x.hi - x.mean]],
                        yerr=[[y.mean - y.lo], [y.hi - y.mean]], fmt="none",
                        ecolor=style._LEADER, elinewidth=1, capsize=0, zorder=1)  # fmt: skip
            ax.scatter([x.mean], [y.mean], s=80, zorder=4 if on else 3, linewidths=2,
                       facecolors=style._FRONTIER if on else style._SURFACE,
                       edgecolors=style._SURFACE if on else style._DOMINATED)  # fmt: skip
        xs = [x.mean for _, x, _ in points]
        if min(xs) > 0 and max(xs) / min(xs) > 20:
            ax.set_xscale("log")
        style._place_labels(fig, ax, flat, front)
        ax.scatter([], [], s=80, color=style._FRONTIER, label="on the efficiency frontier")
        ax.scatter([], [], s=80, facecolors=style._SURFACE, edgecolors=style._DOMINATED, linewidths=2,
                   label="dominated")  # fmt: skip
        ax.plot([], [], color=style._LEADER, linewidth=1, label="95% bootstrap CI")
        ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=style._TEXT_2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: xfmt(v)))
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("mean quality (judge, 0–1)", color=style._TEXT_2, fontsize=9)
    ax.set_xlabel(xlabel, color=style._TEXT_2, fontsize=9)
    ax.set_title(title, color=style._TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=style._SURFACE)
    plt.close(fig)


def _usd_1k(v: float) -> str:
    return f"${v:,.3g}"


# --- findings -----------------------------------------------------------------------------------

_BANNER = (
    "> **ILLUSTRATIVE — not results.** This section was generated from runs made with fake "
    "providers (`--fake`). The numbers only exercise the pipeline; do not cite them.\n"
)
_TODO = "_Interpretation: TODO — write this from real runs; nothing here is auto-concluded._"


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:+.0%}"


def _c(ci: CI | None, spec: str = ".3f", scale: float = 1.0, prefix: str = "") -> str:
    return "—" if ci is None else ci.fmt(spec, scale, prefix)


def _held(v: bool | None) -> str:
    return "reference" if v is None else ("**held**" if v else "degraded")


def write_findings(runs: dict[str, Run], settings: Settings, out_dir: Path) -> dict[str, Path]:
    """Write findings.md (+ figures) from the runs named in the manifest."""
    cfg = settings.analysis
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {"findings": out_dir / "findings.md"}
    level = f"{cfg.confidence:.0%}"
    lines = [
        "# Findings: quality per token",
        "",
        f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `gpte findings` / "
        f"notebooks/results.ipynb (commit {git_commit() or 'unknown'})._",
        "",
        "## Method",
        "",
        f"- Uncertainty: {level} percentile bootstrap over items ({cfg.n_boot} resamples, seed {cfg.seed}); "
        "config comparisons are paired on the same items.",
        f"- Quality held = non-inferiority: lower {level} bound of (config − reference) quality > −{cfg.margin} "
        f"(quality is judge score mapped to 0–1; {cfg.margin} ≈ {cfg.margin * 9:.2g} judge points).",
        f"- Reference: first of {cfg.reference_order} present in the run.",
        "- Tokens/query = LLM in + out (incl. thinking) + embedding + summarizer tokens; cost excludes the judge.",
        "",
        "| role | run | fake | dataset | judge | rubric |",
        "|---|---|---|---|---|---|",
    ]
    for role, run in runs.items():
        m = run.meta
        lines.append(f"| {role} | `{run.path}` | {'**yes**' if m.fake else 'no'} | {m.dataset or '—'} "
                     f"| {m.judge_model or '—'} | {m.rubric_version or '—'} |")  # fmt: skip

    routing = runs.get("routing")
    lines += ["", "## RQ1 — How much cost can be cut before quality degrades?", ""]
    if routing is None:
        lines.append("No routing run in the manifest.")
    else:
        if routing.meta.fake:
            lines.append(_BANNER)
        ref, table = config_table(routing, cfg)
        lines += [
            f"Reference: **{ref or 'none found'}**.",
            "",
            "| config | quality | Δ quality vs ref | quality held? | tokens / query | $ / 1k queries | cost vs ref |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in table:
            lines.append(
                f"| {r.experiment} | {_c(r.quality)} | {_c(r.dq_vs_ref, '+.3f')} | {_held(r.held)} "
                f"| {_c(r.tokens, ',.0f')} | {_c(r.cost, ',.4f', 1000, '$')} | {_pct(r.cost_vs_ref)} |"
            )
        best_cost, best_tok = cheapest_held(table, "cost"), cheapest_held(table, "tokens")
        lines += [""]
        if best_cost:
            lines.append(f"- Cheapest config that held quality (by cost): **{best_cost.experiment}**, "
                         f"{_pct(best_cost.cost_vs_ref)} cost vs. {ref}.")  # fmt: skip
        else:
            lines.append(f"- No cheaper config held quality within −{cfg.margin} of {ref}.")
        if best_tok:
            lines.append(f"- Cheapest by tokens: **{best_tok.experiment}**, {_pct(best_tok.tokens_vs_ref)} tokens.")
        points_cost = [(r.experiment, r.cost.model_copy(update={
            "mean": r.cost.mean * 1000, "lo": r.cost.lo * 1000, "hi": r.cost.hi * 1000}), r.quality)
            for r in table if r.cost and r.quality]  # fmt: skip
        points_tok = [(r.experiment, r.tokens, r.quality) for r in table if r.tokens and r.quality]
        paths["frontier_cost_ci"] = out_dir / "frontier_cost_ci.png"
        paths["frontier_tokens_ci"] = out_dir / "frontier_tokens_ci.png"
        frontier_ci_plot(points_cost, "system cost per 1k queries (USD, judge excluded)", _usd_1k,
                         "Quality vs. cost (95% CIs)", paths["frontier_cost_ci"])  # fmt: skip
        frontier_ci_plot(points_tok, "mean tokens per query", lambda v: f"{v:,.0f}",
                         "Quality vs. tokens (95% CIs)", paths["frontier_tokens_ci"])  # fmt: skip
        lines += ["", "![Quality vs. cost](frontier_cost_ci.png)", "",
                  "![Quality vs. tokens](frontier_tokens_ci.png)", "", _TODO]  # fmt: skip

    lines += ["", "## RQ2 — Learned router vs. heuristic: is the extra complexity worth it?", ""]
    pairs = router_comparison(routing, cfg) if routing else []
    if not pairs:
        lines.append("No learned/heuristic experiment pair in the routing run.")
    else:
        if routing and routing.meta.fake:
            lines.append(_BANNER)
        lines += ["| comparison | Δ quality | Δ $ / 1k queries | verdict |", "|---|---|---|---|"]
        for p in pairs:
            lines.append(f"| {p.a} − {p.b} | {_c(p.dq, '+.3f')} | {_c(p.dcost, '+,.4f', 1000, '$')} | {p.verdict} |")
        lines += ["", _TODO]

    lines += ["", "## RQ3 — Where does semantic caching help, and where does it serve wrong answers?", ""]
    cache_rows, wrong = cache_analysis(routing, cfg, settings.eval.low_quality) if routing else ([], [])
    if not cache_rows:
        lines.append("No cache experiments in the routing run.")
    else:
        if routing and routing.meta.fake:
            lines.append(_BANNER)
        lines += [
            f"Threshold {settings.cache.threshold}; wrong hit = cache hit judged below {settings.eval.low_quality}.",
            "",
            "| config | vs | hits / answered | wrong hits | paraphrase probes hit | near-miss probes hit "
            "| hit similarity | Δ quality | cost change |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in cache_rows:
            lines.append(
                f"| {r.experiment} | {r.counterpart or '—'} | {r.hits} / {r.answered} | {r.wrong_hits} "
                f"| {r.paraphrase_hits} / {r.paraphrase_probes} | {r.near_miss_hits} / {r.near_miss_probes} "
                f"| {_c(r.hit_sim, '.3f')} | {_c(r.dq_vs_counterpart, '+.3f')} | {_pct(r.cost_vs_counterpart)} |"
            )
        if wrong:
            lines += ["", "Wrong hits:", ""] + [
                f"- {w.experiment} / {w.item_id} (sim {w.cache_sim if w.cache_sim is None else round(w.cache_sim, 4)}, "
                f"judge {w.judge_score}, tags {w.tags or '—'})" for w in wrong
            ]  # fmt: skip
        lines += ["", _TODO]

    comp = runs.get("compression")
    lines += ["", "## RQ4 — What does context compression cost in quality per token saved?", ""]
    if comp is None:
        lines.append("No compression run in the manifest.")
    else:
        if comp.meta.fake:
            lines.append(_BANNER)
        rows = compression_analysis(comp, cfg)
        lines += [
            f"Baseline: **{cfg.compression_baseline}**. One-shot = each item pays for its whole history; "
            "amortized = compressor overhead averaged over the conversation's turns.",
            "",
            "| strategy | compressed | quality | Δ quality | tokens (one-shot) | tokens (amortized) "
            "| $ / 1k (one-shot) | $ / 1k (amortized) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r.experiment} | {r.compressed} / {r.answered} | {_c(r.quality)} | {_c(r.dq, '+.3f')} "
                f"| {_c(r.tokens, ',.0f')} | {_c(r.tokens_amortized, ',.0f')} "
                f"| {_c(r.cost, ',.4f', 1000, '$')} | {_c(r.cost_amortized, ',.4f', 1000, '$')} |"
            )
        tags = sorted({t for r in rows for t in r.quality_by_tag})
        if tags:
            lines += ["", "Quality by probe type:", "", "| strategy | " + " | ".join(tags) + " |",
                      "|---|" + "---|" * len(tags)]  # fmt: skip
            for r in rows:
                cells = [f"{r.quality_by_tag[t]:.3f}" if t in r.quality_by_tag else "—" for t in tags]
                lines.append(f"| {r.experiment} | " + " | ".join(cells) + " |")
        paths["compression_amortized"] = out_dir / "compression_amortized.png"
        paths["compression_oneshot"] = out_dir / "compression_oneshot.png"
        scale = lambda c: c.model_copy(update={"mean": c.mean * 1000, "lo": c.lo * 1000, "hi": c.hi * 1000})  # noqa: E731
        frontier_ci_plot([(r.experiment, scale(r.cost_amortized), r.quality) for r in rows
                          if r.cost_amortized and r.quality],
                         "cost per 1k queries, compressor overhead amortized (USD)", _usd_1k,
                         "Compression: quality vs. amortized cost", paths["compression_amortized"])  # fmt: skip
        frontier_ci_plot([(r.experiment, scale(r.cost), r.quality) for r in rows if r.cost and r.quality],
                         "cost per 1k queries, one-shot (USD)", _usd_1k,
                         "Compression: quality vs. one-shot cost", paths["compression_oneshot"])  # fmt: skip
        lines += ["", "![Compression, amortized](compression_amortized.png)", "",
                  "![Compression, one-shot](compression_oneshot.png)", "", _TODO]  # fmt: skip

    lines += [
        "",
        "## Threats to validity",
        "",
        "- **Single judge, same model family.** Answers and judge are all Gemini; a cross-family judge "
        "(deferred with the Anthropic/OpenAI adapters) would test for family bias.",
        "- **Small samples.** Seed set 42 items, 10 conversations: read the CIs, not the point estimates.",
        "- **Estimated tokens.** Embedding and history token counts are character-based estimates.",
        "- **Learned-router labels are pseudo-references** (agreement with the top tier), not ground truth.",
        "- **Preview model.** The frontier tier is a preview model and may change under the benchmark.",
        "- **Sampling variance.** Answers use provider-default temperature and each config ran once.",
        "",
    ]
    paths["findings"].write_text("\n".join(lines))
    return paths
