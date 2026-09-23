"""Quality-per-token report: per-experiment summary, CSV, frontier plots, markdown.

Definitions (also printed in the report):
- quality: judge score (1-10) mapped to 0-1; mean over judged items.
- tokens/query: LLM tokens in + out (incl. thinking) + estimated embedding
  tokens + summarizer tokens; mean over answered items. Cache hits count 0
  LLM tokens.
- cost/query: system cost only (LLM + embeddings). Judge cost is separate.
- failed requests are excluded from quality/token/cost means and counted.
"""

import csv
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from gpt_efficient.config import Settings  # noqa: E402
from gpt_efficient.evals.judge import RUBRIC_VERSION  # noqa: E402
from gpt_efficient.evals.runner import ItemResult  # noqa: E402
from gpt_efficient.schemas import CacheStatus  # noqa: E402

_DIFFICULTY_ORDER = ["easy", "medium", "hard"]


class DifficultySummary(BaseModel):
    n: int
    mean_quality: float | None
    mean_tokens: float | None
    mean_cost_usd: float | None


class ExperimentSummary(BaseModel):
    experiment: str
    n: int
    answered: int
    errors: int
    judged: int
    judge_errors: int
    mean_quality: float | None
    mean_tokens: float | None
    mean_cost_usd: float | None
    mean_latency_ms: float | None
    q_per_1k_tokens: float | None
    q_per_usd: float | None
    cache_hits: int
    wrong_cache_hits: int
    compressed: int
    mean_tokens_saved: float | None  # history tokens removed per answered query (gross)
    # Compressor overhead averaged over each conversation's turns (= one-shot if none).
    mean_tokens_amortized: float | None
    mean_cost_amortized: float | None
    tier_mix: dict[str, int]
    exact_items: int
    exact_acc: float | None
    judge_exact_disagreements: int
    judge_cost_usd: float
    by_difficulty: dict[str, DifficultySummary]


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _is_wrong_hit(r: ItemResult, low_quality: float) -> bool:
    return r.cache_status == CacheStatus.HIT and r.quality is not None and r.quality < low_quality


def _disagrees(r: ItemResult) -> bool:
    """Exact check and judge point opposite ways (either may be the one that's wrong)."""
    if r.exact_match is None or r.judge_score is None:
        return False
    return (r.exact_match and r.judge_score <= 4) or (not r.exact_match and r.judge_score >= 8)


def summarize(results: list[ItemResult], low_quality: float) -> list[ExperimentSummary]:
    """One summary per experiment, in the order experiments first appear."""
    names = list(dict.fromkeys(r.experiment for r in results))
    out = []
    for name in names:
        rows = [r for r in results if r.experiment == name]
        answered = [r for r in rows if r.error is None]
        judged = [r for r in answered if r.quality is not None]
        exact = [r for r in answered if r.exact_match is not None]
        mean_q = _mean([r.quality for r in judged if r.quality is not None])
        mean_tok = _mean([float(r.total_tokens) for r in answered])
        mean_cost = _mean([r.cost_usd for r in answered])
        by_diff = {}
        for d in _DIFFICULTY_ORDER:
            d_ans = [r for r in answered if r.difficulty == d]
            if not d_ans:
                continue
            by_diff[d] = DifficultySummary(
                n=len(d_ans),
                mean_quality=_mean([r.quality for r in d_ans if r.quality is not None]),
                mean_tokens=_mean([float(r.total_tokens) for r in d_ans]),
                mean_cost_usd=_mean([r.cost_usd for r in d_ans]),
            )
        out.append(
            ExperimentSummary(
                experiment=name,
                n=len(rows),
                answered=len(answered),
                errors=len(rows) - len(answered),
                judged=len(judged),
                judge_errors=sum(r.judge_error is not None for r in answered),
                mean_quality=mean_q,
                mean_tokens=mean_tok,
                mean_cost_usd=mean_cost,
                mean_latency_ms=_mean([r.latency_ms for r in answered]),
                q_per_1k_tokens=mean_q / (mean_tok / 1000) if mean_q is not None and mean_tok else None,
                q_per_usd=mean_q / mean_cost if mean_q is not None and mean_cost else None,
                cache_hits=sum(r.cache_status == CacheStatus.HIT for r in answered),
                wrong_cache_hits=sum(_is_wrong_hit(r, low_quality) for r in answered),
                compressed=sum(r.compressed for r in answered),
                mean_tokens_saved=_mean([float(r.tokens_saved) for r in answered]),
                mean_tokens_amortized=_mean(
                    [r.amortized_tokens if r.amortized_tokens is not None else float(r.total_tokens)
                     for r in answered]
                ),
                mean_cost_amortized=_mean(
                    [r.amortized_cost_usd if r.amortized_cost_usd is not None else r.cost_usd
                     for r in answered]
                ),  # fmt: skip
                tier_mix=dict(Counter(r.tier.value for r in answered)),
                exact_items=len(exact),
                exact_acc=_mean([1.0 if r.exact_match else 0.0 for r in exact]),
                judge_exact_disagreements=sum(_disagrees(r) for r in answered),
                judge_cost_usd=sum(r.judge_cost_usd for r in rows),
                by_difficulty=by_diff,
            )
        )
    return out


# --- efficiency-frontier plots ----------------------------------------------------

# Reference palette (dataviz skill), light surface.
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_2 = "#52514e"
_GRID = "#e6e5e0"
_FRONTIER = "#2a78d6"  # categorical slot 1
_DOMINATED = "#8f8e88"  # muted: context, not a category


def _pareto(points: list[tuple[str, float, float]]) -> set[str]:
    """Names of points no other point beats on both axes (lower x, higher y)."""
    front = set()
    for name, x, y in points:
        dominated = any(
            (x2 <= x and y2 >= y) and (x2 < x or y2 > y) for n2, x2, y2 in points if n2 != name
        )
        if not dominated:
            front.add(name)
    return front


# Label spots tried around a point, nearest first: (dx, dy, ha, va) in points.
# Rings beyond the first get a thin leader line back to the point.
_LABEL_SPOTS = [
    (7, 5, "left", "bottom"), (7, -5, "left", "top"), (-7, 5, "right", "bottom"),
    (-7, -5, "right", "top"), (0, 10, "center", "bottom"), (0, -10, "center", "top"),
] + [
    spot
    for k in range(1, 6)
    for spot in ((12, 14 * k, "left", "bottom"), (12, -14 * k, "left", "top"),
                 (-12, 14 * k, "right", "bottom"), (-12, -14 * k, "right", "top"))
]  # fmt: skip
_LEADER = "#b5b4ae"


def _place_labels(fig, ax, points: list[tuple[str, float, float]], front: set[str]) -> None:
    """Direct labels that don't collide with each other or with markers.

    Coincident points share one stacked label; every other label takes the
    nearest spot whose box overlaps nothing already placed.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clusters: list[list[tuple[str, float, float]]] = []
    for pt in points:
        px = ax.transData.transform((pt[1], pt[2]))
        for c in clusters:
            cx = ax.transData.transform((c[0][1], c[0][2]))
            if abs(px[0] - cx[0]) < 6 and abs(px[1] - cx[1]) < 6:
                c.append(pt)
                break
        else:
            clusters.append([pt])
    # markers are obstacles too (8px marker + 2px ring)
    placed = []
    for _, x, y in points:
        px, py = ax.transData.transform((x, y))
        placed.append(Bbox.from_extents(px - 7, py - 7, px + 7, py + 7))
    axes_box = ax.get_window_extent(renderer)

    def annotate(text: str, xy: tuple[float, float], spot, color: str, leader: bool):
        dx, dy, ha, va = spot
        props = {"arrowstyle": "-", "color": _LEADER, "lw": 0.8, "shrinkA": 2, "shrinkB": 6}
        return ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                           ha=ha, va=va, color=color, arrowprops=props if leader else None)  # fmt: skip

    for c in clusters:
        text = "\n".join(n for n, _, _ in c)
        color = _TEXT if any(n in front for n, _, _ in c) else _TEXT_2
        xy = (c[0][1], c[0][2])
        box = None
        for i, spot in enumerate(_LABEL_SPOTS):
            ann = annotate(text, xy, spot, color, leader=i >= 6)
            cand = ann.get_window_extent(renderer).expanded(1.04, 1.1)
            inside = axes_box.x0 <= cand.x0 and cand.x1 <= axes_box.x1 + 60
            if inside and not any(cand.overlaps(b) for b in placed):
                box = cand
                break
            ann.remove()
        if box is None:  # nowhere free: nearest spot anyway
            box = annotate(text, xy, _LABEL_SPOTS[0], color, leader=False).get_window_extent(renderer)
        placed.append(box)


def _frontier_plot(
    summaries: list[ExperimentSummary], x_of, xlabel: str, xfmt, title: str, path: Path
) -> None:
    points = [
        (s.experiment, x_of(s), s.mean_quality)
        for s in summaries
        if s.mean_quality is not None and x_of(s) is not None
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.grid(True, color=_GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=_TEXT_2, labelsize=9)

    if points:
        front = _pareto(points)
        fp = sorted((x, y) for n, x, y in points if n in front)
        ax.plot([p[0] for p in fp], [p[1] for p in fp], color=_FRONTIER, linewidth=2, zorder=2)
        for name, x, y in points:
            on = name in front
            ax.scatter(
                [x], [y], s=80, zorder=4 if on else 3, linewidths=2,  # frontier on top
                facecolors=_FRONTIER if on else _SURFACE,
                edgecolors=_SURFACE if on else _DOMINATED,
            )  # fmt: skip
        xs = [p[1] for p in points]
        if min(xs) > 0 and max(xs) / min(xs) > 20:
            ax.set_xscale("log")
        _place_labels(fig, ax, points, front)
        ax.scatter([], [], s=80, color=_FRONTIER, label="on the efficiency frontier")
        ax.scatter([], [], s=80, facecolors=_SURFACE, edgecolors=_DOMINATED, linewidths=2,
                   label="dominated (another config is cheaper and at least as good)")  # fmt: skip
        ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=_TEXT_2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: xfmt(v)))
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("mean quality (judge, 0–1)", color=_TEXT_2, fontsize=9)
    ax.set_xlabel(xlabel, color=_TEXT_2, fontsize=9)
    ax.set_title(title, color=_TEXT, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
    plt.close(fig)


# --- output -------------------------------------------------------------------------


def _fmt(v: float | None, spec: str) -> str:
    return "—" if v is None else format(v, spec)


def _usd_per_1k(v: float | None) -> str:
    return "—" if v is None else f"${v * 1000:.4f}"


_CSV_FIELDS = [
    "experiment", "n", "answered", "errors", "judged", "judge_errors", "mean_quality",
    "mean_tokens", "mean_cost_usd", "mean_latency_ms", "q_per_1k_tokens", "q_per_usd",
    "cache_hits", "wrong_cache_hits", "compressed", "mean_tokens_saved", "mean_tokens_amortized", "mean_cost_amortized", "exact_items", "exact_acc",
    "judge_exact_disagreements", "judge_cost_usd", "tier_mix",
]  # fmt: skip


def write_report(
    results: list[ItemResult], settings: Settings, out_dir: Path, meta: dict[str, str] | None = None
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    low_q = settings.eval.low_quality
    summaries = summarize(results, low_q)
    paths = {
        "results": out_dir / "results.jsonl",
        "summary_csv": out_dir / "summary.csv",
        "frontier_tokens": out_dir / "frontier_tokens.png",
        "frontier_cost": out_dir / "frontier_cost.png",
        "report": out_dir / "report.md",
    }

    paths["results"].write_text("".join(r.model_dump_json() + "\n" for r in results))

    with paths["summary_csv"].open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for s in summaries:
            row = s.model_dump(include=set(_CSV_FIELDS) - {"tier_mix"})
            row["tier_mix"] = ";".join(f"{k}:{v}" for k, v in sorted(s.tier_mix.items()))
            w.writerow(row)

    _frontier_plot(
        summaries, lambda s: s.mean_tokens, "mean tokens per query (LLM in + out + embedding)",
        lambda v: f"{v:,.0f}", "Quality vs. tokens", paths["frontier_tokens"],
    )  # fmt: skip
    _frontier_plot(
        summaries,
        lambda s: s.mean_cost_usd * 1000 if s.mean_cost_usd is not None else None,
        "system cost per 1k queries (USD, judge excluded)",
        lambda v: f"${v:,.3g}", "Quality vs. cost", paths["frontier_cost"],
    )  # fmt: skip

    if any(r.compressed for r in results):
        # Second view: compressor overhead amortized over each conversation's turns.
        paths["frontier_tokens_amortized"] = out_dir / "frontier_tokens_amortized.png"
        paths["frontier_cost_amortized"] = out_dir / "frontier_cost_amortized.png"
        _frontier_plot(
            summaries, lambda s: s.mean_tokens_amortized,
            "mean tokens per query, compressor overhead amortized over the conversation",
            lambda v: f"{v:,.0f}", "Quality vs. tokens (amortized)", paths["frontier_tokens_amortized"],
        )  # fmt: skip
        _frontier_plot(
            summaries,
            lambda s: s.mean_cost_amortized * 1000 if s.mean_cost_amortized is not None else None,
            "system cost per 1k queries, compressor overhead amortized (USD)",
            lambda v: f"${v:,.3g}", "Quality vs. cost (amortized)", paths["frontier_cost_amortized"],
        )  # fmt: skip

    paths["report"].write_text(_markdown(results, summaries, settings, meta or {}))
    return paths


def _markdown(
    results: list[ItemResult], summaries: list[ExperimentSummary], settings: Settings, meta: dict[str, str]
) -> str:
    low_q = settings.eval.low_quality
    items = len({r.item_id for r in results})
    judge_total = sum(s.judge_cost_usd for s in summaries)
    lines = [
        "# Eval report: quality per token",
        "",
        f"- generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- dataset: `{meta.get('dataset', settings.eval.dataset)}` ({items} items)",
        f"- judge: `{settings.judge.model}` (temperature {settings.judge.temperature}), "
        f"rubric {RUBRIC_VERSION}; judge cost ${judge_total:.4f} (not charged to any config)",
        f"- wrong cache hit = cache hit with quality < {low_q}",
        *[f"- {k}: {v}" for k, v in meta.items() if k != "dataset"],
        "",
        "## Summary",
        "",
        "| experiment | quality | tokens / query | $ / 1k queries | quality / 1k tokens | quality / $ "
        "| cache hits (wrong) | tier mix | answered / n | judge errors |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        mix = ", ".join(f"{k} {v}" for k, v in sorted(s.tier_mix.items()))
        lines.append(
            f"| {s.experiment} | {_fmt(s.mean_quality, '.3f')} | {_fmt(s.mean_tokens, ',.0f')} "
            f"| {_usd_per_1k(s.mean_cost_usd)} | {_fmt(s.q_per_1k_tokens, '.3f')} "
            f"| {_fmt(s.q_per_usd, ',.0f')} | {s.cache_hits} ({s.wrong_cache_hits}) | {mix} "
            f"| {s.answered} / {s.n} | {s.judge_errors} |"
        )
    lines += [
        "",
        "![Quality vs. tokens](frontier_tokens.png)",
        "",
        "![Quality vs. cost](frontier_cost.png)",
        "",
        "## By difficulty",
        "",
        "| experiment | " + " | ".join(f"{d} quality | {d} tokens | {d} $/1k" for d in _DIFFICULTY_ORDER) + " |",
        "|---|" + "---|---|---|" * len(_DIFFICULTY_ORDER),
    ]
    for s in summaries:
        cells = []
        for d in _DIFFICULTY_ORDER:
            b = s.by_difficulty.get(d)
            cells += (
                [_fmt(b.mean_quality, ".3f"), _fmt(b.mean_tokens, ",.0f"), _usd_per_1k(b.mean_cost_usd)]
                if b
                else ["—", "—", "—"]
            )
        lines.append(f"| {s.experiment} | " + " | ".join(cells) + " |")

    hits = [r for r in results if r.cache_status == CacheStatus.HIT and r.error is None]
    lines += ["", "## Cache hits", ""]
    if hits:
        lines += ["| experiment | item | tags | similarity | judge score | wrong? |", "|---|---|---|---|---|---|"]
        for r in hits:
            wrong = "**yes**" if _is_wrong_hit(r, low_q) else "no"
            lines.append(
                f"| {r.experiment} | {r.item_id} | {', '.join(r.tags) or '—'} "
                f"| {_fmt(r.cache_sim, '.4f')} | {r.judge_score or '—'} | {wrong} |"
            )
    else:
        lines.append("No cache hits.")

    if any(s.compressed for s in summaries):
        lines += [
            "",
            "## Context compression",
            "",
            "Two views of the same runs. **One-shot**: each eval item pays for summarizing/"
            "embedding its whole older history at once (worst case). **Amortized**: the "
            "compressor is replayed turn by turn through the conversation (rolling summary and "
            "embeddings reused, as in live chat) and its overhead averaged per request.",
            "",
            "| experiment | compressed / answered | history tokens saved / query (gross) "
            "| summarizer tokens / query | tokens / query (one-shot) | tokens / query (amortized) "
            "| $ / 1k (one-shot) | $ / 1k (amortized) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in summaries:
            rows = [r for r in results if r.experiment == s.experiment and r.error is None]
            summ = _mean([float(r.summary_tokens) for r in rows])
            lines.append(
                f"| {s.experiment} | {s.compressed} / {s.answered} | {_fmt(s.mean_tokens_saved, ',.0f')} "
                f"| {_fmt(summ, ',.0f')} | {_fmt(s.mean_tokens, ',.0f')} "
                f"| {_fmt(s.mean_tokens_amortized, ',.0f')} | {_usd_per_1k(s.mean_cost_usd)} "
                f"| {_usd_per_1k(s.mean_cost_amortized)} |"
            )
        lines += [
            "",
            "![Quality vs. tokens (amortized)](frontier_tokens_amortized.png)",
            "",
            "![Quality vs. cost (amortized)](frontier_cost_amortized.png)",
        ]
        amortize_errors = [r for r in results if r.amortize_error]
        if amortize_errors:
            lines += ["", "Amortization failed (one-shot values used) for: " + ", ".join(
                f"{r.experiment}/{r.item_id}" for r in amortize_errors)]  # fmt: skip

    lines += [
        "",
        "## Judge sanity check (exact match)",
        "",
        "Items with a short canonical answer also get a lenient deterministic check. "
        "Disagreement = exact match but judge ≤ 4, or no exact match but judge ≥ 8.",
        "",
        "| experiment | exact items | exact-match rate | disagreements |",
        "|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.experiment} | {s.exact_items} | {_fmt(s.exact_acc, '.2f')} | {s.judge_exact_disagreements} |"
        )
    disagree = [r for r in results if _disagrees(r)]
    if disagree:
        lines += ["", "Disagreements to inspect:", ""]
        for r in disagree:
            lines.append(
                f"- {r.experiment} / {r.item_id}: exact={r.exact_match} "
                f"(expected `{r.exact_expected}`), judge={r.judge_score} — {r.rationale}"
            )

    failures = [r for r in results if r.error or r.judge_error]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"- {r.experiment} / {r.item_id}: {r.error or 'judge: ' + str(r.judge_error)}")

    lines += [
        "",
        "## Definitions",
        "",
        "- **quality**: judge score (1–10) mapped to 0–1 as (score − 1) / 9; mean over judged items.",
        "- **tokens / query**: LLM input + output (incl. thinking) + estimated embedding tokens "
        "+ summarizer tokens; mean over answered items. A cache hit spends 0 LLM tokens.",
        "- **amortized**: compressor overhead (summarizer + retrieval embeddings) replayed turn "
        "by turn through each conversation and divided by its number of requests, plus the final "
        "request's own answer tokens/cost. Equal to one-shot for items without compression.",
        "- **tokens saved**: history tokens removed by the compressor (estimate, gross); the "
        "summarizer's own tokens are already in tokens / query, so that column is net.",
        "- **$ / 1k queries**: system cost (LLM + embeddings) at configured prices; judge cost excluded.",
        "- **quality / 1k tokens** = quality ÷ (tokens/query ÷ 1000); **quality / $** = quality ÷ $/query.",
        "- Failed requests (after retries) are excluded from the means and listed under Failures.",
        "- Frontier: a config is dominated if another is at least as good and no more expensive on that axis.",
        "",
    ]
    return "\n".join(lines)
