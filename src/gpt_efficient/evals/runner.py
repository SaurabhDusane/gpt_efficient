"""Runs a dataset through each experiment config and judges every answer.

An experiment is config.toml plus a dict of overrides. Each experiment gets a
fresh Engine with its own trace DB and cache DB under `<out_dir>/<name>/`, so
cache hits come only from repeats/paraphrases earlier in the same run.
"""

import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gpt_efficient.cache import SemanticCache
from gpt_efficient.compressor import replay_overhead
from gpt_efficient.config import Settings
from gpt_efficient.engine import Engine
from gpt_efficient.evals.dataset import EvalItem
from gpt_efficient.evals.judge import Judge, exact_match, quality_from_score
from gpt_efficient.providers.base import Embedder, LLMProvider
from gpt_efficient.schemas import CacheStatus, Tier, TraceRow
from gpt_efficient.trace import TraceLogger


class Experiment(BaseModel):
    name: str
    description: str = ""
    overrides: dict[str, Any] = {}


class ItemResult(BaseModel):
    experiment: str
    item_id: str
    difficulty: str
    category: str
    tags: list[str] = []
    answer: str | None = None
    error: str | None = None  # the system failed to answer (after retries)
    # from the request's trace row
    tier: Tier
    provider: str
    model: str
    cache_status: CacheStatus
    cache_sim: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    embed_tokens: int = 0
    summary_tokens: int = 0
    compressed: bool = False
    tokens_saved: int = 0
    route_confidence: float | None = None
    escalated: bool = False
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    # judging
    judge_score: int | None = None
    quality: float | None = None  # judge score mapped to 0-1
    rationale: str = ""
    judge_error: str | None = None
    exact_expected: str | None = None
    exact_match: bool | None = None
    judge_tokens_in: int = 0
    judge_tokens_out: int = 0
    judge_cost_usd: float = 0.0
    # Tokens / cost per request with compressor overhead averaged over the whole
    # conversation (turn-by-turn replay). Equal to the one-shot values when there
    # is nothing to amortize.
    amortized_tokens: float | None = None
    amortized_cost_usd: float | None = None
    amortize_error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out + self.embed_tokens + self.summary_tokens


def load_experiments(path: Path) -> list[Experiment]:
    data = tomllib.loads(Path(path).read_text())
    experiments = [Experiment.model_validate(e) for e in data.get("experiment", [])]
    if not experiments:
        raise ValueError(f"{path}: no [[experiment]] entries")
    names = [e.name for e in experiments]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate experiment names {dupes}")
    return experiments


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(base: Settings, overrides: dict[str, Any]) -> Settings:
    """base settings + overrides (nested tables merge key by key), re-validated."""
    return Settings.model_validate(_deep_merge(base.model_dump(), overrides))


def _with_retries[T](fn: Callable[[], T], retries: int, backoff_s: float, sleep: Callable[[float], None]) -> T:
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == retries:
                raise
            sleep(backoff_s * 2**attempt)
    raise AssertionError("unreachable")


def _from_row(row: TraceRow) -> dict[str, Any]:
    return {
        k: getattr(row, k)
        for k in (
            "tier", "provider", "model", "cache_status", "cache_sim", "tokens_in",
            "tokens_out", "embed_tokens", "summary_tokens", "compressed", "tokens_saved",
            "route_confidence", "escalated", "cost_usd", "latency_ms",
        )
    }  # fmt: skip


def _run_item(
    experiment: str,
    item: EvalItem,
    engine: Engine,
    logger: TraceLogger,
    judge: Judge,
    settings: Settings,
    sleep: Callable[[float], None],
) -> ItemResult:
    cfg = settings.eval
    meta = {
        "experiment": experiment,
        "item_id": item.id,
        "difficulty": item.difficulty,
        "category": item.category,
        "tags": item.tags,
        "exact_expected": item.exact,
    }
    try:
        resp = _with_retries(
            lambda: engine.ask(item.query, item.history), cfg.max_retries, cfg.retry_backoff_s, sleep
        )
    except Exception as exc:
        # The engine logged a row for the failed attempt; report its fields.
        return ItemResult(**meta, **_from_row(logger.all()[-1]), error=f"{type(exc).__name__}: {exc}")

    row = logger.get(resp.trace_id)
    assert row is not None
    result = ItemResult(**meta, **_from_row(row), answer=resp.text)
    _amortize(result, item, engine, cfg.amortize_compression, cfg, sleep)
    if item.exact is not None:
        result.exact_match = exact_match(resp.text, item.exact)
    try:
        verdict = _with_retries(
            lambda: judge.judge(item, resp.text), cfg.max_retries, cfg.retry_backoff_s, sleep
        )
    except Exception as exc:
        result.judge_error = f"{type(exc).__name__}: {exc}"
        return result
    result.judge_score = verdict.score
    result.quality = quality_from_score(verdict.score) if verdict.score is not None else None
    result.rationale = verdict.rationale
    result.judge_error = verdict.error
    result.judge_tokens_in = verdict.tokens_in
    result.judge_tokens_out = verdict.tokens_out
    result.judge_cost_usd = verdict.cost_usd
    return result


def _amortize(result: ItemResult, item: EvalItem, engine: Engine, enabled: bool, cfg, sleep) -> None:
    result.amortized_tokens = float(result.total_tokens)
    result.amortized_cost_usd = result.cost_usd
    s = engine.settings
    if not (enabled and item.history and s.compressor.strategy not in ("none", "truncate")):
        return
    try:
        o = _with_retries(
            lambda: replay_overhead(s, engine.providers, engine.embedder, item.query, item.history),
            cfg.max_retries, cfg.retry_backoff_s, sleep,
        )  # fmt: skip
    except Exception as exc:
        result.amortize_error = f"{type(exc).__name__}: {exc}"
        return
    n = max(o.requests, 1)
    answer_cost = s.cost_usd(result.model, result.tokens_in, result.tokens_out)
    result.amortized_tokens = result.tokens_in + result.tokens_out + (o.summary_tokens + o.embed_tokens) / n
    result.amortized_cost_usd = answer_cost + (o.summary_cost_usd + s.embedding_cost_usd(o.embed_tokens)) / n


def run_eval(
    base: Settings,
    experiments: list[Experiment],
    items: list[EvalItem],
    out_dir: Path,
    *,
    make_providers: Callable[[Settings], dict[str, LLMProvider]],
    make_embedder: Callable[[Settings], Embedder],
    judge: Judge,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[ItemResult]:
    """Run every item through every experiment; results also stream to results.jsonl."""
    out_dir = Path(out_dir)
    # Validate every experiment's config before spending anything.
    configured = [(e, apply_overrides(base, e.overrides)) for e in experiments]
    for e, s in configured:
        if s.router.type == "learned" and not s.router.learned.model_path.exists():
            raise FileNotFoundError(
                f"experiment {e.name!r} needs a learned router model at {s.router.learned.model_path}; "
                "run `gpte router label` and `gpte router train` first"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = out_dir / "results.jsonl"

    results: list[ItemResult] = []
    for exp, settings in configured:
        exp_dir = out_dir / exp.name
        if (exp_dir / "traces.db").exists():
            raise FileExistsError(f"{exp_dir} already has results; use a fresh out_dir")
        exp_dir.mkdir(parents=True, exist_ok=True)
        settings = settings.model_copy(
            update={
                "trace_db": exp_dir / "traces.db",
                "cache": settings.cache.model_copy(update={"db": exp_dir / "cache.db"}),
            }
        )
        logger = TraceLogger(settings.trace_db)
        cache_on = settings.cache.enabled
        engine = Engine(
            settings,
            make_providers(settings),
            logger,
            embedder=make_embedder(settings) if settings.needs_embedder else None,
            cache=SemanticCache(settings.cache.db) if cache_on else None,
        )
        for i, item in enumerate(items):
            if progress:
                progress(exp.name, i, len(items))
            result = _run_item(exp.name, item, engine, logger, judge, base, sleep)
            results.append(result)
            with stream.open("a") as f:
                f.write(result.model_dump_json() + "\n")
            if base.eval.request_delay_s and i < len(items) - 1:
                sleep(base.eval.request_delay_s)
        if progress:
            progress(exp.name, len(items), len(items))
    return results
