"""Learned router: labelling, training and the model it routes with.

Labels (pseudo-reference): every active tier answers a training query; the
judge grades each cheaper tier's answer using the top tier's answer as the
reference. The label is the cheapest tier scoring >= `label_min_score`, else
the top tier. Training fits a class-balanced multinomial logistic regression
on the (normalized) query embedding. The model is plain JSON, not a pickle.

Training queries (evals/router_train.jsonl) must stay disjoint from every eval
dataset, or the router-vs-heuristic benchmark leaks (a test enforces this).
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from gpt_efficient.config import Settings
from gpt_efficient.evals.dataset import Category, Difficulty, EvalItem
from gpt_efficient.evals.judge import Judge
from gpt_efficient.providers.base import LLMProvider
from gpt_efficient.router import rank
from gpt_efficient.schemas import Message, Tier, Vector


class TrainQuery(BaseModel):
    id: str
    query: str
    category: Category
    difficulty: Difficulty | None = None  # authoring hint only; never used as a label


def load_train_queries(path: Path) -> list[TrainQuery]:
    out: list[TrainQuery] = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = TrainQuery.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"{path}: line {n}: {exc}") from None
        if q.id in seen:
            raise ValueError(f"{path}: line {n}: duplicate id {q.id!r}")
        seen.add(q.id)
        out.append(q)
    return out


# --- labelling --------------------------------------------------------------------


class LabelRecord(BaseModel):
    id: str
    query: str
    category: str
    label: Tier
    label_tiers: list[Tier]  # the active tiers answers were collected from
    answers: dict[Tier, str] = {}
    scores: dict[Tier, int | None] = {}  # judge score of each cheaper tier vs. the top tier
    judge_model: str = ""
    tokens: int = 0
    cost_usd: float = 0.0  # answers + judge calls
    fake: bool = False  # produced offline (illustrative only)


def label_from_scores(scores: dict[Tier, int | None], tiers: list[Tier], min_score: int) -> Tier:
    """Cheapest tier whose answer scored >= min_score against the top tier; else the top tier."""
    ranked = rank(tiers)
    for tier in ranked[:-1]:
        score = scores.get(tier)
        if score is not None and score >= min_score:
            return tier
    return ranked[-1]


def label_query(
    q: TrainQuery, settings: Settings, providers: dict[str, LLMProvider], judge: Judge
) -> LabelRecord:
    tiers = rank(settings.active_tiers)
    messages = [
        Message(role="system", content=settings.system_prompt),
        Message(role="user", content=q.query),
    ]
    answers: dict[Tier, str] = {}
    tokens, cost = 0, 0.0
    for tier in tiers:
        target = settings.target(tier)
        assert target.provider is not None
        c = providers[target.provider].complete(messages, max_tokens=settings.max_tokens, model=target.model)
        answers[tier] = c.text
        tokens += c.tokens_in + c.tokens_out
        cost += c.cost_usd

    top = tiers[-1]
    pseudo = EvalItem(id=q.id, query=q.query, reference=answers[top],
                      difficulty=q.difficulty or "medium", category=q.category)  # fmt: skip
    scores: dict[Tier, int | None] = {}
    for tier in tiers[:-1]:
        verdict = judge.judge(pseudo, answers[tier])
        scores[tier] = verdict.score
        tokens += verdict.tokens_in + verdict.tokens_out
        cost += verdict.cost_usd

    return LabelRecord(
        id=q.id,
        query=q.query,
        category=q.category,
        label=label_from_scores(scores, tiers, settings.router.learned.label_min_score),
        label_tiers=tiers,
        answers=answers,
        scores=scores,
        judge_model=judge.model,
        tokens=tokens,
        cost_usd=cost,
    )


# --- model -------------------------------------------------------------------------


def _normalized(v: Vector) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


class RouterModel(BaseModel):
    classes: list[Tier]  # cheapest first
    coef: list[list[float]]  # one row per class
    intercept: list[float]
    embedding_model: str
    embedding_dim: int
    normalize: bool = True
    C: float = 1.0
    n_train: int = 0
    label_counts: dict[str, int] = {}
    cv_accuracy: float | None = None
    trained_at: datetime | None = None
    fake: bool = False  # trained on fake labels/embeddings: illustrative only

    def probabilities(self, vector: Vector) -> dict[Tier, float]:
        x = _normalized(vector) if self.normalize else list(vector)
        logits = [sum(w * xi for w, xi in zip(row, x, strict=True)) + b
                  for row, b in zip(self.coef, self.intercept, strict=True)]  # fmt: skip
        top = max(logits)
        exps = [math.exp(z - top) for z in logits]
        total = sum(exps)
        return {t: e / total for t, e in zip(self.classes, exps, strict=True)}


def train_router(xs: list[Vector], ys: list[Tier], embedding_model: str, C: float = 1.0) -> RouterModel:
    """Class-balanced multinomial logistic regression on normalized embeddings."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    classes = rank(sorted(set(ys), key=list(Tier).index))
    if len(classes) < 2:
        raise ValueError(f"need at least two distinct labels to train, got {classes}")
    X = np.array([_normalized(x) for x in xs])
    y = np.array([classes.index(t) for t in ys])

    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(X, y)
    coef, intercept = clf.coef_.tolist(), clf.intercept_.tolist()
    if len(classes) == 2:  # sklearn returns one row for binary; make it one row per class
        coef, intercept = [[0.0] * X.shape[1], coef[0]], [0.0, intercept[0]]

    counts = {t.value: int((y == i).sum()) for i, t in enumerate(classes)}
    folds = min(5, min(counts.values()))
    cv = None
    if folds >= 2:
        scores = cross_val_score(
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000), X, y,
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=0),
        )  # fmt: skip
        cv = float(scores.mean())
    return RouterModel(
        classes=classes,
        coef=coef,
        intercept=intercept,
        embedding_model=embedding_model,
        embedding_dim=X.shape[1],
        C=C,
        n_train=len(ys),
        label_counts=counts,
        cv_accuracy=cv,
        trained_at=datetime.now(UTC),
    )
