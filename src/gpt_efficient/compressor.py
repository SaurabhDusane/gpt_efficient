"""Context compressor: shrink long conversation history before the LLM call.

A turn is one user+assistant exchange. When the history estimate exceeds
`compressor.trigger_tokens`, the last `keep_recent_turns` exchanges are kept
verbatim and older exchanges are replaced, per `compressor.strategy`, by:

- truncate:           nothing (dropped)
- summary:            a rolling summary written by the `summary_tier` model
- retrieval:          the `retrieve_k` older exchanges most similar to the query
- summary+retrieval:  both

Summary and excerpts are added as `system` messages ahead of the verbatim
turns, so user/assistant alternation is preserved. Splitting, estimating,
similarity and assembly are pure functions; only `Compressor` makes calls.
"""

import hashlib
import math

from pydantic import BaseModel

from gpt_efficient.cache import estimate_tokens
from gpt_efficient.config import Settings
from gpt_efficient.providers.base import Embedder, LLMProvider
from gpt_efficient.schemas import Message, Vector

SUMMARY_PROMPT = (
    "You maintain a running summary of a conversation between a user and an assistant, "
    "so that the assistant can continue the conversation without the full transcript. "
    "You are given the previous summary (possibly empty) and new exchanges. Write an updated "
    "summary that keeps every concrete detail the user might ask about later: names, numbers, "
    "dates, codes, preferences, constraints, decisions and open questions. Drop pleasantries "
    "and restated explanations. Write compact bullet points, no preamble."
)


class CompressionResult(BaseModel):
    messages: list[Message]  # the history to send (may include system context blocks)
    compressed: bool = False
    tokens_saved: int = 0  # history estimate before - after (gross; excludes summarizer cost)
    summary_tokens: int = 0  # summarizer tokens in + out
    summary_cost_usd: float = 0.0
    embed_tokens: int = 0


# --- pure helpers -----------------------------------------------------------------


def estimate_messages_tokens(messages: list[Message], chars_per_token: float) -> int:
    return sum(estimate_tokens(m.content, chars_per_token) for m in messages)


def split_exchanges(history: list[Message]) -> list[list[Message]]:
    """Group messages into exchanges, each starting at a user message."""
    exchanges: list[list[Message]] = []
    for m in history:
        if m.role == "user" or not exchanges:
            exchanges.append([m])
        else:
            exchanges[-1].append(m)
    return exchanges


def exchange_text(exchange: list[Message]) -> str:
    return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in exchange)


def _cosine(a: Vector, b: Vector) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb) if na and nb else 0.0


def top_k_indices(query: Vector, vectors: list[Vector], k: int) -> list[int]:
    """Indices of the k vectors most similar to `query`, in their original order."""
    ranked = sorted(range(len(vectors)), key=lambda i: _cosine(query, vectors[i]), reverse=True)
    return sorted(ranked[:k])


def assemble(recent: list[Message], summary: str | None, excerpts: list[str]) -> list[Message]:
    out: list[Message] = []
    if summary:
        out.append(Message(role="system", content=f"Summary of the earlier conversation:\n{summary}"))
    if excerpts:
        joined = "\n\n".join(excerpts)
        out.append(Message(role="system", content=f"Relevant earlier exchanges (verbatim):\n{joined}"))
    return out + recent


def _key(texts: list[str], model: str) -> str:
    h = hashlib.sha256(model.encode())
    for t in texts:
        h.update(b"\x00" + t.encode())
    return h.hexdigest()


class SummaryStore:
    """Summaries keyed by the exact older exchanges they cover (in-memory).

    Lets a growing conversation summarize only its new exchanges on top of the
    previous summary instead of re-summarizing everything each turn.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, str] = {}

    def longest_prefix(self, texts: list[str], model: str) -> tuple[int, str | None]:
        for n in range(len(texts), 0, -1):
            summary = self._by_key.get(_key(texts[:n], model))
            if summary is not None:
                return n, summary
        return 0, None

    def put(self, texts: list[str], model: str, summary: str) -> None:
        self._by_key[_key(texts, model)] = summary


# --- compressor -----------------------------------------------------------------


class Compressor:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, LLMProvider],
        embedder: Embedder | None = None,
        store: SummaryStore | None = None,
    ) -> None:
        self.settings = settings
        self.cfg = settings.compressor
        self.providers = providers
        self.embedder = embedder
        self.store = store or SummaryStore()
        # Exchange embeddings by text, so a growing conversation embeds each exchange once.
        self._vectors: dict[str, Vector] = {}
        if "retrieval" in self.cfg.strategy and embedder is None:
            raise ValueError(f"compressor.strategy {self.cfg.strategy!r} needs an embedder")

    def compress(self, query: str, history: list[Message]) -> CompressionResult:
        cfg = self.cfg
        before = estimate_messages_tokens(history, cfg.chars_per_token)
        exchanges = split_exchanges(history)
        if (
            cfg.strategy == "none"
            or before <= cfg.trigger_tokens
            or len(exchanges) <= cfg.keep_recent_turns
        ):
            return CompressionResult(messages=history)

        cut = len(exchanges) - cfg.keep_recent_turns
        older, recent = exchanges[:cut], [m for ex in exchanges[cut:] for m in ex]
        older_texts = [exchange_text(ex) for ex in older]
        result = CompressionResult(messages=[], compressed=True)

        summary = None
        if "summary" in cfg.strategy:
            summary = self._summarize(older_texts, result)
        excerpts: list[str] = []
        if "retrieval" in cfg.strategy:
            excerpts = self._retrieve(query, older_texts, result)

        result.messages = assemble(recent, summary, excerpts)
        result.tokens_saved = before - estimate_messages_tokens(result.messages, cfg.chars_per_token)
        return result

    def _summarize(self, older_texts: list[str], result: CompressionResult) -> str:
        target = self.settings.target(self.cfg.summary_tier)
        covered, previous = self.store.longest_prefix(older_texts, target.model)
        new = older_texts[covered:]
        if not new and previous is not None:
            return previous  # nothing new since the stored summary
        body = f"PREVIOUS SUMMARY:\n{previous or '(none)'}\n\nNEW EXCHANGES:\n" + "\n\n".join(new)
        assert target.provider is not None
        completion = self.providers[target.provider].complete(
            [Message(role="system", content=SUMMARY_PROMPT), Message(role="user", content=body)],
            max_tokens=self.cfg.summary_max_tokens,
            model=target.model,
        )
        result.summary_tokens += completion.tokens_in + completion.tokens_out
        result.summary_cost_usd += completion.cost_usd
        self.store.put(older_texts, target.model, completion.text)
        return completion.text

    def _retrieve(self, query: str, older_texts: list[str], result: CompressionResult) -> list[str]:
        assert self.embedder is not None
        new = [t for t in dict.fromkeys(older_texts) if t not in self._vectors]
        query_vec, *new_vecs = self.embedder.embed([query, *new])
        self._vectors.update(zip(new, new_vecs, strict=True))
        cpt = self.settings.embedding_chars_per_token
        result.embed_tokens += sum(estimate_tokens(t, cpt) for t in [query, *new])
        vecs = [self._vectors[t] for t in older_texts]
        return [older_texts[i] for i in top_k_indices(query_vec, vecs, self.cfg.retrieve_k)]
