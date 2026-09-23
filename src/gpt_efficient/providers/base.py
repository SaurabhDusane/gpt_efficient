"""Provider-neutral interfaces every adapter implements.

Completions and embeddings are separate protocols on purpose: the embedding
backend may differ from the completion provider (e.g. local embeddings with a
hosted LLM), so the cache/router/compressor depend only on `Embedder`.
"""

from typing import Protocol, runtime_checkable

from gpt_efficient.schemas import Completion, Message, Vector


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, messages: list[Message], max_tokens: int, model: str) -> Completion:
        """Run one completion. A leading `system` message is the system prompt."""
        ...


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[Vector]:
        """Embed each text; returns one vector per input, in order."""
        ...
