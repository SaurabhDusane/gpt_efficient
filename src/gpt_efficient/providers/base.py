"""The single interface every provider adapter implements."""

from typing import Protocol, runtime_checkable

from gpt_efficient.schemas import Completion, Message


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, messages: list[Message], max_tokens: int, model: str) -> Completion:
        """Run one completion. A leading `system` message is the system prompt."""
        ...
