"""Anthropic adapter. The only module that imports the Anthropic SDK."""

import time
from typing import Any

import anthropic

from gpt_efficient.config import Settings
from gpt_efficient.schemas import Completion, Message


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        # Client construction is lazy so the unit suite never needs credentials.
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        messages: list[Message],
        max_tokens: int,
        model: str,
        temperature: float | None = None,
    ) -> Completion:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": turns}
        if system:
            kwargs["system"] = system
        if temperature is not None:
            # Note: current Claude models (Opus 5, Sonnet 5, Opus 4.7/4.8) reject
            # sampling params with a 400; leave temperature unset for those.
            kwargs["temperature"] = temperature

        start = time.perf_counter()
        resp = self.client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        text = "".join(b.text for b in resp.content if b.type == "text")
        usage = resp.usage
        # Count every input token the model processed, including any prompt-cache
        # reads/writes, so tokens_in is comparable across providers.
        tokens_in = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        tokens_out = usage.output_tokens
        return Completion(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._settings.cost_usd(model, tokens_in, tokens_out),
            latency_ms=latency_ms,
            model=resp.model,
        )
