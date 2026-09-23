"""Provider registry: maps config provider names to adapters."""

from gpt_efficient.config import Settings
from gpt_efficient.providers.base import LLMProvider


def build_providers(settings: Settings) -> dict[str, LLMProvider]:
    from gpt_efficient.providers.anthropic_provider import AnthropicProvider

    return {"anthropic": AnthropicProvider(settings)}


__all__ = ["LLMProvider", "build_providers"]
