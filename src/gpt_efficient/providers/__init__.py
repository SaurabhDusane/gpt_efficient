"""Provider registry: maps config provider names to adapters.

Adapters are imported lazily so an SDK is only loaded when its provider is used.
"""

from gpt_efficient.config import Settings
from gpt_efficient.providers.base import Embedder, LLMProvider


def build_providers(settings: Settings) -> dict[str, LLMProvider]:
    from gpt_efficient.providers.anthropic_provider import AnthropicProvider
    from gpt_efficient.providers.gemini_provider import GeminiProvider

    return {"anthropic": AnthropicProvider(settings), "gemini": GeminiProvider(settings)}


def build_embedder(settings: Settings) -> Embedder:
    name = settings.embedding_provider or settings.default_provider
    if name == "gemini":
        from gpt_efficient.providers.gemini_provider import GeminiEmbedder

        return GeminiEmbedder(settings)
    raise ValueError(f"No embedder available for provider {name!r}")


__all__ = ["Embedder", "LLMProvider", "build_embedder", "build_providers"]
