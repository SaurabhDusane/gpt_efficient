"""Gemini adapter. The only module that imports the Gemini SDK (google-genai)."""

import time
from typing import Any

from google import genai
from google.genai import types

from gpt_efficient.config import Settings
from gpt_efficient.schemas import Completion, Message, Vector

_ROLES = {"user": "user", "assistant": "model"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        # Client construction is lazy so the unit suite never needs credentials.
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        return self._client

    def complete(self, messages: list[Message], max_tokens: int, model: str) -> Completion:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            types.Content(role=_ROLES[m.role], parts=[types.Part(text=m.content)])
            for m in messages
            if m.role != "system"
        ]
        # Note: on thinking models max_output_tokens also caps thinking tokens.
        config = types.GenerateContentConfig(
            system_instruction=system or None, max_output_tokens=max_tokens
        )

        start = time.perf_counter()
        resp = self.client.models.generate_content(model=model, contents=contents, config=config)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = resp.usage_metadata
        # prompt_token_count already includes any cached-content tokens.
        tokens_in = usage.prompt_token_count or 0
        # candidates_token_count excludes thinking; thinking is billed as output.
        tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        return Completion(
            text=resp.text or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=self._settings.cost_usd(model, tokens_in, tokens_out),
            latency_ms=latency_ms,
            # Report the requested ID (it keys the pricing table), not model_version.
            model=model,
        )


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def embed(self, texts: list[str]) -> list[Vector]:
        dim = self._settings.embedding_dim
        # A bare list of strings is aggregated into ONE embedding by
        # gemini-embedding-2; wrapping each text in its own Content yields one each.
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        resp = self.client.models.embed_content(
            model=self._settings.embedding_model,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=dim),
        )
        vectors = [[float(x) for x in e.values] for e in resp.embeddings]
        if len(vectors) != len(texts):
            raise ValueError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        if dim is not None and any(len(v) != dim for v in vectors):
            raise ValueError(f"embedding dimension mismatch: expected {dim}")
        return vectors
