"""Request pipeline. So far: semantic cache -> router -> provider -> trace.

The compressor slots in between cache and router in a later milestone.
"""

import time
import uuid
from datetime import UTC, datetime

from gpt_efficient.cache import CacheEntry, SemanticCache, cache_namespace, estimate_tokens
from gpt_efficient.config import Settings
from gpt_efficient.providers.base import Embedder, LLMProvider
from gpt_efficient.router import Router, build_router
from gpt_efficient.schemas import CacheStatus, Message, Response, TraceRow
from gpt_efficient.trace import TraceLogger, query_hash


class Engine:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, LLMProvider],
        logger: TraceLogger,
        embedder: Embedder | None = None,
        cache: SemanticCache | None = None,
        router: Router | None = None,
    ) -> None:
        if settings.cache.enabled and (embedder is None or cache is None):
            raise ValueError("cache.enabled requires an embedder and a SemanticCache")
        self.settings = settings
        self.providers = providers
        self.logger = logger
        self.embedder = embedder
        self.cache = cache
        self.router = router or build_router(settings)
        self.namespace = cache_namespace(settings)

    def ask(self, query: str, history: list[Message] | None = None) -> Response:
        """Answer one query. Always logs exactly one trace row, even on failure."""
        # Placeholder until the router runs (it only runs on a cache miss).
        tier = self.settings.default_tier
        target = self.settings.target(tier)
        row = TraceRow(
            id=uuid.uuid4().hex,
            ts=datetime.now(UTC),
            query_hash=query_hash(query),
            cache_status=CacheStatus.DISABLED,
            tier=tier,
            provider=target.provider,
            model=target.model,
        )
        start = time.perf_counter()
        try:
            text = self._answer(query, history or [], row)
        except Exception as exc:
            row.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row.latency_ms = (time.perf_counter() - start) * 1000
            row.cost_usd += self.settings.embedding_cost_usd(row.embed_tokens)
            self.logger.log(row)
        return Response(text=text, trace_id=row.id, tier=row.tier, model=row.model)

    def _answer(self, query: str, history: list[Message], row: TraceRow) -> str:
        cfg = self.settings.cache
        vector = None
        if cfg.enabled and history:
            # A cached answer to a standalone query may be wrong in context.
            row.cache_status = CacheStatus.BYPASS
        elif cfg.enabled:
            assert self.embedder is not None and self.cache is not None
            row.cache_status = CacheStatus.MISS  # until a match proves otherwise
            embed_text = cfg.embed_template.format(text=query)
            [vector] = self.embedder.embed([embed_text])
            row.embed_tokens = estimate_tokens(embed_text, self.settings.embedding_chars_per_token)
            match = self.cache.nearest(vector, self.namespace)
            row.cache_sim = match.similarity if match else None
            if match and match.similarity >= cfg.threshold:
                entry = match.entry
                row.cache_status = CacheStatus.HIT
                row.tier, row.provider, row.model = entry.tier, entry.provider, entry.model
                row.response_len = len(entry.response)
                return entry.response

        decision = self.router.route(query, history)
        target = self.settings.target(decision.tier)
        row.tier, row.provider, row.model = decision.tier, target.provider, target.model

        messages = [
            Message(role="system", content=self.settings.system_prompt),
            *history,
            Message(role="user", content=query),
        ]
        completion = self.providers[row.provider].complete(
            messages, max_tokens=self.settings.max_tokens, model=row.model
        )
        row.model = completion.model
        row.tokens_in = completion.tokens_in
        row.tokens_out = completion.tokens_out
        row.cost_usd = completion.cost_usd
        row.response_len = len(completion.text)

        if vector is not None and completion.text:
            assert self.cache is not None
            self.cache.store(
                vector,
                self.namespace,
                CacheEntry(
                    query=query,
                    response=completion.text,
                    tier=row.tier,
                    provider=row.provider,
                    model=row.model,
                ),
            )
        return completion.text
