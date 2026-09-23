"""Request pipeline. Milestone 1: default tier -> provider -> trace.

Cache, compressor and router slot in here in later milestones.
"""

import time
import uuid
from datetime import UTC, datetime

from gpt_efficient.config import Settings
from gpt_efficient.providers.base import LLMProvider
from gpt_efficient.schemas import CacheStatus, Completion, Message, Response, TraceRow
from gpt_efficient.trace import TraceLogger, query_hash


class Engine:
    def __init__(
        self,
        settings: Settings,
        providers: dict[str, LLMProvider],
        logger: TraceLogger,
    ) -> None:
        self.settings = settings
        self.providers = providers
        self.logger = logger

    def ask(self, query: str, history: list[Message] | None = None) -> Response:
        trace_id = uuid.uuid4().hex
        tier = self.settings.default_tier
        target = self.settings.target(tier)
        messages = [
            Message(role="system", content=self.settings.system_prompt),
            *(history or []),
            Message(role="user", content=query),
        ]
        row = TraceRow(
            id=trace_id,
            ts=datetime.now(UTC),
            query_hash=query_hash(query),
            cache_status=CacheStatus.DISABLED,
            tier=tier,
            provider=target.provider,
            model=target.model,
        )

        start = time.perf_counter()
        try:
            completion: Completion = self.providers[target.provider].complete(
                messages, max_tokens=self.settings.max_tokens, model=target.model
            )
        except Exception as exc:
            row.latency_ms = (time.perf_counter() - start) * 1000
            row.error = f"{type(exc).__name__}: {exc}"
            self.logger.log(row)
            raise

        row.model = completion.model
        row.tokens_in = completion.tokens_in
        row.tokens_out = completion.tokens_out
        row.cost_usd = completion.cost_usd
        row.latency_ms = completion.latency_ms
        row.response_len = len(completion.text)
        self.logger.log(row)
        return Response(text=completion.text, trace_id=trace_id, tier=tier, model=completion.model)
