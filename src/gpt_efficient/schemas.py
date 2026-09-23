"""Provider-agnostic data schemas shared across the pipeline."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


type Vector = list[float]


class Tier(StrEnum):
    LOCAL = "local"
    MID = "mid"
    FRONTIER = "frontier"


class CacheStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    DISABLED = "disabled"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Completion(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    model: str


class TraceRow(BaseModel):
    """One row per request. Keep in sync with SPEC.md §3.6 and CLAUDE.md."""

    id: str
    ts: datetime
    query_hash: str
    cache_status: CacheStatus
    cache_sim: float | None = None
    tier: Tier
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    compressed: bool = False
    tokens_saved: int = 0
    escalated: bool = False
    response_len: int = 0
    error: str | None = Field(default=None, description="Set when the request failed.")


class Response(BaseModel):
    text: str
    trace_id: str
    tier: Tier
    model: str
