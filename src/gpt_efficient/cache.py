"""Semantic cache: nearest-neighbour lookup of past answers by query embedding.

The store only holds and searches vectors; the engine owns embedding and the
hit/miss decision (similarity >= configured threshold), so everything here is
testable without an embedder or API key.
"""

import hashlib
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec
from pydantic import BaseModel, Field

from gpt_efficient.config import Settings
from gpt_efficient.schemas import Tier, Vector


class CacheEntry(BaseModel):
    query: str
    response: str
    tier: Tier
    provider: str
    model: str
    created_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CacheMatch(BaseModel):
    entry: CacheEntry
    similarity: float


def cache_namespace(settings: Settings) -> str:
    """Hash of every setting that changes what answer a query should get.

    Entries are only matched within one namespace, so changing any of these
    (system prompt, tier policy, embedding space, ...) can never serve an
    answer produced under the old configuration.
    """
    key = {
        "version": settings.cache.version,
        "system_prompt": settings.system_prompt,
        "max_tokens": settings.max_tokens,
        "default_tier": settings.default_tier,
        "tiers": {t: settings.target(t).model_dump() for t in settings.active_tiers},
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "embed_template": settings.cache.embed_template,
        # The router decides which tier answers, so its config changes answers.
        "router": _router_key(settings),
    }
    blob = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _router_key(settings: Settings) -> dict[str, object]:
    r = settings.router
    key: dict[str, object] = {"type": r.type}
    if r.type == "heuristic":  # only the active router's params matter
        key["heuristic"] = r.heuristic.model_dump(mode="json")
    return key


def estimate_tokens(text: str, chars_per_token: float) -> int:
    """Rough token count for APIs that don't report one (e.g. Gemini embeddings)."""
    return math.ceil(len(text) / chars_per_token) if text else 0


class SemanticCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY,
                namespace TEXT NOT NULL,
                entry TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    @staticmethod
    def _vec_table(dim: int) -> str:
        # vec0 columns have a fixed dimension, so each dimension gets its own table.
        return f"cache_vec_{dim}"

    def _has_table(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", [name]
        ).fetchone()
        return row is not None

    def nearest(self, vector: Vector, namespace: str) -> CacheMatch | None:
        table = self._vec_table(len(vector))
        if not self._has_table(table):
            return None
        row = self._conn.execute(
            f"""SELECT e.entry, v.distance
                FROM {table} v JOIN cache_entries e ON e.id = v.rowid
                WHERE v.embedding MATCH ? AND k = 1 AND v.namespace = ?""",
            [sqlite_vec.serialize_float32(vector), namespace],
        ).fetchone()
        if row is None:
            return None
        entry_json, distance = row
        return CacheMatch(
            entry=CacheEntry.model_validate_json(entry_json),
            similarity=1.0 - distance,  # vec0 cosine distance = 1 - cosine similarity
        )

    def store(self, vector: Vector, namespace: str, entry: CacheEntry) -> None:
        table = self._vec_table(len(vector))
        with self._conn:
            self._conn.execute(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(
                    namespace TEXT PARTITION KEY,
                    embedding float[{len(vector)}] distance_metric=cosine
                )"""
            )
            cur = self._conn.execute(
                "INSERT INTO cache_entries (namespace, entry) VALUES (?, ?)",
                [namespace, entry.model_dump_json()],
            )
            self._conn.execute(
                f"INSERT INTO {table} (rowid, namespace, embedding) VALUES (?, ?, ?)",
                [cur.lastrowid, namespace, sqlite_vec.serialize_float32(vector)],
            )
