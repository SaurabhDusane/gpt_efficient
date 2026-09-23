"""Experiment configuration. Every knob an experiment varies lives here.

Values load from `config.toml` (path overridable via GPTE_CONFIG), then
GPTE_* environment variables, then explicit constructor kwargs (highest).
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from gpt_efficient.schemas import Tier


class TierTarget(BaseModel):
    model: str
    provider: str | None = None  # None -> Settings.default_provider


class ModelPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float
    # Some models charge more once the prompt exceeds a size threshold.
    long_context_threshold: int | None = None
    input_per_mtok_long: float | None = None
    output_per_mtok_long: float | None = None


class CacheConfig(BaseModel):
    enabled: bool = False
    # Cosine similarity at or above which a stored answer is served.
    threshold: float = 0.95
    db: Path = Path("data/cache.db")
    # What gets embedded for the lookup; "{text}" is the raw query.
    embed_template: str = "{text}"
    # Bump to invalidate every cached answer without touching other settings.
    version: str = "1"


class HeuristicRouterConfig(BaseModel):
    """Points per signal; the summed score is compared against tier_cutoffs."""

    long_query_words: int = 40
    very_long_query_words: int = 150
    length_points: float = 1.0  # awarded once per length threshold crossed
    keywords: list[str] = [
        "prove", "proof", "derive", "derivation", "analyze", "analyse", "compare",
        "contrast", "design", "architecture", "optimize", "optimise", "debug",
        "refactor", "trade-off", "tradeoff", "step by step", "explain why",
        "in depth", "algorithm", "complexity", "rigorous", "formally",
    ]  # fmt: skip
    keyword_points: float = 1.0  # per distinct keyword found
    max_keyword_points: float = 2.0
    code_points: float = 2.0
    math_points: float = 2.0
    # Minimum score for each tier above the cheapest; the cheapest active tier
    # is the floor. The highest active tier whose cutoff the score meets wins.
    tier_cutoffs: dict[Tier, float] = {Tier.MID: 1.0, Tier.FRONTIER: 3.0}


class RouterConfig(BaseModel):
    # "fixed" always uses default_tier (the pre-router baseline).
    type: Literal["fixed", "heuristic"] = "fixed"
    heuristic: HeuristicRouterConfig = HeuristicRouterConfig()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GPTE_", env_nested_delimiter="__")

    default_provider: str = "gemini"
    default_tier: Tier = Tier.MID
    # tier_mode picks one entry of tier_modes; that list is the active tier set.
    # With no tier_modes configured, every tier in `tiers` is active.
    tier_mode: str = "three"
    tier_modes: dict[str, list[Tier]] = {}
    tiers: dict[Tier, TierTarget] = {}
    pricing: dict[str, ModelPrice] = {}
    embedding_provider: str | None = None  # None -> default_provider
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int | None = None  # None -> model's native dimensionality
    embedding_price_per_mtok: float = 0.0
    # The embed API returns no token counts, so embed_tokens is estimated.
    embedding_chars_per_token: float = 4.0
    cache: CacheConfig = CacheConfig()
    router: RouterConfig = RouterConfig()
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = 1024
    trace_db: Path = Path("data/traces.db")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_path = Path(os.environ.get("GPTE_CONFIG", "config.toml"))
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=toml_path),
        )

    @model_validator(mode="after")
    def _check_tiers(self) -> "Settings":
        if self.tier_modes and self.tier_mode not in self.tier_modes:
            raise ValueError(
                f"tier_mode {self.tier_mode!r} not in tier_modes {sorted(self.tier_modes)}"
            )
        active = self.active_tiers
        missing = [t for t in active if t not in self.tiers]
        if missing:
            raise ValueError(f"active tiers {missing} have no entry in [tiers]")
        if self.tiers and self.default_tier not in active:
            raise ValueError(f"default_tier {self.default_tier!r} is not active in {active}")
        return self

    @property
    def active_tiers(self) -> list[Tier]:
        if self.tier_modes:
            return list(self.tier_modes[self.tier_mode])
        return list(self.tiers)

    def target(self, tier: Tier) -> TierTarget:
        if tier not in self.active_tiers:
            raise ValueError(f"Tier {tier!r} is not active (tier_mode={self.tier_mode!r})")
        t = self.tiers[tier]
        return TierTarget(model=t.model, provider=t.provider or self.default_provider)

    def cost_usd(self, model: str, tokens_in: int, tokens_out: int) -> float:
        price = self.pricing.get(model)
        if price is None:
            raise ValueError(f"No pricing configured for model {model!r}")
        rate_in, rate_out = price.input_per_mtok, price.output_per_mtok
        if price.long_context_threshold is not None and tokens_in > price.long_context_threshold:
            if price.input_per_mtok_long is not None:
                rate_in = price.input_per_mtok_long
            if price.output_per_mtok_long is not None:
                rate_out = price.output_per_mtok_long
        return (tokens_in * rate_in + tokens_out * rate_out) / 1e6

    def embedding_cost_usd(self, tokens: int) -> float:
        return tokens * self.embedding_price_per_mtok / 1e6
