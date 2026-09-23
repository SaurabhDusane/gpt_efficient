"""Experiment configuration. Every knob an experiment varies lives here.

Values load from `config.toml` (path overridable via GPTE_CONFIG), then
GPTE_* environment variables, then explicit constructor kwargs (highest).
"""

import os
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from gpt_efficient.schemas import Tier


class TierTarget(BaseModel):
    provider: str
    model: str


class ModelPrice(BaseModel):
    input_per_mtok: float
    output_per_mtok: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GPTE_", env_nested_delimiter="__")

    default_tier: Tier = Tier.MID
    tiers: dict[Tier, TierTarget] = {}
    pricing: dict[str, ModelPrice] = {}
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

    def target(self, tier: Tier) -> TierTarget:
        try:
            return self.tiers[tier]
        except KeyError:
            raise ValueError(f"No model configured for tier {tier!r}") from None

    def cost_usd(self, model: str, tokens_in: int, tokens_out: int) -> float:
        price = self.pricing.get(model)
        if price is None:
            raise ValueError(f"No pricing configured for model {model!r}")
        return (tokens_in * price.input_per_mtok + tokens_out * price.output_per_mtok) / 1e6
