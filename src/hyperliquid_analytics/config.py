"""Gestion centralisée de la configuration de l’agent Hyperliquid."""

from functools import lru_cache
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="HYPERLIQUID_ANALYTICS_",
        extra="ignore",
    )

    api_key: str = Field(default=False)
    base_url: str = Field(default=False)
    symbols_raw: str = Field(default=False)
    log_level: str = Field(default="INFO")
    enable_auto_trading: bool = Field(default=False)

    @computed_field(return_type=list[str])
    def symbols(self) -> list[str]:
        raw = self.symbols_raw.split(",")
        symbols = [item.strip().upper() for item in raw if item.strip()]
        if not symbols:
            raise ValueError("At least one symbol must be provided.")
        return symbols

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"Invalid log level '{value}'.")
        return normalized

    @field_validator("enable_auto_trading")
    @classmethod
    def forbid_auto_trading(cls, value: bool) -> bool:
        if value:
            raise ValueError("Auto trading must remain disabled (set to false).")
        return value

    @field_validator("symbols_raw")
    @classmethod
    def symbols_raw_cannot_be_empty(cls, value: str):
        if not value or value.isspace():
            raise ValueError("At lease one symbol must ne provided")
        return value

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne la configuration de l’agent (mise en cache)."""
    return Settings()
