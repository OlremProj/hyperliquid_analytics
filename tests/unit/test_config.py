"""Tests pour la configuration de l’agent."""

import pytest
from pydantic import ValidationError

import src.hyperliquid_analytics.config as config_module


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Nettoie le cache `get_settings` avant chaque test."""
    config_module.get_settings.cache_clear()
    yield
    config_module.get_settings.cache_clear()


def test_settings_load_from_env(monkeypatch):
    """Les variables d’environnement sont lues et normalisées correctement."""
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_API_KEY", "secret-key")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_BASE_URL", "https://api.hyperliquid.xyz")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_SYMBOLS_RAW", "btc, eth ")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_LOG_LEVEL", "debug")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_ENABLE_AUTO_TRADING", "false")

    settings = config_module.Settings()
    print("symbols : ")
    print(settings)
    assert settings.api_key == "secret-key"
    assert str(settings.base_url) == "https://api.hyperliquid.xyz/"
    assert settings.symbols == ["BTC", "ETH"]
    assert settings.log_level == "DEBUG"
    assert settings.enable_auto_trading is False


def test_symbols_cannot_be_empty(monkeypatch):
    """Au moins un symbole doit être fourni."""
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_API_KEY", "secret")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_BASE_URL", "https://api.hyperliquid.xyz")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_SYMBOLS_RAW", "  ")

    with pytest.raises(ValidationError):
        config_module.Settings()


def test_log_level_validation(monkeypatch):
    """Le niveau de log doit faire partie de la liste autorisée."""
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_API_KEY", "secret")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_BASE_URL", "https://api.hyperliquid.xyz")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_SYMBOLS_RAW", "BTC")

    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_LOG_LEVEL", "INVALID")
    with pytest.raises(ValidationError):
        config_module.Settings()


def test_auto_trading_cannot_be_enabled(monkeypatch):
    """Forcer enable_auto_trading à true doit lever une erreur explicite."""
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_API_KEY", "secret")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_BASE_URL", "https://api.hyperliquid.xyz")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_SYMBOLS_RAW", "BTC")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_ENABLE_AUTO_TRADING", "true")

    with pytest.raises(ValidationError) as exc:
        config_module.Settings()

    assert "must remain disabled" in str(exc.value)


def test_get_settings_returns_singleton(monkeypatch):
    """`get_settings` doit renvoyer la même instance (cache)."""
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_API_KEY", "secret")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_BASE_URL", "https://api.hyperliquid.xyz")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_SYMBOLS_RAW", "BTC")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_LOG_LEVEL", "INFO")
    monkeypatch.setenv("HYPERLIQUID_ANALYTICS_ENABLE_AUTO_TRADING", "false")

    settings_a = config_module.get_settings()
    settings_b = config_module.get_settings()

    assert settings_a is settings_b
