"""Tests pour HyperliquidClient."""

from datetime import datetime, timezone
from typing import Any

import pytest

from src.hyperliquid_analytics.api.hyperliquid_client import (
    HyperliquidAPIError,
    HyperliquidClient,
)
from src.hyperliquid_analytics.models.data_models import TimeFrame


@pytest.fixture(autouse=True)
def stub_settings(monkeypatch):
    """Remplace Settings par une version minimale pour les tests."""

    class DummySettings:
        def __init__(self) -> None:
            self.base_url = "https://example.com"

    monkeypatch.setattr(
        "src.hyperliquid_analytics.api.hyperliquid_client.Settings", DummySettings
    )


@pytest.mark.asyncio
async def test_fetch_ohlcv_success(monkeypatch):
    """Vérifie que fetch_ohlcv transforme la réponse brute en MarketData."""
    captured_payload: dict[str, Any] = {}

    sample_response = [
        {
            "t": 1_700_000_000_000,
            "o": "100.0",
            "h": "110.0",
            "l": "95.0",
            "c": "105.0",
            "v": "123.45",
        }
    ]

    class DummyApiClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post_json(self, path: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
            captured_payload["value"] = payload
            return sample_response

    monkeypatch.setattr(
        "src.hyperliquid_analytics.api.hyperliquid_client.ApiClient", DummyApiClient
    )

    client = HyperliquidClient()
    market_data = await client.fetch_ohlcv("btc", TimeFrame.ONE_HOUR, limit=1)

    # Vérifie le payload envoyé à l'API
    sent = captured_payload["value"]
    assert sent["type"] == "candleSnapshot"
    assert sent["req"]["coin"] == "btc"
    assert sent["req"]["interval"] == TimeFrame.ONE_HOUR.value
    assert sent["req"]["startTime"] < sent["req"]["endTime"]

    # Vérifie la transformation en MarketData
    assert market_data.symbol == "BTC"
    assert market_data.timeframe is TimeFrame.ONE_HOUR
    assert len(market_data.candles) == 1

    candle = market_data.candles[0]
    assert candle.open == 100.0
    assert candle.high == 110.0
    assert candle.low == 95.0
    assert candle.close == 105.0
    assert candle.volume == 123.45
    assert candle.timestamp == datetime.fromtimestamp(
        sample_response[0]["t"] / 1000, tz=timezone.utc
    )


@pytest.mark.asyncio
async def test_fetch_ohlcv_raises_hyperliquid_error(monkeypatch):
    """Vérifie que fetch_ohlcv propage HyperliquidAPIError."""

    class FailingApiClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post_json(self, path: str, payload: dict[str, Any]):
            raise HyperliquidAPIError(422, "Failed to deserialize")

    monkeypatch.setattr(
        "src.hyperliquid_analytics.api.hyperliquid_client.ApiClient", FailingApiClient
    )

    client = HyperliquidClient()

    with pytest.raises(HyperliquidAPIError) as exc:
        await client.fetch_ohlcv("BTC", TimeFrame.ONE_HOUR, limit=10)

    assert exc.value.status == 422
    assert "Failed to deserialize" in str(exc.value)

