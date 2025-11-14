"""Tests pour HyperliquidClient."""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Any

import pytest

from hyperliquid_analytics.api.hyperliquid_client import (
    HyperliquidAPIError,
    HyperliquidClient,
)
from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.models.perp_models import MetaAndAssetCtxsResponse


@pytest.fixture(autouse=True)
def stub_settings(monkeypatch):
    """Remplace Settings par une version minimale pour les tests."""

    class DummySettings:
        def __init__(self) -> None:
            self.base_url = "https://example.com"

    monkeypatch.setattr(
        "hyperliquid_analytics.api.hyperliquid_client.Settings", DummySettings
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
        "hyperliquid_analytics.api.hyperliquid_client.ApiClient", DummyApiClient
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
        "hyperliquid_analytics.api.hyperliquid_client.ApiClient", FailingApiClient
    )

    client = HyperliquidClient()

    with pytest.raises(HyperliquidAPIError) as exc:
        await client.fetch_ohlcv("BTC", TimeFrame.ONE_HOUR, limit=10)

    assert exc.value.status == 422
    assert "Failed to deserialize" in str(exc.value)


@pytest.mark.asyncio
async def test_fetch_meta_and_asset_contexts_success(monkeypatch):
    """Vérifie que fetch_meta_and_asset_contexts mappe correctement la réponse."""

    captured_payload: dict[str, Any] = {}

    sample_response = [
        {
            "universe": [
                {
                    "name": "BTC",
                    "szDecimals": 5,
                    "maxLeverage": "50",
                    "onlyIsolated": False,
                    "marginMode": None,
                    "isDelisted": False,
                }
            ],
            "marginTables": [
                [
                    50,
                    {
                        "description": "standard",
                        "marginTiers": [
                            {
                                "lowerBound": "0.0",
                                "maxLeverage": "50",
                            }
                        ],
                    },
                ]
            ],
        },
        [
            {
                "dayNtlVlm": "1169046.29406",
                "funding": "0.0000125",
                "impactPxs": ["14.3047", "14.3444"],
                "markPx": "14.3161",
                "midPx": "14.314",
                "openInterest": "688.11",
                "oraclePx": "14.32",
                "premium": "0.00031774",
                "prevDayPx": "15.322",
            }
        ],
    ]

    class DummyApiClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post_json(self, path: str, payload: dict[str, Any]):
            captured_payload["value"] = payload
            return sample_response

    monkeypatch.setattr(
        "hyperliquid_analytics.api.hyperliquid_client.ApiClient", DummyApiClient
    )

    client = HyperliquidClient()
    result = await client.fetch_meta_and_asset_contexts()

    assert captured_payload["value"] == {"type": "metaAndAssetCtxs"}
    assert isinstance(result, MetaAndAssetCtxsResponse)

    universe = result.meta.universe
    assert len(universe) == 1
    btc = universe[0]
    assert btc.name == "BTC"
    assert btc.sz_decimals == 5
    assert btc.max_leverage == Decimal("50")

    margin_tables = result.meta.margin_tables
    assert len(margin_tables) == 1
    assert margin_tables[0].identifier == 50
    assert margin_tables[0].table.description == "standard"
    assert margin_tables[0].table.margin_tiers[0].max_leverage == Decimal("50")

    contexts = result.asset_contexts
    assert len(contexts) == 1
    btc_ctx = contexts[0]
    assert btc_ctx.open_interest == Decimal("688.11")
    assert btc_ctx.funding == Decimal("0.0000125")
    assert btc_ctx.impact_prices == (
        Decimal("14.3047"),
        Decimal("14.3444"),
    )
    assert btc_ctx.mid_price == Decimal("14.314")
    assert btc_ctx.mark_price == Decimal("14.3161")
    assert btc_ctx.premium == Decimal("0.00031774")
    assert btc_ctx.previous_day_price == Decimal("15.322")


@pytest.mark.asyncio
async def test_fetch_meta_and_asset_contexts_handles_null_fields(monkeypatch):
    """Vérifie que les champs optionnels à None sont bien gérés."""

    sample_response = [
        {
            "universe": [
                {
                    "name": "ARB",
                    "szDecimals": 3,
                    "maxLeverage": "20",
                }
            ],
            "marginTables": [],
        },
        [
            {
                "dayNtlVlm": "1000.0",
                "funding": "0",
                "impactPxs": None,
                "markPx": "1.12",
                "midPx": None,
                "openInterest": "12.5",
                "oraclePx": "1.10",
                "premium": None,
                "prevDayPx": None,
            }
        ],
    ]

    class DummyApiClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post_json(self, path: str, payload: dict[str, Any]):
            return sample_response

    monkeypatch.setattr(
        "hyperliquid_analytics.api.hyperliquid_client.ApiClient", DummyApiClient
    )

    client = HyperliquidClient()
    result = await client.fetch_meta_and_asset_contexts()

    ctx = result.asset_contexts[0]
    assert ctx.impact_prices is None
    assert ctx.mid_price is None
    assert ctx.premium is None
    assert ctx.previous_day_price is None
