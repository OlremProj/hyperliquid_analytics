"""Tests pour AnalyticsService."""

from datetime import datetime
import pytest

from hyperliquid_analytics.services.analytics_service import AnalyticsService


@pytest.fixture
def service(monkeypatch):
    captures: dict[str, object] = {}

    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.PerpRepository.__init__",
        lambda self: None,
    )
    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.HyperliquidClient.__init__",
        lambda self: None,
    )

    async def fake_fetch_meta(self):
        return captures["meta_response"]

    def fake_save_snapshot(self, snapshot, fetched_at):
        captures["save_snapshot"] = (snapshot, fetched_at)

    def fake_fetch_latest(self, symbol):
        captures["fetch_latest_called_with"] = symbol
        return captures["latest_response"]

    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.HyperliquidClient.fetch_meta_and_asset_contexts",
        fake_fetch_meta,
    )
    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.PerpRepository.save_snapshot",
        fake_save_snapshot,
    )
    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.PerpRepository.fetch_latest",
        fake_fetch_latest,
    )

    captures["meta_response"] = {"ok": True}
    captures["latest_response"] = {"symbol": "BTC"}

    svc = AnalyticsService()
    return svc, captures


@pytest.mark.asyncio
async def test_save_market_data_uses_repository(service):
    svc, captures = service

    expected = {"ok": True}
    captures["meta_response"] = expected

    result_snapshot, result_timestamp = await svc.save_market_data()
    assert result_snapshot is expected
    assert isinstance(result_timestamp, datetime)


@pytest.mark.asyncio
async def test_get_market_data_delegates_to_repository(service):
    svc, captures = service

    expected = {"symbol": "ETH"}
    captures["latest_response"] = expected

    result = await svc.get_market_data("eth")

    assert result is expected
    assert captures["fetch_latest_called_with"] == "eth"

