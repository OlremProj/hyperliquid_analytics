"""Tests pour AnalyticsService."""

from datetime import datetime
import pytest

from hyperliquid_analytics.services.analytics_service import AnalyticsService


@pytest.fixture
def service(monkeypatch):
    captures: dict[str, object] = {"to_thread_calls": []}

    def fake_repo_init(self, db_path=None):
        captures["repo_init_db_path"] = db_path

    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.PerpRepository.__init__",
        fake_repo_init,
    )
    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.HyperliquidClient.__init__",
        lambda self: None,
    )

    async def fake_fetch_meta(self):
        captures["fetch_meta_called"] = True
        return captures["meta_response"]

    def fake_save_snapshot(self, snapshot, fetched_at):
        captures["save_snapshot"] = (snapshot, fetched_at)
        return captures.get("save_snapshot_result")

    def fake_fetch_latest(self, symbol):
        captures["fetch_latest_called_with"] = symbol
        return captures["latest_response"]

    def fake_fetch_history(self, symbol, *, since=None, limit=None, ascending=False):
        captures["fetch_history_called_with"] = {
            "symbol": symbol,
            "since": since,
            "limit": limit,
            "ascending": ascending,
        }
        return captures["history_response"]

    async def fake_to_thread(func, *args, **kwargs):
        captures["to_thread_calls"].append((func, args, kwargs))
        return func(*args, **kwargs)

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
    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.PerpRepository.fetch_history",
        fake_fetch_history,
    )
    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.asyncio.to_thread",
        fake_to_thread,
    )

    captures["meta_response"] = {"ok": True}
    captures["latest_response"] = {"symbol": "BTC"}
    captures["history_response"] = [("ts", {"symbol": "BTC"})]

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
    assert captures["save_snapshot"] == (expected, result_timestamp)
    assert len(captures["to_thread_calls"]) == 1
    func, args, kwargs = captures["to_thread_calls"][0]
    assert func.__name__ == "fake_save_snapshot"
    assert args[0] is expected
    assert kwargs == {}


@pytest.mark.asyncio
async def test_get_market_data_delegates_to_repository(service):
    svc, captures = service

    expected = {"symbol": "ETH"}
    captures["latest_response"] = expected

    result = await svc.get_market_data("eth")

    assert result is expected
    assert captures["fetch_latest_called_with"] == "eth"
    assert len(captures["to_thread_calls"]) == 1
    func, args, kwargs = captures["to_thread_calls"][0]
    assert func.__name__ == "fake_fetch_latest"
    assert args == ("eth",)
    assert kwargs == {}


@pytest.mark.asyncio
async def test_get_market_history_delegates_to_repository(service):
    svc, captures = service

    history_payload = [("ts", {"symbol": "ETH"})]
    captures["history_response"] = history_payload

    result = await svc.get_market_history("eth", limit=5, ascending=True)

    assert result is history_payload
    called_with = captures["fetch_history_called_with"]
    assert called_with["symbol"] == "eth"
    assert called_with["limit"] == 5
    assert called_with["ascending"] is True
    assert len(captures["to_thread_calls"]) == 1
    func, args, kwargs = captures["to_thread_calls"][0]
    assert func.__name__ == "fake_fetch_history"
    assert args == ("eth",)
    assert kwargs == {"limit": 5, "since": None, "ascending": True}

