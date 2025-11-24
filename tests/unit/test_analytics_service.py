"""Tests pour AnalyticsService."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.services.analytics_service import AnalyticsService


@pytest.fixture
def service(monkeypatch):
    captures: dict[str, object] = {
        "to_thread_calls": [],
        "latest_candle_timestamp": None,
        "saved_candles_calls": [],
        "ohlcv_response": SimpleNamespace(candles=[]),
    }

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

    def fake_fetch_latest_candle_timestamp(self, symbol, timeframe):
        captures["fetch_latest_candle_timestamp_called_with"] = (symbol, timeframe)
        return captures.get("latest_candle_timestamp")

    def fake_save_candles_repo(self, symbol, timeframe, candles):
        captures["saved_candles_calls"].append(
            {"symbol": symbol, "timeframe": timeframe, "count": len(list(candles))}
        )

    async def fake_fetch_ohlcv(self, symbol, timeframe, limit):
        captures["fetch_ohlcv_called_with"] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
        }
        return captures["ohlcv_response"]

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
        "hyperliquid_analytics.services.analytics_service.PerpRepository.fetch_latest_candle_timestamp",
        fake_fetch_latest_candle_timestamp,
    )
    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.PerpRepository.save_candles",
        fake_save_candles_repo,
    )
    monkeypatch.setattr(
        "hyperliquid_analytics.services.analytics_service.HyperliquidClient.fetch_ohlcv",
        fake_fetch_ohlcv,
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


@pytest.mark.asyncio
async def test_save_candles_returns_up_to_date_when_gap_small(service):
    svc, captures = service
    captures["latest_candle_timestamp"] = datetime.now(timezone.utc) - timedelta(minutes=30)

    summary = await svc.save_candles("btc", TimeFrame.ONE_HOUR)

    assert summary["status"] == "up_to_date"
    assert summary["fetched"] == 0
    assert captures.get("fetch_ohlcv_called_with") is None


@pytest.mark.asyncio
async def test_save_candles_fetches_when_gap_exceeds_timeframe(service):
    svc, captures = service
    captures["latest_candle_timestamp"] = datetime.now(timezone.utc) - timedelta(hours=3)
    candles = [
        SimpleNamespace(timestamp=datetime.now(timezone.utc) - timedelta(hours=2)),
        SimpleNamespace(timestamp=datetime.now(timezone.utc) - timedelta(hours=1)),
    ]
    captures["ohlcv_response"] = SimpleNamespace(candles=candles)

    summary = await svc.save_candles("btc", TimeFrame.ONE_HOUR)

    assert summary["status"] == "updated"
    assert summary["fetched"] == len(candles)
    assert captures["fetch_ohlcv_called_with"]["limit"] == 4
    assert captures["saved_candles_calls"][-1]["symbol"] == "BTC"
    assert captures["saved_candles_calls"][-1]["count"] == len(candles)

