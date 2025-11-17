from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pytest

from hyperliquid_analytics.models.perp_models import PerpAssetContext
from hyperliquid_analytics.repository.perp_repository import PerpRepository
from hyperliquid_analytics.services.indicator_service import (
    IndicatorResult,
    IndicatorService,
    IndicatorType,
)


class _AnalyticsStub:
    def __init__(self, repo: PerpRepository) -> None:
        self.perp_repository = repo


@pytest.fixture
def indicator_service(tmp_path: Path) -> Iterable[IndicatorService]:
    db_path = tmp_path / "indicator.duckdb"
    repo = PerpRepository(db_path=db_path)
    service = IndicatorService(_AnalyticsStub(repo), repo=repo)
    try:
        yield service
    finally:
        repo.close()


def _make_context(mark_price: float) -> PerpAssetContext:
    return PerpAssetContext.model_validate(
        {
            "dayNtlVlm": "1000",
            "funding": "0",
            "impactPxs": None,
            "markPx": str(mark_price),
            "midPx": None,
            "openInterest": "1",
            "oraclePx": str(mark_price),
            "premium": None,
            "prevDayPx": None,
        }
    )


def _populate_prices(
    repo: PerpRepository,
    *,
    symbol: str,
    base_time: datetime,
    prices: Iterable[float],
) -> None:
    for index, price in enumerate(prices):
        fetched_at = base_time + timedelta(minutes=index)
        repo.save_asset_contexts([(symbol, _make_context(price))], fetched_at=fetched_at)


class RecordingConnection:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.last_query: str | None = None
        self.last_params: list | None = None

    def execute(self, query, params):
        self.last_query = query
        self.last_params = list(params)
        return self._inner.execute(query, params)

    def close(self):
        return self._inner.close()

    def __getattr__(self, item):
        return getattr(self._inner, item)


@pytest.mark.asyncio
async def test_compute_indicator_sma_uses_db_and_returns_result(monkeypatch, indicator_service: IndicatorService):
    series = [
        (datetime(2024, 1, 1, tzinfo=timezone.utc), 100.0),
        (datetime(2024, 1, 2, tzinfo=timezone.utc), 105.5),
    ]
    captured: dict[str, object] = {}

    async def fake_to_thread(func: Callable, *args, **kwargs):
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        return series

    monkeypatch.setattr(
        "hyperliquid_analytics.services.indicator_service.asyncio.to_thread",
        fake_to_thread,
    )

    result = await indicator_service.compute_indicator(
        "btc",
        IndicatorType.SMA,
        window=3,
        limit=5,
    )

    assert captured["func"] == indicator_service._compute_sma_db
    assert captured["args"] == ("btc", 3, 5)
    assert isinstance(result, IndicatorResult)
    assert result.symbol == "BTC"
    assert result.indicator is IndicatorType.SMA
    assert result.params == {"window": 3, "limit": 5}
    assert result.series == series


@pytest.mark.asyncio
@pytest.mark.parametrize("window", [None, 0, -1])
async def test_compute_indicator_rejects_invalid_window(window, indicator_service: IndicatorService):
    with pytest.raises(ValueError):
        await indicator_service.compute_indicator(
            "btc",
            IndicatorType.SMA,
            window=window,
        )


def test_compute_sma_db_returns_full_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _populate_prices(
        indicator_service._repo,
        symbol="BTC",
        base_time=base_time,
        prices=[100, 110, 120, 130, 140],
    )
    recording_conn = RecordingConnection(indicator_service._repo._conn)
    indicator_service._repo._conn = recording_conn

    series = indicator_service._compute_sma_db("btc", window=3)

    assert recording_conn.last_query is not None
    assert "LIMIT ?" not in recording_conn.last_query
    assert recording_conn.last_params == ["BTC"]

    timestamps = [timestamp for timestamp, _ in series]
    values = [value for _, value in series]

    assert timestamps == sorted(timestamps)
    assert values[:2] == [None, None]
    assert values[2:] == pytest.approx([110.0, 120.0, 130.0])


def test_compute_sma_db_applies_limit(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _populate_prices(
        indicator_service._repo,
        symbol="BTC",
        base_time=base_time,
        prices=[200, 210, 220, 230, 240],
    )
    recording_conn = RecordingConnection(indicator_service._repo._conn)
    indicator_service._repo._conn = recording_conn

    series = indicator_service._compute_sma_db("btc", window=3, limit=3)

    assert "LIMIT ?" in recording_conn.last_query
    assert recording_conn.last_params == ["BTC", 3]
    assert len(series) == 3
    assert [value for _, value in series] == [None, None, pytest.approx(210.0)]


def test_compute_sma_db_rejects_invalid_inputs(indicator_service: IndicatorService):
    with pytest.raises(ValueError):
        indicator_service._compute_sma_db("", window=3)
    with pytest.raises(ValueError):
        indicator_service._compute_sma_db("btc", window=0)
    with pytest.raises(ValueError):
        indicator_service._compute_sma_db("btc", window=3, limit=0)

