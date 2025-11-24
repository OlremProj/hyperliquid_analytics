from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Iterable

import pytest

from hyperliquid_analytics.models.data_models import OHLCVData, TimeFrame
from hyperliquid_analytics.models.indicator_result_models import (
    IndicatorPoint,
    IndicatorResult,
    IndicatorType,
)
from hyperliquid_analytics.repository.perp_repository import PerpRepository
from hyperliquid_analytics.services.indicator_service import IndicatorService


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


def _make_candle(symbol: str, timestamp: datetime, price: float) -> OHLCVData:
    return OHLCVData(
        symbol=symbol.upper(),
        timestamp=timestamp,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=1.0,
    )


def _populate_candles(
    repo: PerpRepository,
    *,
    symbol: str,
    timeframe: TimeFrame,
    base_time: datetime,
    prices: Iterable[float],
) -> None:
    candles = []
    for index, price in enumerate(prices):
        candle_time = base_time + timedelta(minutes=index)
        candles.append(_make_candle(symbol, candle_time, price))
    repo.save_candles(symbol.upper(), timeframe.value, candles)


DEFAULT_TIMEFRAME = TimeFrame.ONE_HOUR


def _atr_expected(prices: list[float], window: int) -> list[float]:
    if len(prices) < window:
        return []
    trs: list[float] = []
    prev_close: float | None = None
    for price in prices:
        high = price + 1
        low = price - 1
        if prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
        trs.append(tr)
        prev_close = price
    atr_values: list[float] = []
    seed = sum(trs[:window]) / window
    atr_values.append(seed)
    for tr in trs[window:]:
        prev = atr_values[-1]
        atr_values.append(((prev * (window - 1)) + tr) / window)
    return atr_values


def _vwap_expected(prices: list[float], window: int) -> list[float]:
    result: list[float] = []
    for idx in range(len(prices)):
        start = max(0, idx - window + 1)
        window_prices = prices[start : idx + 1]
        vwap = sum(window_prices) / len(window_prices)
        result.append(vwap)
    return result


def _stochastic_expected(prices: list[float], window: int, signal: int = 3) -> list[dict[str, float | None]]:
    percent_k: list[float | None] = []
    for idx in range(len(prices)):
        if idx + 1 < window:
            percent_k.append(None)
            continue
        window_prices = prices[idx + 1 - window : idx + 1]
        highest_high = max(window_prices) + 1
        lowest_low = min(window_prices) - 1
        denom = highest_high - lowest_low
        if denom == 0:
            percent_k.append(None)
            continue
        k_value = 100 * (prices[idx] - lowest_low) / denom
        percent_k.append(k_value)

    percent_d: list[float | None] = []
    for idx, k_value in enumerate(percent_k):
        if k_value is None or idx + 1 < window + signal - 1:
            percent_d.append(None)
            continue
        window_vals = [
            val for val in percent_k[idx - signal + 1 : idx + 1] if val is not None
        ]
        if len(window_vals) < signal:
            percent_d.append(None)
            continue
        percent_d.append(sum(window_vals) / signal)

    series: list[dict[str, float | None]] = []
    for k_val, d_val in zip(percent_k, percent_d, strict=False):
        series.append({"percent_k": k_val, "percent_d": d_val})
    return series


def _ema_series(prices: list[float], window: int) -> list[tuple[int, float]]:
    if len(prices) < window:
        return []
    alpha = 2 / (window + 1)
    ema = mean(prices[:window])
    series: list[tuple[int, float]] = [(window - 1, ema)]
    for idx, price in enumerate(prices[window:], start=window):
        ema = alpha * price + (1 - alpha) * ema
        series.append((idx, ema))
    return series


def _ema_expected(prices: list[float], window: int) -> list[float]:
    return [value for _, value in _ema_series(prices, window)]


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _rsi_expected(prices: list[float], window: int) -> list[float]:
    if len(prices) <= window:
        return []
    gains: list[float] = []
    losses: list[float] = []
    for idx in range(1, len(prices)):
        delta = prices[idx] - prices[idx - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = mean(gains[:window])
    avg_loss = mean(losses[:window])
    rsis = [_rsi_from_avgs(avg_gain, avg_loss)]
    for idx in range(window, len(gains)):
        gain = gains[idx]
        loss = losses[idx]
        avg_gain = ((avg_gain * (window - 1)) + gain) / window
        avg_loss = ((avg_loss * (window - 1)) + loss) / window
        rsis.append(_rsi_from_avgs(avg_gain, avg_loss))
    return rsis


def _bollinger_expected(prices: list[float], window: int, k: float = 2.0) -> list[dict[str, float | None]]:
    bands: list[dict[str, float | None]] = []
    for idx in range(len(prices)):
        if idx + 1 < window:
            bands.append({"middle": None, "upper": None, "lower": None})
            continue
        window_slice = prices[idx + 1 - window : idx + 1]
        avg = mean(window_slice)
        stddev = pstdev(window_slice)
        bands.append(
            {
                "middle": avg,
                "upper": avg + k * stddev,
                "lower": avg - k * stddev,
            }
        )
    return bands


def _scalar_values(result: IndicatorResult) -> list[float | None]:
    return [point.values.get("value") for point in result.points]


def _timestamps(result: IndicatorResult) -> list[datetime]:
    return [point.timestamp for point in result.points]


def _macd_expected(prices: list[float], fast: int, slow: int, signal: int) -> list[dict[str, float]]:
    fast_series = _ema_series(prices, fast)
    slow_series = _ema_series(prices, slow)
    slow_lookup = {idx: value for idx, value in slow_series}
    macd_points = [
        value - slow_lookup[idx]
        for idx, value in fast_series
        if idx in slow_lookup
    ]
    if len(macd_points) < signal:
        return []
    initial_signal = mean(macd_points[:signal])
    signal_values = [initial_signal]
    for macd_val in macd_points[signal:]:
        prev = signal_values[-1]
        next_signal = (prev * (signal - 1) + macd_val) / signal
        signal_values.append(next_signal)
    result: list[dict[str, float]] = []
    macd_tail = macd_points[signal - 1 :]
    for macd_val, signal_val in zip(macd_tail, signal_values):
        result.append(
            {
                "macd": macd_val,
                "signal": signal_val,
                "hist": macd_val - signal_val,
            }
        )
    return result


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
    captured: dict[str, object] = {}
    fake_result = IndicatorResult(
        symbol="BTC",
        indicator=IndicatorType.SMA,
        metadata={"window": 3, "limit": 5},
        points=[
            IndicatorPoint(timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc), values={"value": 100.0}),
            IndicatorPoint(timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc), values={"value": 105.5}),
        ],
    )

    async def fake_to_thread(func: Callable, *args, **kwargs):
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr(
        "hyperliquid_analytics.services.indicator_service.asyncio.to_thread",
        fake_to_thread,
    )

    result = await indicator_service.compute_indicator(
        "btc",
        IndicatorType.SMA,
        timeframe=DEFAULT_TIMEFRAME,
        window=3,
        limit=5,
    )

    assert captured["func"] == indicator_service._compute_sma_db
    assert captured["args"] == ("btc", 3, DEFAULT_TIMEFRAME, 5)
    assert result is fake_result
    assert result.metadata == {"window": 3, "limit": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize("window", [None, 0, -1])
async def test_compute_indicator_rejects_invalid_window(window, indicator_service: IndicatorService):
    with pytest.raises(ValueError):
        await indicator_service.compute_indicator(
            "btc",
            IndicatorType.SMA,
            window=window,
        )


@pytest.mark.asyncio
async def test_compute_indicator_sma_returns_series_from_db(indicator_service: IndicatorService):
    base_time = datetime(2024, 2, 1, tzinfo=timezone.utc)
    prices = [100, 110, 120, 130, 140]
    _populate_candles(
        indicator_service._repo,
        symbol="BTC",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = await indicator_service.compute_indicator(
        "btc",
        IndicatorType.SMA,
        timeframe=DEFAULT_TIMEFRAME,
        window=3,
        limit=5,
    )

    assert result.symbol == "BTC"
    assert result.indicator is IndicatorType.SMA
    assert result.metadata == {"window": 3, "limit": 5, "timeframe": DEFAULT_TIMEFRAME.value}
    values = _scalar_values(result)
    assert values[:2] == [None, None]
    assert values[2:] == pytest.approx([110.0, 120.0, 130.0])


@pytest.mark.asyncio
async def test_compute_indicator_ema_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 2, 1, tzinfo=timezone.utc)
    prices = [100, 120, 140, 160, 180, 200]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="ETH",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = await indicator_service.compute_indicator(
        "eth",
        IndicatorType.EMA,
        timeframe=DEFAULT_TIMEFRAME,
        window=window,
        limit=None,
    )

    expected = _ema_expected(prices, window)
    assert _scalar_values(result) == pytest.approx(expected)
    assert all(point.timestamp.tzinfo is not None for point in result.points)
    assert result.metadata == {"window": window, "limit": None, "timeframe": DEFAULT_TIMEFRAME.value}


@pytest.mark.asyncio
async def test_compute_indicator_rsi_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 3, 1, tzinfo=timezone.utc)
    prices = [100, 105, 103, 108, 104, 109, 111]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="SOL",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = await indicator_service.compute_indicator(
        "sol",
        IndicatorType.RSI,
        timeframe=DEFAULT_TIMEFRAME,
        window=window,
        limit=2,
    )

    expected = _rsi_expected(prices, window)[:2]
    assert _scalar_values(result) == pytest.approx(expected)
    assert result.metadata == {"window": window, "limit": 2, "timeframe": DEFAULT_TIMEFRAME.value}


@pytest.mark.asyncio
async def test_compute_indicator_macd_returns_zero_series_for_constant_prices(indicator_service: IndicatorService):
    base_time = datetime(2024, 4, 1, tzinfo=timezone.utc)
    prices = [50.0] * 40
    window = 26
    _populate_candles(
        indicator_service._repo,
        symbol="ARB",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = await indicator_service.compute_indicator(
        "arb",
        IndicatorType.MACD,
        timeframe=DEFAULT_TIMEFRAME,
        window=window,
        limit=5,
    )

    assert result.metadata == {
        "fast": 12,
        "slow": window,
        "signal": 9,
        "limit": 5,
        "timeframe": DEFAULT_TIMEFRAME.value,
    }
    assert len(result.points) == 5
    for point in result.points:
        value = point.values
        assert value["macd"] == pytest.approx(0.0)
        assert value["signal"] == pytest.approx(0.0)
        assert value["hist"] == pytest.approx(0.0)

@pytest.mark.asyncio
async def test_compute_indicator_vwap_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 5, 1, tzinfo=timezone.utc)
    prices = [100, 110, 120, 130]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="OP",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = await indicator_service.compute_indicator(
        "op",
        IndicatorType.VWAP,
        timeframe=DEFAULT_TIMEFRAME,
        window=window,
        limit=None,
    )

    assert _scalar_values(result) == pytest.approx(_vwap_expected(prices, window))
    assert result.metadata == {"window": window, "limit": None, "timeframe": DEFAULT_TIMEFRAME.value}


@pytest.mark.asyncio
async def test_compute_indicator_atr_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 3, 1, tzinfo=timezone.utc)
    prices = [100, 105, 103, 108, 104, 109]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="XRP",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = await indicator_service.compute_indicator(
        "xrp",
        IndicatorType.ATR,
        timeframe=DEFAULT_TIMEFRAME,
        window=window,
        limit=None,
    )

    expected = _atr_expected(prices, window)
    assert _scalar_values(result) == pytest.approx(expected)
    assert result.metadata == {"window": window, "limit": None, "timeframe": DEFAULT_TIMEFRAME.value}


@pytest.mark.asyncio
async def test_compute_indicator_stochastic_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
    prices = [100, 102, 104, 103, 105, 106, 108]
    window = 3
    signal = 3
    _populate_candles(
        indicator_service._repo,
        symbol="LTC",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = await indicator_service.compute_indicator(
        "ltc",
        IndicatorType.STOCHASTIC,
        timeframe=DEFAULT_TIMEFRAME,
        window=window,
        limit=None,
    )

    expected = _stochastic_expected(prices, window, signal)
    for point, expected_entry in zip(result.points, expected, strict=False):
        for key in ("percent_k", "percent_d"):
            exp_val = expected_entry[key]
            if exp_val is None:
                assert point.values[key] is None
            else:
                assert point.values[key] == pytest.approx(exp_val)
    assert result.metadata == {
        "window": window,
        "signal": signal,
        "limit": None,
        "timeframe": DEFAULT_TIMEFRAME.value,
    }


@pytest.mark.asyncio
async def test_compute_indicator_bollinger_returns_bands(indicator_service: IndicatorService):
    base_time = datetime(2024, 5, 1, tzinfo=timezone.utc)
    prices = [100, 105, 110, 115, 120]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="OP",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = await indicator_service.compute_indicator(
        "op",
        IndicatorType.BOLLINGER,
        timeframe=DEFAULT_TIMEFRAME,
        window=window,
        limit=None,
    )

    expected_bands = _bollinger_expected(prices, window)
    series_values = [point.values for point in result.points]
    assert len(series_values) == len(expected_bands)
    for actual, expected in zip(series_values, expected_bands, strict=True):
        assert actual.keys() == expected.keys()
        for key in actual:
            if expected[key] is None:
                assert actual[key] is None
            else:
                assert actual[key] == pytest.approx(expected[key])
def test_compute_sma_db_returns_full_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _populate_candles(
        indicator_service._repo,
        symbol="BTC",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=[100, 110, 120, 130, 140],
    )
    recording_conn = RecordingConnection(indicator_service._repo._conn)
    indicator_service._repo._conn = recording_conn

    result = indicator_service._compute_sma_db("btc", window=3, timeframe=DEFAULT_TIMEFRAME)

    assert recording_conn.last_query is not None
    assert "LIMIT ?" not in recording_conn.last_query
    assert recording_conn.last_params == ["BTC", DEFAULT_TIMEFRAME.value]

    timestamps = _timestamps(result)
    values = _scalar_values(result)

    assert timestamps == sorted(timestamps)
    assert values[:2] == [None, None]
    assert values[2:] == pytest.approx([110.0, 120.0, 130.0])


def test_compute_sma_db_with_window_larger_than_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _populate_candles(
        indicator_service._repo,
        symbol="BTC",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=[100, 110],
    )

    result = indicator_service._compute_sma_db("btc", window=5, timeframe=DEFAULT_TIMEFRAME)

    assert _scalar_values(result) == [None, None]


def test_compute_sma_db_applies_limit(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _populate_candles(
        indicator_service._repo,
        symbol="BTC",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=[200, 210, 220, 230, 240],
    )
    recording_conn = RecordingConnection(indicator_service._repo._conn)
    indicator_service._repo._conn = recording_conn

    result = indicator_service._compute_sma_db("btc", window=3, timeframe=DEFAULT_TIMEFRAME, limit=3)

    assert "LIMIT ?" in recording_conn.last_query
    assert recording_conn.last_params == ["BTC", DEFAULT_TIMEFRAME.value, 3]
    assert len(result.points) == 3
    assert _scalar_values(result) == [None, None, pytest.approx(210.0)]


def test_compute_sma_db_rejects_invalid_inputs(indicator_service: IndicatorService):
    with pytest.raises(ValueError):
        indicator_service._compute_sma_db("", window=3, timeframe=DEFAULT_TIMEFRAME)
    with pytest.raises(ValueError):
        indicator_service._compute_sma_db("btc", window=0, timeframe=DEFAULT_TIMEFRAME)
    with pytest.raises(ValueError):
        indicator_service._compute_sma_db("btc", window=3, timeframe=DEFAULT_TIMEFRAME, limit=0)


def test_compute_ema_db_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 120, 140, 160, 180]
    _populate_candles(
        indicator_service._repo,
        symbol="ETH",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_ema_db("eth", window=3, timeframe=DEFAULT_TIMEFRAME)

    assert _scalar_values(result) == pytest.approx(_ema_expected(prices, 3))


def test_compute_ema_db_returns_empty_when_not_enough_points(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _populate_candles(
        indicator_service._repo,
        symbol="ETH",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=[100, 110],
    )

    result = indicator_service._compute_ema_db("eth", window=3, timeframe=DEFAULT_TIMEFRAME)

    assert result.points == []


def test_compute_ema_db_applies_limit(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 120, 140, 160, 180]
    _populate_candles(
        indicator_service._repo,
        symbol="ETH",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_ema_db("eth", window=3, timeframe=DEFAULT_TIMEFRAME, limit=2)

    assert len(result.points) == 2
    assert _scalar_values(result) == pytest.approx(_ema_expected(prices, 3)[:2])


def test_compute_rsi_db_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 105, 103, 108, 104, 110]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="SOL",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_rsi_db("sol", window=window, timeframe=DEFAULT_TIMEFRAME)

    assert _scalar_values(result) == pytest.approx(_rsi_expected(prices, window))


def test_compute_rsi_db_returns_empty_when_insufficient_points(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _populate_candles(
        indicator_service._repo,
        symbol="SOL",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=[100, 101, 102],
    )

    result = indicator_service._compute_rsi_db("sol", window=5, timeframe=DEFAULT_TIMEFRAME)

    assert result.points == []


def test_compute_rsi_db_applies_limit(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 105, 103, 108, 104, 110]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="SOL",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_rsi_db("sol", window=window, timeframe=DEFAULT_TIMEFRAME, limit=2)

    assert len(result.points) == 2
    assert _scalar_values(result) == pytest.approx(_rsi_expected(prices, window)[:2])


def test_compute_atr_db_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 105, 103, 108]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="XRP",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_atr_db("xrp", timeframe=DEFAULT_TIMEFRAME, window=window)
    expected = _atr_expected(prices, window)
    assert _scalar_values(result) == pytest.approx(expected)


def test_compute_macd_db_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [float(x) for x in range(1, 15)]
    fast, slow, signal = 4, 6, 3
    _populate_candles(
        indicator_service._repo,
        symbol="ARB",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_macd_db(
        "arb",
        timeframe=DEFAULT_TIMEFRAME,
        fast=fast,
        slow=slow,
        signal=signal,
    )
    expected = _macd_expected(prices, fast, slow, signal)

    assert len(result.points) == len(expected)
    for point, expected_entry in zip(result.points, expected, strict=False):
        for key in ("macd", "signal", "hist"):
            assert point.values[key] == pytest.approx(expected_entry[key], rel=1e-9, abs=1e-9)


def test_compute_macd_db_returns_empty_when_insufficient_points(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    _populate_candles(
        indicator_service._repo,
        symbol="ARB",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=[1.0, 2.0, 3.0, 4.0],
    )

    result = indicator_service._compute_macd_db("arb", timeframe=DEFAULT_TIMEFRAME, fast=4, slow=6, signal=3)

    assert result.points == []


def test_compute_macd_db_applies_limit(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [float(x) for x in range(1, 30)]
    _populate_candles(
        indicator_service._repo,
        symbol="ARB",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_macd_db(
        "arb",
        timeframe=DEFAULT_TIMEFRAME,
        fast=4,
        slow=6,
        signal=3,
        limit=2,
    )

    assert len(result.points) == 2


def test_compute_bollinger_db_returns_expected_bands(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 110, 120, 130, 140]
    window = 3
    _populate_candles(
        indicator_service._repo,
        symbol="OP",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_bollinger_db("op", timeframe=DEFAULT_TIMEFRAME, window=window, k=2.0)
    expected = _bollinger_expected(prices, window)

    assert len(result.points) == len(expected)
    for point, expected_entry in zip(result.points, expected, strict=False):
        for key in ("middle", "upper", "lower"):
            if expected_entry[key] is None:
                assert point.values[key] is None
            else:
                assert point.values[key] == pytest.approx(expected_entry[key])


def test_compute_bollinger_db_returns_none_values_when_insufficient_window(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 110]
    window = 4
    _populate_candles(
        indicator_service._repo,
        symbol="OP",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_bollinger_db("op", timeframe=DEFAULT_TIMEFRAME, window=window, k=2.0)

    assert all(point.values["middle"] is None for point in result.points)
    assert len(result.points) == len(prices)


def test_compute_bollinger_db_applies_limit(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 110, 120, 130]
    window = 2
    _populate_candles(
        indicator_service._repo,
        symbol="OP",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_bollinger_db("op", timeframe=DEFAULT_TIMEFRAME, window=window, k=2.0, limit=2)

    assert len(result.points) == 2


def test_compute_vwap_db_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 110, 120]
    window = 2
    _populate_candles(
        indicator_service._repo,
        symbol="OP",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_vwap_db("op", timeframe=DEFAULT_TIMEFRAME, window=window)
    assert _scalar_values(result) == pytest.approx(_vwap_expected(prices, window))


def test_compute_stochastic_db_returns_expected_series(indicator_service: IndicatorService):
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prices = [100, 103, 101, 104, 106]
    window = 3
    signal = 3
    _populate_candles(
        indicator_service._repo,
        symbol="ADA",
        timeframe=DEFAULT_TIMEFRAME,
        base_time=base_time,
        prices=prices,
    )

    result = indicator_service._compute_stochastic_db(
        "ada",
        timeframe=DEFAULT_TIMEFRAME,
        window=window,
        signal_window=signal,
    )
    expected = _stochastic_expected(prices, window, signal)
    for point, expected_entry in zip(result.points, expected, strict=False):
        for key in ("percent_k", "percent_d"):
            exp_val = expected_entry[key]
            if exp_val is None:
                assert point.values[key] is None
            else:
                assert point.values[key] == pytest.approx(exp_val)

