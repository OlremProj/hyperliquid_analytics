from __future__ import annotations

import asyncio
from typing import Any

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.repository.perp_repository import PerpRepository
from hyperliquid_analytics.services.analytics_service import AnalyticsService
from hyperliquid_analytics.models.indicator_result_models import IndicatorResult, IndicatorType
from hyperliquid_analytics.services.indicators.moving_average import (
    compute_ema as compute_ema_indicator,
    compute_sma as compute_sma_indicator,
)
from hyperliquid_analytics.services.indicators.oscillators import (
    compute_atr as compute_atr_indicator,
    compute_rsi as compute_rsi_indicator,
    compute_stochastic as compute_stochastic_indicator,
)
from hyperliquid_analytics.services.indicators.trend import (
    compute_macd as compute_macd_indicator,
)
from hyperliquid_analytics.services.indicators.volatility import (
    compute_bollinger as compute_bollinger_indicator,
)
from hyperliquid_analytics.services.indicators.volume import (
    compute_vwap as compute_vwap_indicator,
)


class IndicatorService:
    DEFAULT_INDICATOR_TIMEFRAME = TimeFrame.ONE_HOUR

    def __init__(self, analytics_service: AnalyticsService, repo: PerpRepository | None = None) -> None:
        self._analytics_service = analytics_service
        self._repo = repo or analytics_service.perp_repository
    
    async def compute_indicator(
        self,
        symbol: str,
        indicator: IndicatorType,
        *,
        timeframe: TimeFrame = DEFAULT_INDICATOR_TIMEFRAME,
        window: int | None = None,
        limit: int | None = None,
    ) -> IndicatorResult:
        tf = self._normalize_timeframe(timeframe)
        period = self._validate_window(window, indicator)

        if indicator is IndicatorType.SMA:
            return await asyncio.to_thread(
                self._compute_sma_db,
                symbol,
                period,
                tf,
                limit,
            )
        if indicator is IndicatorType.EMA:
            return await asyncio.to_thread(
                self._compute_ema_db,
                symbol,
                period,
                tf,
                limit,
            )
        if indicator is IndicatorType.RSI:
            return await asyncio.to_thread(
                self._compute_rsi_db,
                symbol,
                period,
                tf,
                limit,
            )
        if indicator is IndicatorType.MACD:
            fast = 12
            signal = 9
            return await asyncio.to_thread(
                self._compute_macd_db,
                symbol,
                tf,
                fast,
                period,
                signal,
                limit,
            )
        if indicator is IndicatorType.BOLLINGER:
            k = 2.0
            return await asyncio.to_thread(
                self._compute_bollinger_db,
                symbol,
                tf,
                period,
                k,
                limit,
            )
        if indicator is IndicatorType.ATR:
            return await asyncio.to_thread(
                self._compute_atr_db,
                symbol,
                tf,
                period,
                limit,
            )
        if indicator is IndicatorType.VWAP:
            return await asyncio.to_thread(
                self._compute_vwap_db,
                symbol,
                tf,
                period,
                limit,
            )
        if indicator is IndicatorType.STOCHASTIC:
            signal_window = 3
            return await asyncio.to_thread(
                self._compute_stochastic_db,
                symbol,
                tf,
                period,
                signal_window,
                limit,
            )

    def _compute_bollinger_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        window: int = 20,
        k: float = 2.0,
        limit: int | None = None,
    ) -> IndicatorResult:
        return compute_bollinger_indicator(
            self._repo,
            symbol,
            timeframe,
            window,
            k,
            limit,
        )


    def _compute_macd_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        limit: int | None = None,
    ) -> IndicatorResult:
        return compute_macd_indicator(
            self._repo,
            symbol,
            timeframe,
            fast,
            slow,
            signal,
            limit,
        )
    
    def _compute_rsi_db(
        self,
        symbol: str,
        window: int,
        timeframe: TimeFrame,
        limit: int | None = None,
    ) -> IndicatorResult:
        return compute_rsi_indicator(self._repo, symbol, timeframe, window, limit)

    def _compute_ema_db(
        self,
        symbol: str,
        window: int,
        timeframe: TimeFrame,
        limit: int | None = None,
    ) -> IndicatorResult:
        return compute_ema_indicator(self._repo, symbol, timeframe, window, limit)


    def _compute_sma_db(
        self,
        symbol: str,
        window: int,
        timeframe: TimeFrame,
        limit: int | None = None,
    ) -> IndicatorResult:
        if not symbol:
            raise ValueError("symbol must be provided")
        if window < 1:
            raise ValueError("window must be >= 1")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1 when provided")
        return compute_sma_indicator(self._repo, symbol, timeframe, window, limit)

    def _compute_atr_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        window: int,
        limit: int | None = None,
    ) -> IndicatorResult:
        return compute_atr_indicator(self._repo, symbol, timeframe, window, limit)


    def _compute_vwap_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        window: int,
        limit: int | None = None,
    ) -> IndicatorResult:
        return compute_vwap_indicator(self._repo, symbol, timeframe, window, limit)


    def _compute_stochastic_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        window: int,
        signal_window: int = 3,
        limit: int | None = None,
    ) -> IndicatorResult:
        return compute_stochastic_indicator(
            self._repo,
            symbol,
            timeframe,
            window,
            signal_window,
            limit,
        )

    @staticmethod
    def _validate_window(window: int | None, indicator: IndicatorType) -> int:
        if window is None or window < 1:
            raise ValueError(f"window must be >= 1 for {indicator.value.upper()}")
        return window

    @staticmethod
    def _normalize_timeframe(timeframe: TimeFrame | str | None) -> TimeFrame:
        if timeframe is None:
            return IndicatorService.DEFAULT_INDICATOR_TIMEFRAME
        if isinstance(timeframe, TimeFrame):
            return timeframe
        return TimeFrame(timeframe)
