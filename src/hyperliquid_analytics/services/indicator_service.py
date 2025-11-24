from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Sequence

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.repository.perp_repository import PerpRepository
from hyperliquid_analytics.services.analytics_service import AnalyticsService
from hyperliquid_analytics.models.indicator_result_models import (
    IndicatorType,
    IndicatorResult,
    wrap_scalar_series,
    wrap_record_series,
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
        raise NotImplementedError(f"{indicator.value} not implemented yet")

    def _compute_bollinger_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        window: int = 20,
        k: float = 2.0,
        limit: int | None = None,
    ) -> IndicatorResult:
        params = [symbol.upper(), timeframe.value]
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)

        query = f"""
            WITH ordered AS (
                SELECT
                    timestamp,
                    close,
                    ROW_NUMBER() OVER (ORDER BY timestamp) AS row_num
                FROM candles
                WHERE symbol = ? AND timeframe = ?
            ),
            bands AS (
                SELECT
                    timestamp,
                    AVG(close) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) AS sma,
                    STDDEV_POP(close) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) AS stddev,
                    row_num
                FROM ordered
            )
            SELECT
                timestamp,
                CASE WHEN row_num >= {window} THEN sma ELSE NULL END AS middle,
                CASE WHEN row_num >= {window} THEN sma + {k} * stddev ELSE NULL END AS upper,
                CASE WHEN row_num >= {window} THEN sma - {k} * stddev ELSE NULL END AS lower
            FROM bands
            ORDER BY timestamp ASC
            {limit_clause}
        """
        rows = self._repo._conn.execute(query, params).fetchall()
        return wrap_record_series(
            symbol,
            IndicatorType.BOLLINGER,
            rows,
            fields=("middle", "upper", "lower"),
            metadata={"window": window, "k": k, "limit": limit, "timeframe": timeframe.value},
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
        alpha_fast = 2 / (fast + 1)
        beta_fast = 1 - alpha_fast
        alpha_slow = 2 / (slow + 1)
        beta_slow = 1 - alpha_slow

        params = [
            symbol.upper(),      
            timeframe.value,
            alpha_fast, beta_fast,    # EMAs rapides
            alpha_slow, beta_slow,    # EMAs lentes
        ]
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)

        query = f"""
            WITH RECURSIVE
                ordered AS (
                    SELECT
                        timestamp,
                        close,
                        ROW_NUMBER() OVER (ORDER BY timestamp) AS row_num
                    FROM candles
                    WHERE symbol = ? AND timeframe = ?
                ),
                fast_seed AS (
                    SELECT row_num, timestamp, ema
                    FROM (
                        SELECT
                            row_num,
                            timestamp,
                            AVG(close) OVER (
                                ORDER BY row_num
                                ROWS BETWEEN {fast - 1} PRECEDING AND CURRENT ROW
                            ) AS ema
                        FROM ordered
                    )
                    WHERE row_num = {fast}
                ),
                fast_ema AS (
                    SELECT row_num, timestamp, ema
                    FROM fast_seed
                    UNION ALL
                    SELECT
                        o.row_num,
                        o.timestamp,
                        (? * o.close) + (? * fast_ema.ema) AS ema
                    FROM ordered o
                    JOIN fast_ema ON o.row_num = fast_ema.row_num + 1
                ),
                slow_seed AS (
                    SELECT row_num, timestamp, ema
                    FROM (
                        SELECT
                            row_num,
                            timestamp,
                            AVG(close) OVER (
                                ORDER BY row_num
                                ROWS BETWEEN {slow - 1} PRECEDING AND CURRENT ROW
                            ) AS ema
                        FROM ordered
                    )
                    WHERE row_num = {slow}
                ),
                slow_ema AS (
                    SELECT row_num, timestamp, ema
                    FROM slow_seed
                    UNION ALL
                    SELECT
                        o.row_num,
                        o.timestamp,
                        (? * o.close) + (? * slow_ema.ema) AS ema
                    FROM ordered o
                    JOIN slow_ema ON o.row_num = slow_ema.row_num + 1
                ),
                macd_base AS (
                    SELECT
                        f.timestamp,
                        f.ema AS ema_fast,
                        s.ema AS ema_slow,
                        f.ema - s.ema AS macd,
                        ROW_NUMBER() OVER (ORDER BY f.timestamp) AS macd_row
                    FROM fast_ema f
                    JOIN slow_ema s
                      ON f.timestamp = s.timestamp
                ),
                signal_seed AS (
                    SELECT macd_row, timestamp, signal
                    FROM (
                        SELECT
                            macd_row,
                            timestamp,
                            AVG(macd) OVER (
                                ORDER BY macd_row
                                ROWS BETWEEN {signal - 1} PRECEDING AND CURRENT ROW
                            ) AS signal
                        FROM macd_base
                    )
                    WHERE macd_row = {signal}
                ),
                signal_ema AS (
                    SELECT macd_row, timestamp, signal
                    FROM signal_seed
                    UNION ALL
                    SELECT
                        m.macd_row,
                        m.timestamp,
                        (signal_ema.signal * {signal - 1} + m.macd) / {signal} AS signal
                    FROM macd_base m
                    JOIN signal_ema ON m.macd_row = signal_ema.macd_row + 1
                )
                SELECT
                    m.timestamp,
                    m.macd,
                    s.signal,
                    m.macd - s.signal AS hist
                FROM macd_base m
                JOIN signal_ema s ON m.macd_row = s.macd_row
                ORDER BY m.timestamp ASC
                {limit_clause}
                """ 
        rows = self._repo._conn.execute(query, params).fetchall()
        return wrap_record_series(
            symbol,
            IndicatorType.MACD,
            rows,
            fields=("macd", "signal", "hist"),
            metadata={"fast": fast, "slow": slow, "signal": signal, "limit": limit, "timeframe": timeframe.value},
        )
    
    def _compute_rsi_db(
        self,
        symbol: str,
        window: int,
        timeframe: TimeFrame,
        limit: int | None = None,
    ) -> IndicatorResult:
        params = [symbol.upper(), timeframe.value]

        limit_clause = "LIMIT ?" if limit is not None else ""

        if limit is not None:
            params.append(limit)

        ordered = """
            ordered AS (
                SELECT
                    timestamp,
                    close,
                    ROW_NUMBER() OVER (ORDER BY timestamp) AS row_num
                FROM candles
                WHERE symbol = ? AND timeframe = ?
            ),
        """

        deltas = """
            deltas AS (
                SELECT
                    row_num,
                    timestamp,
                    close,
                    close - LAG(close) OVER (ORDER BY timestamp) AS diff,
                    GREATEST(close - LAG(close) OVER (ORDER BY timestamp), 0) AS gain,
                    GREATEST(LAG(close) OVER (ORDER BY timestamp) - close, 0) AS loss
                FROM ordered
            ),
        """
        
        seed = f"""
           seed as (
                select row_num, timestamp, avg_gain, avg_loss
                from (
                    select
                        row_num,
                        timestamp,
                        avg(gain) over (
                            order by row_num
                            rows between {window - 1} preceding and current row
                        ) as avg_gain,
                        avg(loss) over (
                            order by row_num
                            rows between {window - 1} preceding and current row
                        ) as avg_loss
                    from deltas
                )
                where row_num = {window + 1}
            ),
        """
 
        ema = f"""
           ema AS (
                SELECT
                    row_num,
                    timestamp,
                    avg_gain AS ema_gain,
                    avg_loss AS ema_loss
                FROM seed

                UNION ALL

                SELECT
                    d.row_num,
                    d.timestamp,
                    (ema.ema_gain * {window - 1} + d.gain) / {window} AS ema_gain,
                    (ema.ema_loss * {window - 1} + d.loss) / {window} AS ema_loss
                FROM deltas d
                JOIN ema
                    ON d.row_num = ema.row_num + 1
            )
        """
        query = f"""
            WITH RECURSIVE
            {ordered}
            {deltas}
            {seed}
            {ema}
            SELECT
                timestamp,
                CASE
                    WHEN ema_loss = 0 THEN 100
                    ELSE 100 - 100 / (1 + ema_gain / ema_loss)
                END AS rsi
            FROM ema
            ORDER BY timestamp ASC
            {limit_clause}
        """
        rows = self._repo._conn.execute(query, params).fetchall()
        return wrap_scalar_series(
            symbol,
            IndicatorType.RSI,
            rows,
            field="value",
            metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
        )

    def _compute_ema_db(
        self,
        symbol: str,
        window: int,
        timeframe: TimeFrame,
        limit: int | None = None,
    ) -> IndicatorResult:
        alpha = 2 / (window + 1)
        beta = 1 - alpha
        params = [symbol.upper(), timeframe.value, alpha, beta]

        limit_clause = "LIMIT ?" if limit is not None else ""

        if limit is not None:
            params.append(limit)

        ordered = """
            ordered AS (
                SELECT
                    timestamp,
                    close,
                    ROW_NUMBER() OVER (ORDER BY timestamp) AS row_num
                FROM candles
                WHERE symbol = ? AND timeframe = ?
            ),
        """

        seed = f"""
            seed AS (
                SELECT row_num, timestamp, ema
                FROM (
                    SELECT
                        row_num,
                        timestamp,
                        AVG(close) OVER (
                            ORDER BY row_num
                            ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                        ) AS ema
                    FROM ordered
                )
                WHERE row_num = {window}
            ),
        """

        ema = """
            ema AS (
                SELECT row_num, timestamp, ema
                FROM seed

                UNION ALL

                SELECT
                    o.row_num,
                    o.timestamp,
                    (? * o.close) + (? * ema.ema) AS ema
                FROM ordered o
                JOIN ema
                    ON o.row_num = ema.row_num + 1
            )
        """

        query = f"""
            WITH RECURSIVE
            {ordered}
            {seed}
            {ema}
            SELECT timestamp, ema
            FROM ema
            ORDER BY timestamp ASC
            {limit_clause}
        """

        rows = self._repo._conn.execute(query, params).fetchall()

        return wrap_scalar_series(
            symbol,
            IndicatorType.EMA,
            rows,
            field="value",
            metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
        )


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

        params: list[Any] = [symbol.upper(), timeframe.value]
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)

        query = f"""
            WITH ordered AS (
                SELECT
                    timestamp,
                    close,
                    ROW_NUMBER() OVER (ORDER BY timestamp) AS row_num,
                    AVG(close) OVER (
                        ORDER BY timestamp
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) AS sma_raw
                FROM candles
                WHERE symbol = ? AND timeframe = ?
            )
            SELECT
                timestamp,
                CASE WHEN row_num >= {window} THEN sma_raw ELSE NULL END AS sma
            FROM ordered
            ORDER BY timestamp ASC
            {limit_clause}
        """

        rows: Sequence[tuple[datetime, float | None]] = (
            self._repo._conn.execute(
                query,
                params,
            ).fetchall()
        )
        return wrap_scalar_series(
            symbol,
            IndicatorType.SMA,
            rows,
            field="value",
            metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
        )

    def _compute_atr_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        window: int,
        limit: int | None = None,
    ) -> IndicatorResult:
        params: list[Any] = [symbol.upper(), timeframe.value]
        limit_clause = "LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(limit)

        query = f"""
            WITH RECURSIVE
            ordered AS (
                SELECT
                    timestamp,
                    high,
                    low,
                    close,
                    LAG(close) OVER (ORDER BY timestamp) AS prev_close,
                    ROW_NUMBER() OVER (ORDER BY timestamp) AS row_num
                FROM candles
                WHERE symbol = ? AND timeframe = ?
            ),
            tr AS (
                SELECT
                    row_num,
                    timestamp,
                    CASE
                        WHEN prev_close IS NULL THEN high - low
                        ELSE GREATEST(
                            high - low,
                            ABS(high - prev_close),
                            ABS(low - prev_close)
                        )
                    END AS true_range
                FROM ordered
            ),
            seed AS (
                SELECT row_num, timestamp, atr
                FROM (
                    SELECT
                        row_num,
                        timestamp,
                        AVG(true_range) OVER (
                            ORDER BY row_num
                            ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                        ) AS atr
                    FROM tr
                )
                WHERE row_num = {window}
            ),
            atr_calc AS (
                SELECT row_num, timestamp, atr
                FROM seed

                UNION ALL

                SELECT
                    tr.row_num,
                    tr.timestamp,
                    ((atr_calc.atr * ({window} - 1)) + tr.true_range) / {window} AS atr
                FROM tr
                JOIN atr_calc ON tr.row_num = atr_calc.row_num + 1
            )
            SELECT timestamp, atr
            FROM atr_calc
            ORDER BY timestamp ASC
            {limit_clause}
        """
        rows = self._repo._conn.execute(query, params).fetchall()
        return wrap_scalar_series(
            symbol,
            IndicatorType.ATR,
            rows,
            field="value",
            metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
        )


    def _compute_vwap_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        window: int,
        limit: int | None = None
    ) -> IndicatorResult:
        params: list[Any] = [symbol.upper(), timeframe.value]
        limit_clause = "LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(limit)
        query = f"""
            WITH ordered AS (
                SELECT
                    timestamp,
                    close,
                    volume,
                    ROW_NUMBER() OVER (ORDER BY timestamp) AS row_num
                FROM candles
                WHERE symbol = ? AND timeframe = ?
            ),
            rolling AS (
                SELECT
                    timestamp,
                    SUM(close * volume) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) AS pv,
                    SUM(volume) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) AS vol
                FROM ordered
            )
            SELECT
                timestamp,
                CASE WHEN vol > 0 THEN pv / vol ELSE NULL END AS vwap
            FROM rolling
            ORDER BY timestamp ASC
            {limit_clause}
        """
        rows = self._repo._conn.execute(query, params).fetchall()
        return wrap_scalar_series(
            symbol,
            IndicatorType.VWAP,
            rows,
            field="value",
            metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
        )

    def _compute_stochastic_db(
        self,
        symbol: str,
        timeframe: TimeFrame,
        window: int,
        signal_window: int = 3,
        limit: int | None = None,
    ) -> IndicatorResult:
        params: list[Any] = [symbol.upper(), timeframe.value]
        limit_clause = "LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(limit)

        query = f"""
            WITH ordered AS (
                SELECT
                    timestamp,
                    high,
                    low,
                    close,
                    ROW_NUMBER() OVER (ORDER BY timestamp) AS row_num
                FROM candles
                WHERE symbol = ? AND timeframe = ?
            ),
            ranges AS (
                SELECT
                    timestamp,
                    row_num,
                    close,
                    MIN(low) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) AS lowest_low,
                    MAX(high) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) AS highest_high
                FROM ordered
            ),
            percent_k AS (
                SELECT
                    timestamp,
                    row_num,
                    CASE
                        WHEN row_num < {window} THEN NULL
                        WHEN highest_high = lowest_low THEN NULL
                        ELSE
                            100.0 * (close - lowest_low) / NULLIF(highest_high - lowest_low, 0)
                    END AS k
                FROM ranges
            ),
            percent_d AS (
                SELECT
                    timestamp,
                    row_num,
                    k,
                    AVG(k) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {signal_window - 1} PRECEDING AND CURRENT ROW
                    ) AS d_raw
                FROM percent_k
            ),
            final AS (
                SELECT
                    timestamp,
                    k AS percent_k,
                    CASE
                        WHEN row_num < {window + signal_window - 1} THEN NULL
                        ELSE d_raw
                    END AS percent_d
                FROM percent_d
            )
            SELECT timestamp, percent_k, percent_d
            FROM final
            ORDER BY timestamp ASC
            {limit_clause}
        """
        rows = self._repo._conn.execute(query, params).fetchall()
        return wrap_record_series(
            symbol,
            IndicatorType.STOCHASTIC,
            rows,
            fields=("percent_k", "percent_d"),
            metadata={
                "window": window,
                "signal": signal_window,
                "limit": limit,
                "timeframe": timeframe.value,
            },
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
