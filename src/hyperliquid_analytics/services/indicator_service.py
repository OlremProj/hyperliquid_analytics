from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Sequence

from dataclasses import dataclass

from hyperliquid_analytics.repository.perp_repository import PerpRepository
from hyperliquid_analytics.services.analytics_service import AnalyticsService


class IndicatorService:
    def __init__(self, analytics_service: AnalyticsService, repo: PerpRepository | None = None) -> None:
        self._analytics_service = analytics_service
        self._repo = repo or analytics_service.perp_repository
    
    async def compute_indicator(
        self,
        symbol: str,
        indicator: IndicatorType,
        *,
        window: int | None = None,
        limit: int | None = None,
    ) -> IndicatorResult:
       window = self._validate_window(window, indicator)
       match indicator:
            case IndicatorType.SMA:
                series = await asyncio.to_thread(
                    self._compute_sma_db,
                    symbol,
                    window,
                    limit,
                )
            case IndicatorType.EMA:
                series = await asyncio.to_thread(
                    self._compute_ema_db,
                    symbol,
                    window,
                    limit,
                )
            case IndicatorType.RSI:
                series = await asyncio.to_thread(
                    self._compute_rsi_db,
                    symbol,
                    window,
                    limit,
                )
            case IndicatorType.MACD:
                slow = window
                fast = 12
                signal = 9
                series = await asyncio.to_thread(
                    self._compute_macd_db,
                    symbol,
                    fast,
                    slow,
                    signal,
                    limit,
                )
            case IndicatorType.BOLLINGER:
                k = 2.0
                series = await asyncio.to_thread(
                    self._compute_bollinger_db,
                    symbol,
                    window,
                    k,
                    limit,
                )
            case _:
                raise NotImplementedError(f"{indicator.value} not implemented yet")

       return IndicatorResult(
            symbol=symbol.upper(),
            indicator=indicator,
            params={"window": window, "limit": limit},
            series=series,
       )

    def _compute_bollinger_db(
        self,
        symbol: str,
        window: int = 20,
        k: float = 2.0,
        limit: int | None = None,
    ) -> list[tuple[datetime, float | None]]:
        params = [symbol.upper()]

        if limit is not None:
            params.append(limit)

        query = f"""
            WITH ordered AS (
                SELECT
                    fetched_at,
                    mark_px,
                    ROW_NUMBER() OVER (ORDER BY fetched_at) AS row_num
                FROM perp_asset_ctxs
                WHERE symbol = ?
            ),
            bands AS (
                SELECT
                    fetched_at,
                    AVG(mark_px) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) as sma,
                    STDDEV_POP(mark_px) OVER (
                        ORDER BY row_num
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) as stddev,
                    row_num
                FROM ordered
            )
            SELECT
                fetched_at,
                CASE WHEN row_num >= {window} THEN sma ELSE NULL END AS middle,
                CASE WHEN row_num >= {window} THEN sma + {k} * stddev ELSE NULL END AS upper,
                CASE WHEN row_num >= {window} THEN sma - {k} * stddev ELSE NULL END AS lower
            FROM bands
            ORDER BY fetched_at ASC
            { "LIMIT ?" if limit is not None else "" }
        """
        rows = self._repo._conn.execute(query, params).fetchall()
        return [
                (
                    ts,
                    {
                        "middle": None if middle is None else float(middle),
                        "upper": None if upper is None else float(upper),
                        "lower": None if lower is None else float(lower),
                    },
                )
                for ts, middle, upper, lower in rows
        ]

    def _compute_macd_db(
        self,
        symbol: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        limit: int | None = None,
    ) -> list[tuple[datetime, float | None]]:
        alpha_fast = 2 / (fast + 1)
        beta_fast = 1 - alpha_fast
        alpha_slow = 2 / (slow + 1)
        beta_slow = 1 - alpha_slow

        params = [
            symbol.upper(),           # pour ordered
            alpha_fast, beta_fast,    # EMAs rapides
            alpha_slow, beta_slow,    # EMAs lentes
        ]
        if limit is not None:
            params.append(limit)

        query = f"""
            WITH RECURSIVE
                ordered AS (
                    SELECT
                        fetched_at,
                        mark_px,
                        ROW_NUMBER() OVER (ORDER BY fetched_at) AS row_num
                    FROM perp_asset_ctxs
                    WHERE symbol = ?
                ),
                fast_seed AS (
                    SELECT row_num, fetched_at, ema
                    FROM (
                        SELECT
                            row_num,
                            fetched_at,
                            AVG(mark_px) OVER (
                                ORDER BY row_num
                                ROWS BETWEEN {fast - 1} PRECEDING AND CURRENT ROW
                            ) AS ema
                        FROM ordered
                    )
                    WHERE row_num = {fast}
                ),
                fast_ema AS (
                    SELECT row_num, fetched_at, ema
                    FROM fast_seed
                    UNION ALL
                    SELECT
                        o.row_num,
                        o.fetched_at,
                        (? * o.mark_px) + (? * fast_ema.ema) AS ema
                    FROM ordered o
                    JOIN fast_ema ON o.row_num = fast_ema.row_num + 1
                ),
                slow_seed AS (
                    SELECT row_num, fetched_at, ema
                    FROM (
                        SELECT
                            row_num,
                            fetched_at,
                            AVG(mark_px) OVER (
                                ORDER BY row_num
                                ROWS BETWEEN {slow - 1} PRECEDING AND CURRENT ROW
                            ) AS ema
                        FROM ordered
                    )
                    WHERE row_num = {slow}
                ),
                slow_ema AS (
                    SELECT row_num, fetched_at, ema
                    FROM slow_seed
                    UNION ALL
                    SELECT
                        o.row_num,
                        o.fetched_at,
                        (? * o.mark_px) + (? * slow_ema.ema) AS ema
                    FROM ordered o
                    JOIN slow_ema ON o.row_num = slow_ema.row_num + 1
                ),
                macd_base AS (
                    SELECT
                        f.fetched_at,
                        f.ema AS ema_fast,
                        s.ema AS ema_slow,
                        f.ema - s.ema AS macd,
                        ROW_NUMBER() OVER (ORDER BY f.fetched_at) AS macd_row
                    FROM fast_ema f
                    JOIN slow_ema s
                      ON f.fetched_at = s.fetched_at
                ),
                signal_seed AS (
                    SELECT macd_row, fetched_at, signal
                    FROM (
                        SELECT
                            macd_row,
                            fetched_at,
                            AVG(macd) OVER (
                                ORDER BY macd_row
                                ROWS BETWEEN {signal - 1} PRECEDING AND CURRENT ROW
                            ) AS signal
                        FROM macd_base
                    )
                    WHERE macd_row = {signal}
                ),
                signal_ema AS (
                    SELECT macd_row, fetched_at, signal
                    FROM signal_seed
                    UNION ALL
                    SELECT
                        m.macd_row,
                        m.fetched_at,
                        (signal_ema.signal * {signal - 1} + m.macd) / {signal} AS signal
                    FROM macd_base m
                    JOIN signal_ema ON m.macd_row = signal_ema.macd_row + 1
                )
                SELECT
                    m.fetched_at,
                    m.macd,
                    s.signal,
                    m.macd - s.signal AS histogram
                FROM macd_base m
                JOIN signal_ema s ON m.macd_row = s.macd_row
                ORDER BY m.fetched_at ASC
                { "LIMIT ?" if limit is not None else "" }
                """ 
        rows = self._repo._conn.execute(query, params).fetchall()
        return [
                (
                    ts,
                    {
                        "macd": float(macd),
                        "signal": float(sig),
                        "hist": float(macd - sig) if hist is None else float(hist),
                    },
                )
                for ts, macd, sig, hist in rows
        ]
        
    
    def _compute_rsi_db(
        self,
        symbol: str,
        window: int,
        limit: int | None = None,
    ) -> list[tuple[datetime, float | None]]:
        params = [symbol.upper()]

        limit_clause = "LIMIT ?" if limit is not None else ""

        if limit is not None:
            params.append(limit)

        ordered = """
            ordered AS (
                SELECT
                    fetched_at,
                    mark_px,
                    ROW_NUMBER() OVER (ORDER BY fetched_at) AS row_num
                FROM perp_asset_ctxs
                WHERE symbol = ?
            ),
        """

        deltas = """
            deltas AS (
                SELECT
                    row_num,
                    fetched_at,
                    mark_px,
                    mark_px - LAG(mark_px) OVER (ORDER BY fetched_at) AS diff,
                    GREATEST(mark_px - LAG(mark_px) OVER (ORDER BY fetched_at), 0) AS gain,
                    GREATEST(LAG(mark_px) OVER (ORDER BY fetched_at) - mark_px, 0) AS loss
                FROM ordered
            ),
        """
        
        seed = f"""
           seed AS (
                SELECT row_num, fetched_at, avg_gain, avg_loss
                FROM (
                    SELECT
                        row_num,
                        fetched_at,
                        AVG(gain) OVER (
                            ORDER BY row_num
                            ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                        ) AS avg_gain,
                        AVG(loss) OVER (
                            ORDER BY row_num
                            ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                        ) AS avg_loss
                    FROM deltas
                )
                WHERE row_num = {window + 1}
            ),
        """

        ema = f"""
           ema AS (
                SELECT
                    row_num,
                    fetched_at,
                    avg_gain AS ema_gain,
                    avg_loss AS ema_loss
                FROM seed

                UNION ALL

                SELECT
                    d.row_num,
                    d.fetched_at,
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
                fetched_at,
                CASE
                    WHEN ema_loss = 0 THEN 100
                    ELSE 100 - 100 / (1 + ema_gain / ema_loss)
                END AS rsi
            FROM ema
            ORDER BY fetched_at ASC
            {limit_clause}
        """
        rows = self._repo._conn.execute(query, params).fetchall()
        return [(ts, float(val) if val is not None else None) for ts, val in rows]

    def _compute_ema_db(
        self,
        symbol: str,
        window: int,
        limit: int | None = None,
    ) -> list[tuple[datetime, float | None]]:
        alpha = 2 / (window + 1)
        beta = 1 - alpha
        params = [symbol.upper(), alpha, beta]

        limit_clause = "LIMIT ?" if limit is not None else ""

        if limit is not None:
            params.append(limit)

        ordered = """
            ordered AS (
                SELECT
                    fetched_at,
                    mark_px,
                    ROW_NUMBER() OVER (ORDER BY fetched_at) AS row_num
                FROM perp_asset_ctxs
                WHERE symbol = ?
            ),
        """

        seed = f"""
            seed AS (
                SELECT row_num, fetched_at, ema
                FROM (
                    SELECT
                        row_num,
                        fetched_at,
                        AVG(mark_px) OVER (
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
                SELECT row_num, fetched_at, ema
                FROM seed

                UNION ALL

                SELECT
                    o.row_num,
                    o.fetched_at,
                    (? * o.mark_px) + (? * ema.ema) AS ema
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
            SELECT fetched_at, ema
            FROM ema
            ORDER BY fetched_at ASC
            {limit_clause}
        """

        rows = self._repo._conn.execute(query, params).fetchall()
        return [(ts, float(val) if val is not None else None) for ts, val in rows]

    def _compute_sma_db(
        self,
        symbol: str,
        window: int,
        limit: int | None = None,
    ) -> list[tuple[datetime, float | None]]:
        if not symbol:
            raise ValueError("symbol must be provided")
        if window < 1:
            raise ValueError("window must be >= 1")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1 when provided")

        params: list[Any] = [symbol.upper()]
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)

        query = f"""
            WITH ordered AS (
                SELECT
                    fetched_at,
                    mark_px,
                    ROW_NUMBER() OVER (ORDER BY fetched_at) AS row_num,
                    AVG(mark_px) OVER (
                        ORDER BY fetched_at
                        ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                    ) AS sma_raw
                FROM perp_asset_ctxs
                WHERE symbol = ?
            )
            SELECT
                fetched_at,
                CASE WHEN row_num >= {window} THEN sma_raw ELSE NULL END AS sma
            FROM ordered
            ORDER BY fetched_at ASC
            {limit_clause}
        """

        rows: Sequence[tuple[datetime, float | None]] = (
            self._repo._conn.execute(
                query,
                params,
            ).fetchall()
        )

        return [
            (fetched_at, float(sma) if sma is not None else None)
            for fetched_at, sma in rows
        ]
    
    @staticmethod
    def _validate_window(window: int | None, indicator: IndicatorType) -> int:
        if window is None or window < 1:
            raise ValueError(f"window must be >= 1 for {indicator.value.upper()}")
        return window



class IndicatorType(str, Enum):
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    VWAP = "vwap"

@dataclass
class IndicatorResult:
    symbol: str
    indicator: IndicatorType
    params: dict[str, Any]
    series: list[tuple[datetime, float | dict[str, float] | None]]
