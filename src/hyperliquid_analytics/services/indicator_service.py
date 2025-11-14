from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Sequence

from dataclasses import dataclass

from duckdb import query
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
        match indicator:
            case IndicatorType.SMA:
                if window is None or window < 1:
                    raise ValueError("window must be >= 1 for SMA")
                series = await asyncio.to_thread(
                    self._compute_sma_db,
                    symbol,
                    window,
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


    def _compute_sma(
            self, 
            symbol: str, 
            window: int, 
            limit: int | None = None
        ) -> list[tuple[datetime, float | None]]:

        placeholders: list[Any] = [symbol.upper()]
        limit_clause:str =  ""

        if limit is not None:
            limit_clause = "LIMIT ?"
            placeholders.append(limit)

            query = f"""
                SELECT fetched_at,
                       AVG(mark_px) OVER (
                           ORDER BY fetched_at
                           ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                        ) AS sma
                FROM perp_asset_ctxs
                WHERE symbol = ?
                ORDER BY fetched_at ASC
                {limit_clause}
            """

            rows: Sequence[tuple[datetime, float | None]] = self._repo._conn.execute(
                    query,
                    placeholders,
                    ).fetchall()

            return [(fetched_at, sma) for fetched_at, sma in rows]




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
    series: list[tuple[datetime, float | dict]]
