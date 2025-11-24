from __future__ import annotations

from typing import Any

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.models.indicator_result_models import (
    IndicatorResult,
    IndicatorType,
    wrap_scalar_series,
)
from hyperliquid_analytics.repository.perp_repository import PerpRepository


def compute_vwap(
    repo: PerpRepository,
    symbol: str,
    timeframe: TimeFrame,
    window: int,
    limit: int | None,
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
    rows = repo._conn.execute(query, params).fetchall()
    return wrap_scalar_series(
        symbol,
        IndicatorType.VWAP,
        rows,
        field="value",
        metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
    )

