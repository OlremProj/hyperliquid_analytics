from __future__ import annotations

from typing import Any

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.models.indicator_result_models import (
    IndicatorResult,
    IndicatorType,
    wrap_record_series,
)
from hyperliquid_analytics.repository.perp_repository import PerpRepository


def compute_bollinger(
    repo: PerpRepository,
    symbol: str,
    timeframe: TimeFrame,
    window: int,
    k: float,
    limit: int | None,
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
    rows = repo._conn.execute(query, params).fetchall()
    return wrap_record_series(
        symbol,
        IndicatorType.BOLLINGER,
        rows,
        fields=("middle", "upper", "lower"),
        metadata={"window": window, "k": k, "limit": limit, "timeframe": timeframe.value},
    )

