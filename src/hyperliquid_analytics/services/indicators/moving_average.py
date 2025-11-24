from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.models.indicator_result_models import (
    IndicatorResult,
    IndicatorType,
    wrap_scalar_series,
)
from hyperliquid_analytics.repository.perp_repository import PerpRepository


def compute_sma(
    repo: PerpRepository,
    symbol: str,
    timeframe: TimeFrame,
    window: int,
    limit: int | None,
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

    rows: Sequence[tuple[datetime, float | None]] = repo._conn.execute(query, params).fetchall()
    return wrap_scalar_series(
        symbol,
        IndicatorType.SMA,
        rows,
        field="value",
        metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
    )


def compute_ema(
    repo: PerpRepository,
    symbol: str,
    timeframe: TimeFrame,
    window: int,
    limit: int | None,
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

    rows = repo._conn.execute(query, params).fetchall()

    return wrap_scalar_series(
        symbol,
        IndicatorType.EMA,
        rows,
        field="value",
        metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
    )

