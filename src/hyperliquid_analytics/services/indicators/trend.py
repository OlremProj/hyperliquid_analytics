from __future__ import annotations

from typing import Any

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.models.indicator_result_models import (
    IndicatorResult,
    IndicatorType,
    wrap_record_series,
)
from hyperliquid_analytics.repository.perp_repository import PerpRepository


def compute_macd(
    repo: PerpRepository,
    symbol: str,
    timeframe: TimeFrame,
    fast: int,
    slow: int,
    signal: int,
    limit: int | None,
) -> IndicatorResult:
    alpha_fast = 2 / (fast + 1)
    beta_fast = 1 - alpha_fast
    alpha_slow = 2 / (slow + 1)
    beta_slow = 1 - alpha_slow

    params = [
        symbol.upper(),
        timeframe.value,
        alpha_fast,
        beta_fast,
        alpha_slow,
        beta_slow,
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
    rows = repo._conn.execute(query, params).fetchall()
    return wrap_record_series(
        symbol,
        IndicatorType.MACD,
        rows,
        fields=("macd", "signal", "hist"),
        metadata={
            "fast": fast,
            "slow": slow,
            "signal": signal,
            "limit": limit,
            "timeframe": timeframe.value,
        },
    )

