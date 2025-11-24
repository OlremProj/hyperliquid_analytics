from __future__ import annotations

from typing import Any

from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.models.indicator_result_models import (
    IndicatorResult,
    IndicatorType,
    wrap_record_series,
    wrap_scalar_series,
)
from hyperliquid_analytics.repository.perp_repository import PerpRepository


def compute_rsi(
    repo: PerpRepository,
    symbol: str,
    timeframe: TimeFrame,
    window: int,
    limit: int | None,
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
        seed AS (
            SELECT row_num, timestamp, avg_gain, avg_loss
            FROM (
                SELECT
                    row_num,
                    timestamp,
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
    rows = repo._conn.execute(query, params).fetchall()
    return wrap_scalar_series(
        symbol,
        IndicatorType.RSI,
        rows,
        field="value",
        metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
    )


def compute_atr(
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
    rows = repo._conn.execute(query, params).fetchall()
    return wrap_scalar_series(
        symbol,
        IndicatorType.ATR,
        rows,
        field="value",
        metadata={"window": window, "limit": limit, "timeframe": timeframe.value},
    )


def compute_stochastic(
    repo: PerpRepository,
    symbol: str,
    timeframe: TimeFrame,
    window: int,
    signal_window: int,
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
    rows = repo._conn.execute(query, params).fetchall()
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

