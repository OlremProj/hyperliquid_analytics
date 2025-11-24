from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple


class IndicatorType(str, Enum):
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    ATR = "atr"
    STOCHASTIC = "STOCHASTIC"
    VWAP = "vwap"


@dataclass
class IndicatorPoint:
    timestamp: datetime
    values: Dict[str, Optional[float]]


@dataclass
class IndicatorResult:
    symbol: str
    indicator: IndicatorType
    metadata: dict
    points: Sequence[IndicatorPoint]

    def to_rows(self) -> list[dict]:
        return [
            {
                "symbol": self.symbol,
                "indicator": self.indicator.value,
                "timestamp": point.timestamp.isoformat(),
                **{k: v for k, v in point.values.items()},
            }
            for point in self.points
        ]

    def with_metadata(self, meta: dict) -> "IndicatorResult":
        merged = {**self.metadata, **(meta or {})}
        return IndicatorResult(
            symbol=self.symbol,
            indicator=self.indicator,
            metadata=merged,
            points=self.points,
        )


def wrap_scalar_series(
    symbol: str,
    indicator: IndicatorType,
    rows: Sequence[Tuple[datetime, Optional[float]]],
    *,
    field: str = "value",
    metadata: Optional[dict] = None,
) -> IndicatorResult:
    points = [
        IndicatorPoint(
            timestamp=ts,
            values={field: float(val) if val is not None else None},
        )
        for ts, val in rows
    ]
    return IndicatorResult(
        symbol=symbol.upper(),
        indicator=indicator,
        metadata=metadata or {},
        points=points,
    )


def wrap_record_series(
    symbol: str,
    indicator: IndicatorType,
    rows: Sequence[Tuple],
    *,
    fields: Tuple[str, ...],
    metadata: Optional[dict] = None,
) -> IndicatorResult:
    wrapped_rows = []
    for row in rows:
        ts = row[0]
        values = {
            field: (
                float(value)
                if isinstance(value, (int, float)) and value is not None
                else value
            )
            for field, value in zip(fields, row[1:])
        }
        wrapped_rows.append(IndicatorPoint(timestamp=ts, values=values))
    return IndicatorResult(
        symbol=symbol.upper(),
        indicator=indicator,
        metadata=metadata or {},
        points=wrapped_rows,
    )
