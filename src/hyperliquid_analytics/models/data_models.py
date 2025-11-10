from enum import Enum
from datetime import datetime
from pydantic import BaseModel, field_validator

class TimeFrame(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"

    @property
    def millis(self) -> int:
        return {
            TimeFrame.ONE_MINUTE: 60_000,
            TimeFrame.FIVE_MINUTES: 300_000,
            TimeFrame.FIFTEEN_MINUTES: 900_000,
            TimeFrame.ONE_HOUR: 3_600_000,
            TimeFrame.FOUR_HOURS: 14_400_000,
            TimeFrame.ONE_DAY: 86_400_000,
        }[self]

    @property
    def label(self) -> str:
        return self.value

class OHLCVData(BaseModel):
    timestamp: datetime
    open: float
    low: float
    high: float
    close: float
    volume: float
        
    @field_validator('high')
    @classmethod
    def validate_high_not_lower_than_low(cls, v, info):
        if 'low' in info.data and v < info.data['low']:
            raise ValueError('High must be >= low')
        return v

    @field_validator('volume')
    @classmethod
    def validate_volume_is_higher_than_zero(cls, v):
        if v < 0:
            raise ValueError("Volume must be >= 0")
        return v

class MarketData(BaseModel):
    symbol: str
    timeframe: TimeFrame
    candles: list[OHLCVData]
    last_updated: datetime
    
    @field_validator('symbol')
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.upper().strip()
    
    @field_validator('candles')
    @classmethod
    def validate_candles_not_empty(cls, v: list[OHLCVData]) -> list[OHLCVData]:
        if len(v) == 0:
            raise ValueError("Candles list cannot be empty")
        return v
