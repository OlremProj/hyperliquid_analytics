from enum import Enum
from datetime import datetime
from pydantic import BaseModel

from hyperliquid_analytics.models.data_models import TimeFrame

class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class Signal(BaseModel):
    symbol: str
    timeframe: TimeFrame
    timestamp: datetime 
    signal_type: SignalType
    strategy_name: str
    reason: str
    metadata: dict | None = None
