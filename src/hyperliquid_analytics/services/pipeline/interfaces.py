from abc import ABC, abstractmethod 

from hyperliquid_analytics.services.indicator_service import IndicatorService
from hyperliquid_analytics.models.data_models import TimeFrame

from hyperliquid_analytics.models.signal_models import Signal 

class BaseStrategy(ABC):
    def __init__(self, indicator_service: IndicatorService):
        self.indicators = indicator_service

    @abstractmethod
    async def analyze(self, symbol: str, timeframe: TimeFrame) -> Signal | None:
        """
        Analyse le marché et retourne un Signal si une condition est remplie.
        Retourne None si rien à signaler.
        """
        pass
