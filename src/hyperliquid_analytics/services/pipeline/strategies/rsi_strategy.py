from time import time
from hyperliquid_analytics.models.indicator_result_models import IndicatorType
from hyperliquid_analytics.services.pipeline.interfaces import BaseStrategy
from hyperliquid_analytics.services.indicator_service import IndicatorService 

class RsiStrategy(BaseStrategy):
    def __init__(self) -> None:
        self.indicator_service = IndicatorService
        pass

    async def analyze(self, symbol: str, timeframe: TimeFrame) -> Signal | None:
        await super().analyze(symbol, timeframe)
        rsi_result = self.indicator_service.compute_indicator(
                                                            symbol,
                                                            IndicatorType.RSI,
                                                            timeframe=timeframe,
                    )


