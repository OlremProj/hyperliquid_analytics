from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.models.indicator_result_models import IndicatorType
from hyperliquid_analytics.models.signal_models import Signal
from hyperliquid_analytics.services.pipeline.interfaces import BaseStrategy

class RsiStrategy(BaseStrategy):

    async def analyze(self, symbol: str, timeframe: TimeFrame) -> Signal | None:
        rsi_result = await self.indicators.compute_indicator(
                    symbol=symbol,
                    indicator=IndicatorType.RSI,
                    timeframe=timeframe,
                    window=14,
        )
        last_rsi = rsi_result.points[-1].values["rsi"]
        print(f" [DEBUG] {symbol} RSI: {last_rsi}")


