from time import time
from typing import List
from hyperliquid_analytics.models.data_models import TimeFrame
from hyperliquid_analytics.models.signal_models import Signal
from hyperliquid_analytics.services.pipeline.interfaces import BaseStrategy

class AnalysisPipeline:
    def __init__(self) -> None:
        self.strategies: List[BaseStrategy] = []

    def add_strategy(self, strategy: BaseStrategy):
        self.strategies.append(strategy)

    async def process(self, symbol: str, timeframe: TimeFrame) -> List[Signal]:
        signals = []
        for strategy in self.strategies:
            signal = await strategy.analyze(symbol, timeframe)
            if signal:
                signals.append(signal)
                print(f"🔴 SIGNAL DÉTECTÉ: {signal.model_dump_json()}")

        return signals
