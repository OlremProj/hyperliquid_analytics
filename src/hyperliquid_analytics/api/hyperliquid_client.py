from typing import Any
from time import time
from datetime import datetime, timezone

import logging

from src.hyperliquid_analytics.config import Settings
from src.hyperliquid_analytics.api.client_api import ApiClient
from src.hyperliquid_analytics.models.data_models import OHLCVData, TimeFrame, MarketData

class HyperliquidAPIError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"[{status}] {message}")
        self.status = status
        self.message = message

class HyperliquidClient:
    
    def __init__(self) -> None:
        self.settings = Settings()
        self._logger = logging.getLogger(__name__)
        pass

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with ApiClient(self.settings.base_url) as api:
            data = await api.post_json("/info", payload)
            self._logger.debug("Payload=%s data=%s", payload, data)
            return data

    async def fetch_ohlcv(self, symbol: str, timeframe: TimeFrame, limit: int) -> dict[str, Any]:
        end_ms = int(time()) * 1_000
        start_ms = end_ms - timeframe.millis*limit

        payload = {
                "type": "candleSnapshot",
                "req":
                    {
                        "coin": symbol,
                        "interval": timeframe.value,
                        "startTime": start_ms,
                        "endTime": end_ms,
                    },
                }
        raw = await self._request(payload)
        candles = [
            OHLCVData(
                timestamp=datetime.fromtimestamp(item["t"] / 1000, tz=timezone.utc),
                open=float(item["o"]),
                high=float(item["h"]),
                low=float(item["l"]),
                close=float(item["c"]),
                volume=float(item["v"]),
            )
            for item in raw
        ]

        return MarketData(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            last_updated=datetime.now(tz=timezone.utc),
        )
