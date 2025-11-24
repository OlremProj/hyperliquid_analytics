from typing import Any
from time import time
from datetime import datetime, timezone

import logging

from hyperliquid_analytics.models.perp_models import MarginTableEntry, PerpMeta, PerpUniverseAsset, PerpAssetContext, MetaAndAssetCtxsResponse
from hyperliquid_analytics.config import Settings
from hyperliquid_analytics.api.client_api import ApiClient
from hyperliquid_analytics.models.data_models import OHLCVData, TimeFrame, MarketData

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

    async def fetch_ohlcv(self, symbol: str, timeframe: TimeFrame, limit: int) -> MarketData:
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
                symbol=symbol.upper(),
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
    
        
    async def fetch_user_fills(self, user:str, start_ms, end_ms, aggregate=False) -> dict[str, Any]:
        payload = {
            "type": "userFills",
            "user": user,
            "startTime": start_ms,
            "endTime": end_ms,
            "aggregateByTime": aggregate,
            }
        return await self._request(payload)

    async def fetch_meta_and_asset_contexts(self) -> MetaAndAssetCtxsResponse:
        payload={
                "type": "metaAndAssetCtxs",
                }
        raw_meta, asset_ctxs = await self._request(payload)

        perp_universe = [
            PerpUniverseAsset.model_validate(item)
            for item in raw_meta.get("universe", [])
        ]
        margin_tables = [
            MarginTableEntry.model_validate(item)
            for item in raw_meta.get("marginTables", [])
        ]
        contexts = [
            PerpAssetContext.model_validate(item) 
            for item in asset_ctxs
        ]

        perp_meta = PerpMeta(universe=perp_universe, margin_tables=margin_tables)

        meta_assets = MetaAndAssetCtxsResponse(meta=perp_meta, asset_contexts=contexts)
        return meta_assets
