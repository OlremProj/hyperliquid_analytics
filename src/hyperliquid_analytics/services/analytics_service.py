import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperliquid_analytics.api.hyperliquid_client import HyperliquidClient
from hyperliquid_analytics.models.data_models import OHLCVData, TimeFrame
from hyperliquid_analytics.models.perp_models import PerpAssetContext
from hyperliquid_analytics.repository.perp_repository import PerpRepository

DEFAULT_CANDLE_FETCH = 200
MAX_CANDLE_FETCH = 500

class AnalyticsService:

    def __init__(self, db_path: Path | None = None) -> None:
        self.perp_repository = PerpRepository(db_path=db_path)
        self.hl_client = HyperliquidClient()

    async def save_market_data(self):
        meta_assets = await self.hl_client.fetch_meta_and_asset_contexts()
        fetched_at = datetime.now(timezone.utc)
        await asyncio.to_thread(
                self.perp_repository.save_snapshot,
                meta_assets,
                fetched_at,
        )
        return meta_assets, fetched_at

    async def get_market_data(self, symbol: str):
        return await asyncio.to_thread(self.perp_repository.fetch_latest, symbol)

    async def get_market_history(self, 
                                 symbol: str, 
                                 *, 
                                 since: datetime | None = None, 
                                 limit: int | None = None, 
                                 ascending: bool = False
    ) -> list[tuple[datetime, PerpAssetContext]]:
        return await asyncio.to_thread(
            self.perp_repository.fetch_history,
            symbol,
            since=since,
            limit=limit,
            ascending=ascending,
        )

    async def save_candles(
        self, symbol: str, timeframe: TimeFrame, limit: int | None = None
    ) -> dict[str, Any]:
        tf = timeframe if isinstance(timeframe, TimeFrame) else TimeFrame(timeframe)
        symbol_norm = symbol.upper()
        last_ts = await asyncio.to_thread(
            self.perp_repository.fetch_latest_candle_timestamp,
            symbol_norm,
            tf.value,
        )
        fetch_count = limit if limit and limit > 0 else None
        now_ts = datetime.now(timezone.utc)

        if fetch_count is None:
            if last_ts is None:
                fetch_count = DEFAULT_CANDLE_FETCH
            else:
                millis_gap = (now_ts - last_ts).total_seconds() * 1000
                missing = int(millis_gap // tf.millis)
                if missing <= 1:
                    return {
                        "symbol": symbol_norm,
                        "timeframe": tf.value,
                        "status": "up_to_date",
                        "requested": 0,
                        "fetched": 0,
                        "last_timestamp": last_ts.isoformat(),
                    }
                fetch_count = min(max(missing + 1, 2), MAX_CANDLE_FETCH)

        if fetch_count is None or fetch_count <= 0:
            return {
                "symbol": symbol_norm,
                "timeframe": tf.value,
                "status": "skipped",
                "requested": 0,
                "fetched": 0,
                "last_timestamp": last_ts.isoformat() if last_ts else None,
            }

        ohlcv_data = await self.hl_client.fetch_ohlcv(symbol_norm, tf, fetch_count)
        if not ohlcv_data.candles:
            return {
                "symbol": symbol_norm,
                "timeframe": tf.value,
                "status": "no_data",
                "requested": fetch_count,
                "fetched": 0,
                "last_timestamp": last_ts.isoformat() if last_ts else None,
            }

        await asyncio.to_thread(
            self.perp_repository.save_candles,
            symbol_norm,
            tf.value,
            ohlcv_data.candles,
        )

        latest_ts = ohlcv_data.candles[-1].timestamp
        return {
            "symbol": symbol_norm,
            "timeframe": tf.value,
            "status": "updated",
            "requested": fetch_count,
            "fetched": len(ohlcv_data.candles),
            "last_timestamp": latest_ts.isoformat(),
        }

    async def get_candles(
        self,
        symbol: str,
        timeframe: TimeFrame,
        *,
        limit: int | None = None,
        since: datetime | None = None,
        ascending: bool = True,
    ) -> list[OHLCVData]:
        return await asyncio.to_thread(
            self.perp_repository.fetch_candles,
            symbol,
            timeframe,
            limit=limit,
            since=since,
            ascending=ascending,
        )
