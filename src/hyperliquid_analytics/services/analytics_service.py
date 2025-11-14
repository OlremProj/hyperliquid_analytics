from datetime import datetime, timezone
import asyncio
from hyperliquid_analytics.api.hyperliquid_client import HyperliquidClient 
from hyperliquid_analytics.repository.perp_repository import PerpRepository 

class AnalyticsService:

    def __init__(self) -> None:
        self.perp_repository = PerpRepository()
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

    async def get_market_history(self, symbol: str, *, since: datetime | None = None, limit: int | None = None, ascending: bool = False):
        return await asyncio.to_thread(
            self.perp_repository.fetch_history,
            symbol,
            since=since,
            limit=limit,
            ascending=ascending,
        )
