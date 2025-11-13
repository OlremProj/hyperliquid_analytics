from datetime import datetime
from hyperliquid_analytics.api.hyperliquid_client import HyperliquidClient 
from hyperliquid_analytics.repository.perp_repository import PerpRepository 

class AnalyticsService:

    def __init__(self) -> None:
        self.perp_repository = PerpRepository()
        self.hl_client = HyperliquidClient()

    async def save_market_data(self):
        meta_assets = await self.hl_client.fetch_meta_and_asset_contexts()
        self.perp_repository.save_snapshot(meta_assets, datetime.now())
        return meta_assets

    async def get_market_data(self, symbol: str):
        return self.perp_repository.fetch_latest(symbol)
