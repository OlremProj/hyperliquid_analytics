import asyncio
from src.hyperliquid_analytics.models.data_models import TimeFrame
from src.hyperliquid_analytics.api.hyperliquid_client import HyperliquidClient

async def main():
    client = HyperliquidClient()
    data = await client.fetch_ohlcv("BTC", TimeFrame.ONE_HOUR, 500)
    print(data)

asyncio.run(main())
