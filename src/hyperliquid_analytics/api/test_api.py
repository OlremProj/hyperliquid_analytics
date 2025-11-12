import asyncio
from src.hyperliquid_analytics.api.hyperliquid_client import HyperliquidClient

async def main():
    client = HyperliquidClient()
    data = await client.fetch_meta_and_asset_contexts()
    print(data)

asyncio.run(main())
