import asyncio
from time import time
from src.hyperliquid_analytics.api.hyperliquid_client import HyperliquidClient

async def main():
    client = HyperliquidClient()
    data = await client.fetch_user_fills("0x31ca8395cf837de08b24da3f660e77761dfb974b", int(time() * 1_000) - 7_889_400_000, int(time() * 1_000))
    print(data)

asyncio.run(main())
