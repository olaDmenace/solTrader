import asyncio
import aiohttp
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_alchemy():
    # Try both URLs
    urls = [
        "https://api.mainnet-beta.solana.com"  # Try public RPC as fallback
        "https://solana-mainnet.g.alchemy.com/v2/Y0oIiCzHLttRIv9YRleVRtXbHK9z5LW8",
        "https://mainnet.solana.g.alchemy.com/v2/Y0oIiCzHLttRIv9YRleVRtXbHK9z5LW8",
    ]
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                data = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getHealth",
                }
                logger.info(f"Testing URL: {url}")
                async with session.post(url, json=data) as response:
                    result = await response.json()
                    logger.info(f"Response: {result}")
                    return True
            except Exception as e:
                logger.error(f"Error with {url}: {str(e)}")
    return False

if __name__ == "__main__":
    asyncio.run(test_alchemy())