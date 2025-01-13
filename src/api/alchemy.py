import asyncio
import logging
from typing import Dict, Any, Optional, List
import aiohttp
import ssl
import certifi
from solana.rpc import async_api
from solana.rpc.commitment import Confirmed
import traceback
from aiohttp import TCPConnector

logger = logging.getLogger(__name__)

class AlchemyClient:
    """Client for interacting with Alchemy RPC API"""

    def __init__(self, rpc_url: str) -> None:
        """Initialize the client with RPC URL"""
        if not rpc_url:
            raise ValueError("RPC URL is required")

        logger.info(f"Initializing Alchemy client with URL: {rpc_url}")
        self.rpc_url = rpc_url

        # Create SSL context with proper certificate handling
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

        # Initialize the Solana client
        self.client = async_api.AsyncClient(rpc_url, commitment=Confirmed)
        self.session: Optional[aiohttp.ClientSession] = None

    async def ensure_session(self) -> None:
        """Ensure aiohttp session exists and is active"""
        if self.session is None or self.session.closed:
            connector = TCPConnector(
                ssl=self.ssl_context,
                force_close=True,  # Ensure connections are properly closed
                enable_cleanup_closed=True  # Clean up closed connections
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
            )

    async def test_connection(self) -> bool:
        for attempt in range(3):  # Try 3 times
            try:
                await self.ensure_session()
                data = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getHealth"
                }
                async with self.session.post(self.rpc_url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if 'result' in result and result['result'] == 'ok':
                            return True
                    # Add more detailed error logging
                    logger.error(f"Connection test failed with status {response.status}")
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Error response: {error_text}")

                # Exponential backoff between retries
                await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientSSLError as e:
                logger.error(f"SSL Error on attempt {attempt + 1}: {str(e)}")
                if attempt == 2:  # Last attempt
                    logger.error(f"SSL Certificate details: {self.ssl_context.get_ca_certs()}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}\n{traceback.format_exc()}")
                await asyncio.sleep(2 ** attempt)
        return False

    async def get_balance(self, wallet_address: str) -> float:
        """Get SOL balance for a wallet address"""
        try:
            await self.ensure_session()
            data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [wallet_address]
            }
            async with self.session.post(self.rpc_url, json=data) as response:
                response_data = await response.json()
                if 'result' in response_data:
                    return float(response_data['result']['value']) / 1e9  # Convert lamports to SOL
                return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}\n{traceback.format_exc()}")
            return 0.0

    async def get_token_accounts(self, wallet_address: str) -> Dict[str, Any]:
        """Get all token accounts owned by a wallet"""
        try:
            await self.ensure_session()
            data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    wallet_address,
                    {
                        "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
                    },
                    {
                        "encoding": "jsonParsed"
                    }
                ]
            }
            async with self.session.post(self.rpc_url, json=data) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting token accounts: {str(e)}\n{traceback.format_exc()}")
            return {"result": {"value": []}}

    async def get_token_balance(self, token_address: str) -> float:
        """Get balance for a specific token account"""
        try:
            data = await self._make_request("getTokenAccountBalance", [token_address])
            if data and 'result' in data:
                return float(data['result'].get('value', 0))
            return 0.0
        except Exception as e:
            logger.error(f"Error getting token balance: {str(e)}")
            return 0.0

    async def _make_request(self, method: str, params: List[Any]) -> Optional[Dict]:
        """Make a generic RPC request to Alchemy"""
        try:
            await self.ensure_session()
            data = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params
            }
            async with self.session.post(self.rpc_url, json=data) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Request error: {str(e)}\n{traceback.format_exc()}")
            return None

    async def close(self) -> None:
        """Close all client connections"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            if self.client:
                await self.client.close()
        except Exception as e:
            logger.error(f"Error closing client: {str(e)}\n{traceback.format_exc()}")