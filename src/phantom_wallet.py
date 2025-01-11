import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
import base58
from src.api.alchemy import AlchemyClient
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

# Type aliases
CallbackType = Callable[[Dict[str, Any]], Awaitable[None]]
TokenAccountType = Dict[str, Any]
JsonRpcResponse = Dict[str, Any]

class PhantomWallet:
    """Class to interact with Solana wallets through Alchemy API"""

    def __init__(self, alchemy_client: AlchemyClient) -> None:
        """
        Initialize PhantomWallet with an Alchemy client

        Args:
            alchemy_client: An initialized AlchemyClient instance
        """
        if not alchemy_client:
            raise ValueError("Alchemy client is required")

        self.client: AlchemyClient = alchemy_client
        self.connected: bool = False
        self.wallet_address: Optional[str] = None
        self.token_accounts: Dict[str, str] = {}
        self.last_signature: Optional[str] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self.last_update: Optional[datetime] = None

    async def connect(self, wallet_address: str) -> bool:
        """
        Connect to wallet and verify it's accessible

        Args:
            wallet_address: The wallet's public key address

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to wallet: {wallet_address}")

            if not self._validate_address(wallet_address):
                logger.error("Invalid wallet address format")
                return False

            self.wallet_address = wallet_address

            # Verify wallet by checking balance
            balance = await self.get_balance()
            if balance is not None:
                self.connected = True
                self.last_update = datetime.now()
                await self._initialize_token_accounts()
                logger.info(f"Successfully connected to wallet with balance: {balance} SOL")
                return True

            logger.error("Could not verify wallet balance")
            return False

        except Exception as e:
            logger.error(f"Error connecting to wallet: {str(e)}")
            return False

    async def get_balance(self) -> Optional[float]:
        """
        Get wallet's SOL balance

        Returns:
            Optional[float]: Balance in SOL or None if error occurs
        """
        try:
            if not self.wallet_address:
                raise ValueError("No wallet address provided")

            balance = await self.client.get_balance(self.wallet_address)
            self.last_update = datetime.now()
            return balance

        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return None

    async def _initialize_token_accounts(self) -> None:
        """Initialize token accounts mapping"""
        try:
            if not self.wallet_address:
                return

            response = await self.client.get_token_accounts(self.wallet_address)

            self.token_accounts.clear()
            if isinstance(response, dict) and 'result' in response:
                accounts = response['result'].get('value', [])
                for account in accounts:
                    account_data = account.get('account', {})
                    account_info = account_data.get('data', {}).get('parsed', {}).get('info', {})
                    mint = account_info.get('mint')
                    account_pubkey = account.get('pubkey')
                    if mint and account_pubkey:
                        self.token_accounts[mint] = account_pubkey

            logger.info(f"Initialized {len(self.token_accounts)} token accounts")

        except Exception as e:
            logger.error(f"Error initializing token accounts: {str(e)}")

    async def get_token_accounts(self) -> List[TokenAccountType]:
        """
        Get all token accounts for the wallet

        Returns:
            List[TokenAccountType]: List of token account information
        """
        try:
            if not self.wallet_address:
                return []

            accounts: List[TokenAccountType] = []
            for mint, pubkey in self.token_accounts.items():
                balance = await self.get_token_balance(mint)
                accounts.append({
                    'mint': mint,
                    'pubkey': pubkey,
                    'balance': balance
                })

            return accounts

        except Exception as e:
            logger.error(f"Error getting token accounts: {str(e)}")
            return []

    async def get_token_balance(self, token_address: str) -> Optional[float]:
        """
        Get balance of specific token

        Args:
            token_address: The token's mint address

        Returns:
            Optional[float]: Token balance or None if error occurs
        """
        try:
            if not self.connected:
                raise ValueError("Wallet not connected")

            account = self.token_accounts.get(token_address)
            if not account:
                return 0.0

            response = await self.client.get_token_balance(account)

            if response and isinstance(response, dict) and 'value' in response:
                value = response['value']
                amount = float(value.get('amount', 0))
                decimals = int(value.get('decimals', 0))
                return amount / (10 ** decimals)

            return 0.0

        except Exception as e:
            logger.error(f"Error getting token balance: {str(e)}")
            return None

    async def monitor_transactions(
        self,
        callback: Optional[CallbackType] = None,
        interval: float = 1.0
    ) -> bool:
        """
        Monitor wallet transactions

        Args:
            callback: Async function to call when new transaction is detected
            interval: Time between checks in seconds

        Returns:
            bool: True if monitoring started successfully, False otherwise
        """
        try:
            if not self.wallet_address or not self.connected:
                logger.error("Cannot monitor transactions: wallet not connected")
                return False

            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()

            async def monitor_loop() -> None:
                while self.connected:
                    try:
                        response = await self.client._make_request(
                            "getSignaturesForAddress",
                            [self.wallet_address, {"limit": 1}]
                        )

                        if isinstance(response, dict) and 'result' in response:
                            signatures = response['result']
                            if signatures:
                                newest_sig = signatures[0].get('signature')
                                if newest_sig and newest_sig != self.last_signature:
                                    tx_info: JsonRpcResponse = {
                                        'signature': newest_sig,
                                        'timestamp': datetime.now().isoformat(),
                                        'wallet': self.wallet_address
                                    }
                                    if callback:
                                        await callback(tx_info)
                                    self.last_signature = newest_sig
                                    self.last_update = datetime.now()

                        await asyncio.sleep(interval)

                    except asyncio.CancelledError:
                        logger.info("Transaction monitoring cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Error in monitoring loop: {str(e)}")
                        await asyncio.sleep(interval * 5)

            self._monitor_task = asyncio.create_task(monitor_loop())
            logger.info("Transaction monitoring started")
            return True

        except Exception as e:
            logger.error(f"Error setting up transaction monitoring: {str(e)}")
            return False

    def _validate_address(self, address: str) -> bool:
        """
        Validate Solana address format

        Args:
            address: The address to validate

        Returns:
            bool: True if address is valid, False otherwise
        """
        try:
            decoded = base58.b58decode(address)
            return len(decoded) == 32
        except:
            return False

    async def disconnect(self) -> None:
        """Clean up and disconnect wallet"""
        try:
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass

            self.connected = False
            self.wallet_address = None
            self.token_accounts.clear()
            self.last_signature = None
            self.last_update = None
            logger.info("Wallet disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting wallet: {str(e)}")

    def is_stale(self, max_age_seconds: float = 300.0) -> bool:
        """
        Check if wallet data is stale

        Args:
            max_age_seconds: Maximum age in seconds before considering data stale

        Returns:
            bool: True if data is stale, False otherwise
        """
        if not self.last_update:
            return True
        age = (datetime.now() - self.last_update).total_seconds()
        return age > max_age_seconds