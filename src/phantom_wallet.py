import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
import base58
from src.api.alchemy import AlchemyClient
import asyncio
from datetime import datetime
import os
from solders.keypair import Keypair
from solana.transaction import Transaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment
from solana.rpc.types import TxOpts
from solders.pubkey import Pubkey
from solders.hash import Hash

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
        
        # Live trading components
        self.keypair: Optional[Keypair] = None
        self.rpc_client: Optional[AsyncClient] = None
        self.live_mode: bool = False

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
            
            # Close RPC client if in live mode
            if self.rpc_client:
                await self.rpc_client.close()
                self.rpc_client = None

            self.connected = False
            self.live_mode = False
            self.wallet_address = None
            self.keypair = None
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
    
    def _load_keypair_from_private_key(self, private_key_b58: str) -> Keypair:
        """
        Create Solana keypair from base58 private key
        
        Args:
            private_key_b58: Base58 encoded private key string
            
        Returns:
            Keypair: Solana keypair object
            
        Raises:
            ValueError: If private key is invalid
        """
        try:
            if not private_key_b58 or not isinstance(private_key_b58, str):
                raise ValueError("Private key must be a non-empty string")
                
            private_key_bytes = base58.b58decode(private_key_b58)
            
            if len(private_key_bytes) != 64:
                raise ValueError("Private key must be 64 bytes when decoded")
                
            keypair = Keypair.from_secret_key(private_key_bytes)
            logger.info(f"Successfully loaded keypair with public key: {keypair.public_key}")
            return keypair
            
        except Exception as e:
            logger.error(f"Failed to load keypair from private key: {str(e)}")
            raise ValueError(f"Invalid private key: {str(e)}")
    
    async def initialize_live_mode(self, rpc_url: Optional[str] = None) -> bool:
        """
        Initialize live trading mode with private key and RPC connection
        
        Args:
            rpc_url: Optional custom RPC URL
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            private_key = os.getenv('PRIVATE_KEY')
            if not private_key:
                logger.error("PRIVATE_KEY environment variable not found")
                return False
                
            self.keypair = self._load_keypair_from_private_key(private_key)
            
            if not rpc_url:
                network = os.getenv('SOLANA_NETWORK', 'mainnet-beta')
                if network == 'mainnet-beta':
                    rpc_url = "https://api.mainnet-beta.solana.com"
                elif network == 'devnet':
                    rpc_url = "https://api.devnet.solana.com"
                else:
                    rpc_url = "https://api.testnet.solana.com"
                    
            self.rpc_client = AsyncClient(rpc_url)
            
            # Test connection
            try:
                await self.rpc_client.get_latest_blockhash()
                logger.info(f"Successfully connected to Solana RPC at {rpc_url}")
            except Exception as e:
                logger.error(f"Failed to connect to RPC: {str(e)}")
                return False
                
            # Update wallet address to match keypair
            self.wallet_address = str(self.keypair.public_key)
            self.live_mode = True
            
            logger.info(f"Live mode initialized for wallet: {self.wallet_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize live mode: {str(e)}")
            return False
    
    async def get_recent_blockhash(self, priority_fee: Optional[int] = None) -> str:
        """
        Get recent blockhash for transaction building
        
        Args:
            priority_fee: Optional priority fee in microlamports
            
        Returns:
            str: Recent blockhash
            
        Raises:
            RuntimeError: If not in live mode or RPC client unavailable
        """
        if not self.live_mode or not self.rpc_client:
            raise RuntimeError("Live mode not initialized")
            
        try:
            response = await self.rpc_client.get_latest_blockhash()
            if not response.value:
                raise RuntimeError("Failed to get recent blockhash")
                
            return str(response.value.blockhash)
            
        except Exception as e:
            logger.error(f"Error getting recent blockhash: {str(e)}")
            raise
    
    async def sign_and_send_transaction(self, transaction: Transaction, rpc_url: Optional[str] = None) -> Optional[str]:
        """
        Sign and send a transaction to the Solana network
        
        Args:
            transaction: Transaction to sign and send
            rpc_url: Optional custom RPC URL
            
        Returns:
            Optional[str]: Transaction signature if successful, None otherwise
        """
        if not self.live_mode or not self.keypair or not self.rpc_client:
            logger.error("Live mode not initialized - cannot sign transactions")
            return None
            
        try:
            # Sign the transaction
            transaction.sign(self.keypair)
            
            # Send the transaction
            signature = await self.submit_transaction(transaction)
            
            if signature:
                logger.info(f"Transaction sent successfully: {signature}")
                self.last_signature = signature
                return signature
            else:
                logger.error("Failed to submit transaction")
                return None
                
        except Exception as e:
            logger.error(f"Error signing and sending transaction: {str(e)}")
            return None
    
    async def submit_transaction(self, signed_tx: Transaction) -> str:
        """
        Submit a signed transaction to the network
        
        Args:
            signed_tx: Signed transaction to submit
            
        Returns:
            str: Transaction signature
            
        Raises:
            RuntimeError: If submission fails
        """
        if not self.rpc_client:
            raise RuntimeError("RPC client not initialized")
            
        try:
            opts = TxOpts(skip_confirmation=False, skip_preflight=False)
            response = await self.rpc_client.send_transaction(
                signed_tx,
                opts=opts
            )
            
            if response.value:
                return str(response.value)
            else:
                raise RuntimeError("Transaction submission failed - no signature returned")
                
        except Exception as e:
            logger.error(f"Error submitting transaction: {str(e)}")
            raise
    
    async def get_transaction_status(self, signature: str) -> str:
        """
        Get the status of a transaction
        
        Args:
            signature: Transaction signature to check
            
        Returns:
            str: Transaction status ('confirmed', 'finalized', 'failed', 'pending')
        """
        if not self.rpc_client:
            raise RuntimeError("RPC client not initialized")
            
        try:
            response = await self.rpc_client.get_signature_status(signature)
            
            if not response.value:
                return "pending"
                
            status = response.value[0]
            if not status:
                return "pending"
                
            if status.err:
                return "failed"
            elif status.confirmation_status:
                return status.confirmation_status.value
            else:
                return "confirmed"
                
        except Exception as e:
            logger.error(f"Error getting transaction status: {str(e)}")
            return "unknown"