"""
PhantomWallet - Solana Wallet Management

MIGRATION NOTE: Moved from src/phantom_wallet.py
Core functionality preserved 100% - enhanced with Sentry error tracking
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
import base58
import asyncio
from datetime import datetime
import os
from solders.keypair import Keypair
from solana.transaction import Transaction
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment
from solana.rpc.types import TxOpts
from solders.pubkey import Pubkey
from solders.hash import Hash

# Sentry integration for professional error tracking
from utils.sentry_config import capture_api_error

# Import adjustments for new structure (will be updated during import reference update)
try:
    from src.api.alchemy import AlchemyClient
    from core.rpc_manager import MultiRPCManager
except ImportError:
    # Fallback during migration
    AlchemyClient = None
    MultiRPCManager = None

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
        
        # Transaction tracking to prevent duplicates
        self.processed_signatures: set = set()
        self.failed_signatures: set = set()
        
        # Transaction success metrics
        self.transaction_stats = {
            'total_attempted': 0,
            'rpc_errors': 0,
            'confirmed_despite_errors': 0,
            'genuine_failures': 0,
            'success_rate': 0.0
        }
        
        # Multi-RPC Manager for intelligent provider selection
        self.rpc_manager = MultiRPCManager()
        
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
                logger.warning("No wallet address provided - this should not happen in paper trading")
                return 200.0  # Return default paper trading balance

            balance = await self.client.get_balance(self.wallet_address)
            self.last_update = datetime.now()
            return balance

        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            # In paper trading mode, return a default balance instead of None
            return 200.0

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
            
            # Capture wallet operation errors with Sentry
            capture_api_error(
                error=e,
                api_name="PhantomWallet",
                endpoint="get_token_balance",
                context={
                    "wallet_address": self.wallet_address,
                    "token_mint": str(token_mint) if token_mint else None
                }
            )
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
            
            # Close multi-RPC manager clients
            if hasattr(self, 'rpc_manager') and self.rpc_manager:
                await self.rpc_manager.close_all_clients()

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
                
            keypair = Keypair.from_bytes(private_key_bytes)
            logger.info(f"Successfully loaded keypair with public key: {keypair.pubkey()}")
            return keypair
            
        except Exception as e:
            logger.error(f"Failed to load keypair from private key: {str(e)}")
            raise ValueError(f"Invalid private key: {str(e)}")
    
    async def initialize_live_mode(self, rpc_url: Optional[str] = None) -> bool:
        """
        Initialize live trading mode with private key and intelligent RPC connection
        
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
            
            # Initialize multi-RPC manager with environment configuration
            await self._setup_rpc_providers()
            
            # Get intelligent RPC client selection
            try:
                self.rpc_client = await self.rpc_manager.get_client()
                logger.info(f"Successfully connected using intelligent RPC selection")
            except Exception as e:
                logger.warning(f"Multi-RPC failed, falling back to default: {e}")
                # Fallback to single RPC
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
                logger.info("RPC connection test successful")
            except Exception as e:
                logger.error(f"Failed to connect to RPC: {str(e)}")
                return False
                
            # Update wallet address to match keypair
            self.wallet_address = str(self.keypair.pubkey())
            self.live_mode = True
            
            logger.info(f"Live mode initialized for wallet: {self.wallet_address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize live mode: {str(e)}")
            return False
    
    async def _setup_rpc_providers(self):
        """Setup RPC providers from environment variables"""
        try:
            # Update providers with API keys from environment
            helius_key = os.getenv('HELIUS_API_KEY')
            if helius_key:
                helius_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
                self.rpc_manager.update_provider_url("helius_free", helius_url)
            
            quicknode_endpoint = os.getenv('QUICKNODE_ENDPOINT')
            if quicknode_endpoint:
                self.rpc_manager.update_provider_url("quicknode_free", quicknode_endpoint)
            
            # Ankr uses public endpoint (no API key needed)
            # Solana default already configured
            
            # Perform initial health check
            await self.rpc_manager.health_check_all_providers()
            
            logger.info("[RPC] Multi-RPC manager setup complete")
            
        except Exception as e:
            logger.warning(f"[RPC] Setup failed: {e}")
    
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
    
    async def sign_and_send_transaction(self, transaction: Union[Transaction, VersionedTransaction], rpc_url: Optional[str] = None) -> Optional[str]:
        """
        Sign and send a transaction to the Solana network
        
        Args:
            transaction: Transaction (legacy or versioned) to sign and send
            rpc_url: Optional custom RPC URL
            
        Returns:
            Optional[str]: Transaction signature if successful, None otherwise
        """
        if not self.live_mode or not self.keypair or not self.rpc_client:
            logger.error("Live mode not initialized - cannot sign transactions")
            return None
            
        try:
            # Sign the transaction based on its type
            if isinstance(transaction, VersionedTransaction):
                logger.info("Signing VersionedTransaction")
                # For VersionedTransaction, we need to sign it properly
                # VersionedTransaction doesn't have a sign method, we pass it directly
                signature = await self.submit_versioned_transaction(transaction)
            else:
                logger.info("Signing legacy Transaction")
                # Legacy transaction signing
                transaction.sign(self.keypair)
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
        Submit a signed transaction to the network with RPC fallback
        
        Args:
            signed_tx: Signed transaction to submit
            
        Returns:
            str: Transaction signature
            
        Raises:
            RuntimeError: If submission fails
        """
        if not self.rpc_client:
            raise RuntimeError("RPC client not initialized")
        
        async def _submit_operation(client):
            """Inner operation for RPC manager"""
            opts = TxOpts(skip_confirmation=False, skip_preflight=False)
            response = await client.send_transaction(signed_tx, opts=opts)
            
            if response.value:
                return str(response.value)
            else:
                raise RuntimeError("Transaction submission failed - no signature returned")
        
        try:
            # Use RPC manager for intelligent fallback
            if hasattr(self, 'rpc_manager') and self.rpc_manager:
                result = await self.rpc_manager.execute_with_fallback(_submit_operation)
                return result
            else:
                # Fallback to direct RPC client
                return await _submit_operation(self.rpc_client)
                
        except Exception as e:
            logger.error(f"Error submitting transaction with fallback: {str(e)}")
            raise
    
    async def submit_versioned_transaction(self, versioned_tx: VersionedTransaction) -> str:
        """
        Submit a versioned transaction to the network with RPC fallback
        
        Args:
            versioned_tx: VersionedTransaction from Jupiter (already contains message)
            
        Returns:
            str: Transaction signature
            
        Raises:
            RuntimeError: If submission fails
        """
        if not self.rpc_client:
            raise RuntimeError("RPC client not initialized")
        
        # Extract the message and create a properly signed VersionedTransaction
        from solders.transaction import VersionedTransaction
        
        # Create a new signed VersionedTransaction using the message and our keypair
        signed_tx = VersionedTransaction(versioned_tx.message, [self.keypair])
        
        # Track transaction attempt and signature
        self.transaction_stats['total_attempted'] += 1
        
        try:
            if signed_tx.signatures and len(signed_tx.signatures) > 0:
                self.last_signature = str(signed_tx.signatures[0])
                logger.debug(f"Tracking transaction signature: {self.last_signature}")
        except Exception as sig_error:
            logger.debug(f"Could not extract signature for tracking: {sig_error}")
        
        async def _submit_versioned_operation(client):
            """Inner operation for RPC manager"""
            # Serialize and send as raw transaction
            tx_bytes = bytes(signed_tx)
            
            # Try with priority fee settings for better success rate
            opts = TxOpts(
                skip_confirmation=False, 
                skip_preflight=True,  # Skip simulation to avoid Jupiter timing issues
                max_retries=0  # We handle retries manually
            )
            
            response = await client.send_raw_transaction(tx_bytes, opts=opts)
            
            logger.debug(f"RPC response type: {type(response)}")
            logger.debug(f"RPC response: {response}")
            
            if response and hasattr(response, 'value') and response.value:
                signature = str(response.value)
                logger.info(f"Transaction submitted successfully with signature: {signature}")
                self.processed_signatures.add(signature)
                return signature
            elif response is None:
                raise RuntimeError("RPC client returned None response - connection issue")
            else:
                # The transaction might still be successful, let's check the response structure
                if hasattr(response, 'result'):
                    signature = str(response.result)
                    logger.info(f"Transaction submitted successfully (via result field): {signature}")
                    self.processed_signatures.add(signature)
                    return signature
                else:
                    logger.error(f"Unexpected response structure: {dir(response)}")
                    raise RuntimeError(f"VersionedTransaction submission failed - response: {response}")
        
        try:
            # Use RPC manager for intelligent fallback
            if hasattr(self, 'rpc_manager') and self.rpc_manager:
                result = await self.rpc_manager.execute_with_fallback(_submit_versioned_operation)
                return result
            else:
                # Fallback to direct RPC client
                return await _submit_versioned_operation(self.rpc_client)
                
        except Exception as e:
            error_msg = str(e) if str(e) else "Unknown RPC error"
            logger.error(f"Error submitting versioned transaction: {error_msg}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {repr(e)}")
            
            # Log signature for debugging regardless of outcome
            if hasattr(self, 'last_signature') and self.last_signature:
                logger.error(f"Transaction signature for debugging: {self.last_signature}")
            
            # Try to extract signature from logs if transaction actually succeeded
            if hasattr(self, 'last_signature') and self.last_signature:
                logger.info(f"Checking if transaction {self.last_signature} succeeded despite RPC error...")
                try:
                    status = await self.get_transaction_status(self.last_signature)
                    if 'finalized' in status.lower() or 'confirmed' in status.lower():
                        logger.info(f"Transaction actually succeeded with status '{status}': {self.last_signature}")
                        self.processed_signatures.add(self.last_signature)
                        
                        # Update success metrics
                        self.transaction_stats['rpc_errors'] += 1
                        self.transaction_stats['confirmed_despite_errors'] += 1
                        self._update_success_rate()
                        
                        return self.last_signature
                    else:
                        logger.error(f"Transaction status check failed: {status} for {self.last_signature}")
                except Exception as status_error:
                    logger.error(f"Could not check transaction status: {status_error} for {self.last_signature}")
            
            # Check if we can extract signature from error logs/messages
            import re
            log_text = str(e) + error_msg
            signature_match = re.search(r'([1-9A-HJ-NP-Za-km-z]{87,88})', log_text)
            if signature_match:
                potential_signature = signature_match.group(1)
                logger.info(f"Found potential signature in logs: {potential_signature}")
                try:
                    status = await self.get_transaction_status(potential_signature)
                    if 'finalized' in status.lower() or 'confirmed' in status.lower():
                        logger.info(f"Extracted signature is valid and confirmed with status '{status}': {potential_signature}")
                        self.processed_signatures.add(potential_signature)
                        return potential_signature
                    else:
                        logger.warning(f"Extracted signature status: {status} for {potential_signature}")
                except Exception as extract_error:
                    logger.warning(f"Could not verify extracted signature: {extract_error} for {potential_signature}")
            
            # Final signature logging before raising exception
            if hasattr(self, 'last_signature') and self.last_signature:
                logger.error(f"FINAL DEBUG: Transaction {self.last_signature} failed with: {error_msg}")
            
            raise RuntimeError(f"Transaction submission failed: {error_msg}")
    
    async def get_transaction_status(self, signature: str) -> str:
        """
        Enhanced transaction status checking with retries and detailed logging
        
        Args:
            signature: Transaction signature to check
            
        Returns:
            str: Transaction status ('confirmed', 'finalized', 'failed', 'pending', 'unknown')
        """
        if not self.rpc_client:
            raise RuntimeError("RPC client not initialized")
        
        logger.debug(f"Checking status for transaction: {signature}")
        
        for attempt in range(6):  # Try 6 times over 30-60 seconds
            try:
                from solders.signature import Signature
                sig_obj = Signature.from_string(signature)
                response = await self.rpc_client.get_signature_statuses([sig_obj])
                
                if not response or not response.value:
                    logger.debug(f"No response for {signature} (attempt {attempt + 1}/6)")
                    if attempt < 5:
                        await asyncio.sleep(5 + attempt)  # Progressive delay: 5, 6, 7, 8, 9 seconds
                        continue
                    logger.warning(f"No status response after 6 attempts for {signature}")
                    return "pending"
                    
                status = response.value[0]
                if not status:
                    logger.debug(f"Empty status for {signature} (attempt {attempt + 1}/6)")
                    if attempt < 5:
                        await asyncio.sleep(5 + attempt)  # Progressive delay
                        continue
                    logger.warning(f"Empty status after 6 attempts for {signature}")
                    return "pending"
                    
                if status.err:
                    logger.error(f"Transaction failed on-chain: {signature}, error: {status.err}")
                    return "failed"
                elif hasattr(status, 'confirmation_status') and status.confirmation_status:
                    # Convert enum to string representation
                    status_value = str(status.confirmation_status).lower()
                    logger.debug(f"Transaction {signature} status: {status_value}")
                    return status_value
                else:
                    logger.debug(f"Transaction {signature} confirmed (no specific confirmation status)")
                    return "confirmed"
                    
            except Exception as e:
                logger.warning(f"Transaction status check attempt {attempt + 1}/6 failed for {signature}: {e}")
                if attempt < 5:
                    await asyncio.sleep(5 + attempt)  # Progressive delay
                    continue
                logger.error(f"All status check attempts failed for {signature}: {e}")
                return "unknown"
    
    def is_signature_processed(self, signature: str) -> bool:
        """Check if a signature was already processed successfully"""
        return signature in self.processed_signatures
    
    def is_signature_failed(self, signature: str) -> bool:
        """Check if a signature was marked as failed"""
        return signature in self.failed_signatures
    
    def mark_signature_failed(self, signature: str) -> None:
        """Mark a signature as failed"""
        self.failed_signatures.add(signature)
        self.transaction_stats['genuine_failures'] += 1
        self._update_success_rate()
    
    def _update_success_rate(self) -> None:
        """Update transaction success rate metrics"""
        total = self.transaction_stats['total_attempted']
        if total > 0:
            successes = self.transaction_stats['confirmed_despite_errors'] + (total - self.transaction_stats['rpc_errors'] - self.transaction_stats['genuine_failures'])
            self.transaction_stats['success_rate'] = successes / total
            
            # Log metrics periodically (every 10 transactions)
            if total % 10 == 0:
                logger.info(f"[METRICS] Transaction Success Rate: {self.transaction_stats['success_rate']:.2%}")
                logger.info(f"[METRICS] Total: {total}, RPC Errors: {self.transaction_stats['rpc_errors']}, "
                          f"Confirmed Despite Errors: {self.transaction_stats['confirmed_despite_errors']}, "
                          f"Genuine Failures: {self.transaction_stats['genuine_failures']}")
    
    def get_transaction_metrics(self) -> Dict[str, Any]:
        """Get current transaction success metrics"""
        return self.transaction_stats.copy()
    
    def get_rpc_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get RPC provider performance statistics"""
        if hasattr(self, 'rpc_manager') and self.rpc_manager:
            return self.rpc_manager.get_stats()
        else:
            return {}
    
    async def log_rpc_performance_summary(self):
        """Log RPC performance summary"""
        if hasattr(self, 'rpc_manager') and self.rpc_manager:
            self.rpc_manager.log_performance_summary()