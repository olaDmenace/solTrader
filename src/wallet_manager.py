import logging
from typing import Dict, Any, Optional, List
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TokenAccountOpts
from solders.pubkey import Pubkey as PublicKey  # type: ignore
from solders.signature import Signature  # type: ignore
from solders.hash import Hash  # type: ignore
import base58
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

SOL_TO_LAMPORTS = 1000000000

class WalletManager:
    def __init__(self, rpc_client: AsyncClient):
        self.client = rpc_client
        self.connected: bool = False
        self.wallet_address: Optional[str] = None
        self.balance: float = 0
        self.token_accounts: Dict[str, Dict[str, Any]] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self.last_update: Optional[datetime] = None
        self.transaction_history: List[Dict[str, Any]] = []

    async def get_balance(self, address: Optional[str] = None) -> Optional[float]:
        try:
            addr = address or self.wallet_address
            if not addr:
                return None

            # Handle the str | None type issue
            if not isinstance(addr, str):
                return None
                
            pubkey = PublicKey.from_string(addr)
            response = await self.client.get_balance(pubkey, commitment=Confirmed)

            if response and hasattr(response, 'value'):
                balance = float(response.value) / SOL_TO_LAMPORTS
                if not address:
                    self.balance = balance
                    self.last_update = datetime.now()
                return balance
            return None
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            return None

    async def monitor_transactions(self, callback: Optional[callable] = None) -> None:
        if not self.wallet_address or not self.connected:
            return

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

        async def monitor_loop() -> None:
            last_signature = None
            while self.connected:
                try:
                    if not isinstance(self.wallet_address, str):
                        continue
                        
                    address = PublicKey.from_string(self.wallet_address)
                    response = await self.client.get_signatures_for_address(
                        address,
                        commitment=Confirmed
                    )
                    
                    if response and response.value:
                        newest_sig = response.value[0].signature
                        if newest_sig != last_signature:
                            sig = Signature.from_string(str(newest_sig))
                            tx = await self.client.get_transaction(sig, commitment=Confirmed)
                            
                            if tx and tx.value:
                                status = 'failed'
                                if hasattr(tx.value, 'transaction') and hasattr(tx.value.transaction, 'meta'):
                                    status = 'confirmed' if not tx.value.transaction.meta.err else 'failed'
                                
                                tx_info = {
                                    'signature': str(newest_sig),
                                    'timestamp': datetime.fromtimestamp(tx.value.block_time or 0),
                                    'status': status,
                                }
                                
                                if callback:
                                    await callback(tx_info)
                                    
                                self.transaction_history.append(tx_info)
                                last_signature = newest_sig
                                
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in monitor loop: {str(e)}")
                    await asyncio.sleep(5)

        self._monitor_task = asyncio.create_task(monitor_loop())

    def _validate_wallet_address(self, address: str) -> bool:
        try:
            PublicKey.from_string(address)
            return True
        except:
            return False
            
    async def disconnect(self) -> None:
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        self.connected = False
        self.wallet_address = None
        self.balance = 0
        self.token_accounts.clear()
        self.transaction_history.clear()
        self.last_update = None
