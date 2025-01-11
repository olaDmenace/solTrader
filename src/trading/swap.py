import logging
from typing import Optional, Dict, Any
from decimal import Decimal
import base58
import asyncio
from solana.transaction import Transaction
# from solders.hash import Hash
from solders.hash import Hash

logger = logging.getLogger(__name__)

class SwapExecutor:
    def __init__(self, jupiter_client, wallet):
        self.jupiter = jupiter_client
        self.wallet = wallet

    async def execute_swap(
        self,
        input_token: str,
        output_token: str,
        amount: float,
        slippage: float = 0.01
    ) -> Optional[str]:
        try:
            quote = await self._get_quote(input_token, output_token, amount, slippage)
            if not quote:
                return None

            swap_tx = await self._get_swap_tx(quote)
            if not swap_tx:
                return None

            signature = await self.wallet.sign_and_send_transaction(swap_tx)
            if not signature:
                return None

            confirmed = await self._confirm_transaction(signature)
            return signature if confirmed else None

        except Exception as e:
            logger.error(f"Swap execution failed: {str(e)}")
            return None

    async def _get_quote(
        self,
        input_token: str,
        output_token: str,
        amount: float,
        slippage: float
    ) -> Optional[Dict[str, Any]]:
        try:
            return await self.jupiter.get_quote(
                input_mint=input_token,
                output_mint=output_token,
                amount=str(int(amount * 1e9)),
                slippageBps=int(slippage * 10000)
            )
        except Exception as e:
            logger.error(f"Failed to get quote: {str(e)}")
            return None

    async def _get_swap_tx(self, quote: Dict[str, Any]) -> Optional[Transaction]:
        try:
            swap_ix = await self.jupiter.get_swap_ix(quote)
            if not swap_ix:
                return None

            recent_blockhash = await self.wallet.get_recent_blockhash()
            tx = Transaction()
            tx.recent_blockhash = Hash.from_string(recent_blockhash)
            tx.add(swap_ix)
            
            return tx
        except Exception as e:
            logger.error(f"Failed to build swap transaction: {str(e)}")
            return None

    async def _confirm_transaction(self, signature: str, max_retries: int = 30) -> bool:
        for _ in range(max_retries):
            try:
                status = await self.wallet.get_transaction_status(signature)
                if status == 'confirmed':
                    return True
                elif status == 'failed':
                    return False
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error checking transaction status: {str(e)}")
                return False
        return False