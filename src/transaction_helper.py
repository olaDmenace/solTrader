import logging
from typing import Dict, Any, Optional
from solana.transaction import Transaction
import base58

logger = logging.getLogger(__name__)

class TransactionHelper:
    """Helper class for handling Solana transactions"""
    
    @staticmethod
    def create_transfer_instruction(
        from_pubkey: str,
        to_pubkey: str,
        amount: int,
        program_id: str
    ) -> Dict:
        """Create a transfer instruction"""
        return {
            'programId': program_id,
            'keys': [
                {'pubkey': from_pubkey, 'isSigner': True, 'isWritable': True},
                {'pubkey': to_pubkey, 'isSigner': False, 'isWritable': True},
            ],
            'data': base58.b58encode(bytes([2, *amount.to_bytes(8, 'little')])).decode('ascii')
        }
        
    @staticmethod
    def estimate_transaction_size(instruction_count: int) -> int:
        """Estimate transaction size for fee calculation"""
        # Basic transaction structure
        size = 1 + 1 + 1 + 1  # version + header + num_required_signatures + num_readonly_signed
        
        # Add instruction size
        size += instruction_count * 100  # Approximate size per instruction
        
        return size
        
    @staticmethod
    def calculate_fee(instruction_count: int) -> int:
        """Calculate transaction fee"""
        size = TransactionHelper.estimate_transaction_size(instruction_count)
        return size * 5000  # lamports per byte