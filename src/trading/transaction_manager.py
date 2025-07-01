"""
transaction_manager.py - Comprehensive transaction management for Solana trading
"""
import logging
import asyncio
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Callable
import backoff
from solana.transaction import Transaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    """Transaction status enumeration"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TransactionRecord:
    """Record of a transaction and its status"""
    signature: str
    transaction: Transaction
    submitted_at: datetime
    status: TransactionStatus = TransactionStatus.PENDING
    confirmation_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    priority_fee: Optional[int] = None


class TransactionManager:
    """Comprehensive transaction manager for Solana operations"""
    
    def __init__(
        self,
        rpc_client: AsyncClient,
        max_retries: int = 3,
        confirmation_timeout: int = 60,
        retry_delay: float = 2.0
    ):
        """
        Initialize transaction manager
        
        Args:
            rpc_client: Solana RPC client
            max_retries: Maximum number of retry attempts
            confirmation_timeout: Timeout for transaction confirmation (seconds)
            retry_delay: Base delay between retries (seconds)
        """
        self.rpc_client = rpc_client
        self.max_retries = max_retries
        self.confirmation_timeout = confirmation_timeout
        self.retry_delay = retry_delay
        
        # Track active transactions
        self.active_transactions: Dict[str, TransactionRecord] = {}
        self.completed_transactions: List[TransactionRecord] = []
        
        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitoring = False
    
    async def start_monitoring(self) -> None:
        """Start transaction monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_transactions())
        logger.info("Transaction monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop transaction monitoring"""
        self._monitoring = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Transaction monitoring stopped")
    
    async def submit_transaction(
        self,
        transaction: Transaction,
        priority_fee: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> Optional[str]:
        """
        Submit transaction with automatic retry and monitoring
        
        Args:
            transaction: Transaction to submit
            priority_fee: Priority fee in microlamports
            max_retries: Override default max retries
            
        Returns:
            Optional[str]: Transaction signature if successful
        """
        max_retries = max_retries or self.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Submitting transaction (attempt {attempt + 1}/{max_retries + 1})")
                
                # Submit transaction
                response = await self.rpc_client.send_transaction(
                    transaction,
                    opts={"skip_confirmation": False, "skip_preflight": False}
                )
                
                if response.value:
                    signature = str(response.value)
                    
                    # Create transaction record
                    record = TransactionRecord(
                        signature=signature,
                        transaction=transaction,
                        submitted_at=datetime.now(),
                        priority_fee=priority_fee,
                        retry_count=attempt
                    )
                    
                    # Add to tracking
                    self.active_transactions[signature] = record
                    
                    logger.info(f"Transaction submitted successfully: {signature}")
                    return signature
                    
            except Exception as e:
                logger.warning(f"Transaction submission attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < max_retries:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Transaction submission failed after {max_retries + 1} attempts")
                    
        return None
    
    async def wait_for_confirmation(
        self,
        signature: str,
        timeout: Optional[int] = None,
        commitment: Commitment = Commitment("confirmed")
    ) -> TransactionStatus:
        """
        Wait for transaction confirmation
        
        Args:
            signature: Transaction signature
            timeout: Override default timeout
            commitment: Commitment level to wait for
            
        Returns:
            TransactionStatus: Final transaction status
        """
        timeout = timeout or self.confirmation_timeout
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                status = await self.get_transaction_status(signature)
                
                if status in [TransactionStatus.CONFIRMED, TransactionStatus.FINALIZED, TransactionStatus.FAILED]:
                    return status
                    
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.warning(f"Error checking transaction status: {str(e)}")
                await asyncio.sleep(2)
        
        # Timeout reached
        if signature in self.active_transactions:
            self.active_transactions[signature].status = TransactionStatus.TIMEOUT
            
        logger.warning(f"Transaction confirmation timeout: {signature}")
        return TransactionStatus.TIMEOUT
    
    async def get_transaction_status(self, signature: str) -> TransactionStatus:
        """
        Get current transaction status
        
        Args:
            signature: Transaction signature
            
        Returns:
            TransactionStatus: Current status
        """
        try:
            response = await self.rpc_client.get_signature_status(signature)
            
            if not response.value:
                return TransactionStatus.PENDING
                
            status_info = response.value[0]
            if not status_info:
                return TransactionStatus.PENDING
                
            if status_info.err:
                # Update record if we're tracking it
                if signature in self.active_transactions:
                    record = self.active_transactions[signature]
                    record.status = TransactionStatus.FAILED
                    record.error_message = str(status_info.err)
                    
                return TransactionStatus.FAILED
                
            if status_info.confirmation_status:
                if status_info.confirmation_status.value == "finalized":
                    return TransactionStatus.FINALIZED
                elif status_info.confirmation_status.value == "confirmed":
                    return TransactionStatus.CONFIRMED
                    
            return TransactionStatus.PENDING
            
        except Exception as e:
            logger.error(f"Error getting transaction status: {str(e)}")
            return TransactionStatus.PENDING
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30
    )
    async def retry_failed_transaction(self, signature: str) -> Optional[str]:
        """
        Retry a failed transaction
        
        Args:
            signature: Original transaction signature
            
        Returns:
            Optional[str]: New transaction signature if successful
        """
        if signature not in self.active_transactions:
            logger.error(f"Transaction not found for retry: {signature}")
            return None
            
        record = self.active_transactions[signature]
        
        if record.retry_count >= self.max_retries:
            logger.error(f"Max retries exceeded for transaction: {signature}")
            return None
            
        logger.info(f"Retrying failed transaction: {signature}")
        
        # Remove from active transactions
        del self.active_transactions[signature]
        
        # Submit as new transaction
        new_signature = await self.submit_transaction(
            record.transaction,
            record.priority_fee
        )
        
        return new_signature
    
    async def _monitor_transactions(self) -> None:
        """Monitor active transactions for status updates"""
        while self._monitoring:
            try:
                # Check all active transactions
                for signature in list(self.active_transactions.keys()):
                    record = self.active_transactions[signature]
                    
                    # Check if transaction has timed out
                    age = (datetime.now() - record.submitted_at).total_seconds()
                    if age > self.confirmation_timeout:
                        record.status = TransactionStatus.TIMEOUT
                        self._move_to_completed(signature)
                        continue
                    
                    # Check transaction status
                    status = await self.get_transaction_status(signature)
                    record.status = status
                    
                    # Move completed transactions
                    if status in [TransactionStatus.CONFIRMED, TransactionStatus.FINALIZED, TransactionStatus.FAILED, TransactionStatus.TIMEOUT]:
                        if status in [TransactionStatus.CONFIRMED, TransactionStatus.FINALIZED]:
                            record.confirmation_time = datetime.now()
                            
                        self._move_to_completed(signature)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in transaction monitoring: {str(e)}")
                await asyncio.sleep(10)
    
    def _move_to_completed(self, signature: str) -> None:
        """Move transaction from active to completed"""
        if signature in self.active_transactions:
            record = self.active_transactions[signature]
            self.completed_transactions.append(record)
            del self.active_transactions[signature]
            
            # Keep only recent completed transactions (last 100)
            if len(self.completed_transactions) > 100:
                self.completed_transactions = self.completed_transactions[-100:]
    
    def get_active_count(self) -> int:
        """Get number of active transactions"""
        return len(self.active_transactions)
    
    def get_pending_count(self) -> int:
        """Get number of pending transactions"""
        return len([r for r in self.active_transactions.values() if r.status == TransactionStatus.PENDING])
    
    def get_recent_stats(self, hours: int = 24) -> Dict[str, int]:
        """
        Get transaction statistics for recent period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict with transaction statistics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_transactions = [
            r for r in self.completed_transactions 
            if r.submitted_at >= cutoff
        ]
        
        stats = {
            "total": len(recent_transactions),
            "confirmed": len([r for r in recent_transactions if r.status == TransactionStatus.CONFIRMED]),
            "finalized": len([r for r in recent_transactions if r.status == TransactionStatus.FINALIZED]),
            "failed": len([r for r in recent_transactions if r.status == TransactionStatus.FAILED]),
            "timeout": len([r for r in recent_transactions if r.status == TransactionStatus.TIMEOUT]),
        }
        
        # Calculate success rate
        successful = stats["confirmed"] + stats["finalized"]
        stats["success_rate"] = (successful / stats["total"] * 100) if stats["total"] > 0 else 0
        
        return stats
    
    async def cleanup_old_records(self, days: int = 7) -> None:
        """
        Clean up old transaction records
        
        Args:
            days: Number of days to keep records
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        # Clean completed transactions
        self.completed_transactions = [
            r for r in self.completed_transactions 
            if r.submitted_at >= cutoff
        ]
        
        logger.info(f"Cleaned up transaction records older than {days} days")