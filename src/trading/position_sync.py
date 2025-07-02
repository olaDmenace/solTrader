"""
Position Synchronization Module
Synchronizes trading positions with blockchain state
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of position synchronization"""
    success: bool
    updated_positions: List[str] = field(default_factory=list)
    removed_positions: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    sync_time: datetime = field(default_factory=datetime.now)


class PositionSynchronizer:
    """Synchronize trading positions with blockchain state"""
    
    def __init__(self, wallet, jupiter_client, position_manager):
        """
        Initialize position synchronizer
        
        Args:
            wallet: PhantomWallet instance
            jupiter_client: JupiterClient instance  
            position_manager: PositionManager instance
        """
        self.wallet = wallet
        self.jupiter = jupiter_client
        self.position_manager = position_manager
        
        self.last_sync: Optional[datetime] = None
        self.sync_interval = 300  # 5 minutes default
        self.tolerance = 0.001  # 0.1% tolerance for balance differences
        
        # Monitoring
        self._sync_task: Optional[asyncio.Task] = None
        self._monitoring = False
    
    async def start_monitoring(self, interval: int = 300) -> None:
        """Start automatic position synchronization"""
        if self._monitoring:
            return
            
        self.sync_interval = interval
        self._monitoring = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        logger.info(f"Position synchronization monitoring started (interval: {interval}s)")
    
    async def stop_monitoring(self) -> None:
        """Stop automatic synchronization"""
        self._monitoring = False
        
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Position synchronization monitoring stopped")
    
    async def sync_positions(self, force: bool = False) -> SyncResult:
        """
        Synchronize positions with blockchain state
        
        Args:
            force: Force sync even if recently synced
            
        Returns:
            SyncResult with sync details
        """
        try:
            # Check if sync needed
            if not force and self._is_recently_synced():
                logger.debug("Skipping sync - recently synchronized")
                return SyncResult(success=True)
                
            if not self.wallet.live_mode:
                logger.warning("Cannot sync positions - wallet not in live mode")
                return SyncResult(success=False, errors=["Wallet not in live mode"])
                
            logger.info("Starting position synchronization...")
            
            # Get blockchain token balances
            blockchain_balances = await self._get_blockchain_balances()
            if not blockchain_balances:
                return SyncResult(success=False, errors=["Failed to get blockchain balances"])
                
            # Get managed positions
            managed_positions = self.position_manager.positions.copy()
            
            # Sync positions
            result = await self._reconcile_positions(blockchain_balances, managed_positions)
            
            # Update last sync time
            self.last_sync = datetime.now()
            
            logger.info(
                f"Position sync completed: {len(result.updated_positions)} updated, "
                f"{len(result.removed_positions)} removed"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during position synchronization: {str(e)}")
            return SyncResult(success=False, errors=[str(e)])
    
    async def _sync_loop(self) -> None:
        """Automatic synchronization loop"""
        while self._monitoring:
            try:
                await self.sync_positions()
                await asyncio.sleep(self.sync_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _get_blockchain_balances(self) -> Optional[Dict[str, float]]:
        """Get actual token balances from blockchain"""
        try:
            token_accounts = await self.wallet.get_token_accounts()
            balances = {}
            
            for account in token_accounts:
                mint = account.get('mint')
                balance = account.get('balance', 0.0)
                
                if mint and balance > 0:
                    balances[mint] = float(balance)
                    
            logger.debug(f"Retrieved {len(balances)} token balances from blockchain")
            return balances
            
        except Exception as e:
            logger.error(f"Error getting blockchain balances: {str(e)}")
            return None
    
    async def _reconcile_positions(
        self, 
        blockchain_balances: Dict[str, float],
        managed_positions: Dict[str, Any]
    ) -> SyncResult:
        """Reconcile managed positions with blockchain reality"""
        result = SyncResult(success=True)
        
        try:
            # Check existing positions
            for token_address, position in managed_positions.items():
                blockchain_balance = blockchain_balances.get(token_address, 0.0)
                managed_balance = position.size
                
                # Calculate difference
                diff = abs(blockchain_balance - managed_balance)
                relative_diff = diff / max(managed_balance, 0.001)  # Avoid division by zero
                
                if relative_diff > self.tolerance:
                    if blockchain_balance == 0.0:
                        # Position closed on blockchain but still managed
                        logger.warning(
                            f"Position {token_address} closed on blockchain, removing from management"
                        )
                        await self._remove_managed_position(token_address)
                        result.removed_positions.append(token_address)
                        
                    else:
                        # Balance mismatch - update managed position
                        logger.info(
                            f"Updating position {token_address}: {managed_balance} -> {blockchain_balance}"
                        )
                        await self._update_position_size(token_address, blockchain_balance)
                        result.updated_positions.append(token_address)
            
            # Check for new positions on blockchain not in management
            for token_address, balance in blockchain_balances.items():
                if token_address not in managed_positions and balance > self.tolerance:
                    # Found unmanaged position on blockchain
                    logger.info(f"Found unmanaged position {token_address}: {balance}")
                    
                    # Get current price and create position
                    current_price = await self._get_current_price(token_address)
                    if current_price:
                        await self._create_managed_position(
                            token_address, balance, current_price
                        )
                        result.updated_positions.append(token_address)
                    else:
                        result.errors.append(f"Failed to get price for {token_address}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error reconciling positions: {str(e)}")
            result.success = False
            result.errors.append(str(e))
            return result
    
    async def _remove_managed_position(self, token_address: str) -> None:
        """Remove position from management"""
        try:
            if token_address in self.position_manager.positions:
                position = self.position_manager.positions[token_address]
                
                # Close position with sync reason
                await self.position_manager.close_position(
                    token_address, 
                    reason="position_sync_closed"
                )
                
                logger.info(f"Removed managed position: {token_address}")
                
        except Exception as e:
            logger.error(f"Error removing managed position {token_address}: {str(e)}")
    
    async def _update_position_size(self, token_address: str, new_size: float) -> None:
        """Update position size in management"""
        try:
            if token_address in self.position_manager.positions:
                position = self.position_manager.positions[token_address]
                old_size = position.size
                
                # Update position size
                position.size = new_size
                
                # Update position value and PnL
                current_price = await self._get_current_price(token_address)
                if current_price:
                    position.update_price(current_price, 0)  # Volume not critical for sync
                    
                logger.info(
                    f"Updated position size {token_address}: {old_size} -> {new_size}"
                )
                
        except Exception as e:
            logger.error(f"Error updating position size {token_address}: {str(e)}")
    
    async def _create_managed_position(
        self, 
        token_address: str, 
        size: float, 
        current_price: float
    ) -> None:
        """Create managed position for untracked blockchain position"""
        try:
            # Create position with current price as entry (best estimate)
            await self.position_manager.open_position(
                token_address=token_address,
                size=size,
                entry_price=current_price,  # Use current price as entry estimate
                stop_loss=current_price * 0.9,  # 10% stop loss
                take_profit=current_price * 1.2,  # 20% take profit
                source="position_sync"
            )
            
            logger.info(
                f"Created managed position for {token_address}: "
                f"size={size}, price={current_price}"
            )
            
        except Exception as e:
            logger.error(f"Error creating managed position {token_address}: {str(e)}")
    
    async def _get_current_price(self, token_address: str) -> Optional[float]:
        """Get current price for token"""
        try:
            price_data = await self.jupiter.get_price(
                input_mint=token_address,
                output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                amount=1000000  # 1 USDC worth
            )
            
            if price_data and 'outAmount' in price_data:
                out_amount = float(price_data['outAmount'])
                return out_amount / 1e9 if out_amount > 0 else None
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price for {token_address}: {str(e)}")
            return None
    
    def _is_recently_synced(self, max_age_minutes: int = 5) -> bool:
        """Check if positions were recently synchronized"""
        if not self.last_sync:
            return False
            
        age = datetime.now() - self.last_sync
        return age < timedelta(minutes=max_age_minutes)
    
    async def force_sync_position(self, token_address: str) -> bool:
        """Force synchronization of specific position"""
        try:
            if not self.wallet.live_mode:
                logger.error("Cannot sync position - wallet not in live mode")
                return False
                
            # Get blockchain balance for this token
            token_accounts = await self.wallet.get_token_accounts()
            blockchain_balance = 0.0
            
            for account in token_accounts:
                if account.get('mint') == token_address:
                    blockchain_balance = float(account.get('balance', 0.0))
                    break
            
            # Get managed position
            managed_position = self.position_manager.positions.get(token_address)
            
            if not managed_position and blockchain_balance > self.tolerance:
                # Create missing managed position
                current_price = await self._get_current_price(token_address)
                if current_price:
                    await self._create_managed_position(
                        token_address, blockchain_balance, current_price
                    )
                    return True
                    
            elif managed_position:
                if blockchain_balance == 0.0:
                    # Remove managed position
                    await self._remove_managed_position(token_address)
                    return True
                else:
                    # Update position size
                    await self._update_position_size(token_address, blockchain_balance)
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error force syncing position {token_address}: {str(e)}")
            return False
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status information"""
        return {
            'monitoring': self._monitoring,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'sync_interval': self.sync_interval,
            'tolerance': self.tolerance,
            'wallet_live_mode': self.wallet.live_mode,
            'managed_positions': len(self.position_manager.positions)
        }
    
    async def validate_all_positions(self) -> Dict[str, Any]:
        """Validate all positions against blockchain state"""
        try:
            if not self.wallet.live_mode:
                return {'error': 'Wallet not in live mode'}
                
            blockchain_balances = await self._get_blockchain_balances()
            if not blockchain_balances:
                return {'error': 'Failed to get blockchain balances'}
                
            managed_positions = self.position_manager.positions
            validation_results = {
                'total_managed': len(managed_positions),
                'total_blockchain': len(blockchain_balances),
                'matches': 0,
                'discrepancies': [],
                'unmanaged_tokens': [],
                'orphaned_positions': []
            }
            
            # Check managed positions
            for token_address, position in managed_positions.items():
                blockchain_balance = blockchain_balances.get(token_address, 0.0)
                managed_balance = position.size
                
                diff = abs(blockchain_balance - managed_balance)
                relative_diff = diff / max(managed_balance, 0.001)
                
                if relative_diff <= self.tolerance:
                    validation_results['matches'] += 1
                else:
                    validation_results['discrepancies'].append({
                        'token': token_address,
                        'managed': managed_balance,
                        'blockchain': blockchain_balance,
                        'difference': diff,
                        'relative_diff': relative_diff
                    })
            
            # Check for unmanaged positions
            for token_address, balance in blockchain_balances.items():
                if token_address not in managed_positions and balance > self.tolerance:
                    validation_results['unmanaged_tokens'].append({
                        'token': token_address,
                        'balance': balance
                    })
            
            # Check for orphaned positions
            for token_address in managed_positions:
                if token_address not in blockchain_balances:
                    validation_results['orphaned_positions'].append(token_address)
                    
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating positions: {str(e)}")
            return {'error': str(e)}