# src/trading/gas_optimization.py

"""
Gas optimization system for efficient transaction execution
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class GasConfig:
    """Gas optimization configuration"""
    max_base_fee: int = 500  # GWEI
    max_priority_fee: int = 10  # GWEI
    min_priority_fee: int = 1  # GWEI
    base_fee_multiplier: float = 1.2
    dynamic_adjustment: bool = True
    bundle_enabled: bool = True
    flash_enabled: bool = True

@dataclass
class GasHistory:
    """Historical gas data"""
    base_fees: List[int]
    priority_fees: List[int]
    timestamps: List[datetime]
    success_rates: List[float]

class GasOptimizer:
    def __init__(self, settings: Any, jupiter_client: Any):
        """Initialize gas optimizer with settings and Jupiter client"""
        self.settings = settings
        self.jupiter = jupiter_client
        self.config = GasConfig()
        self.gas_history = GasHistory([], [], [], [])
        self._monitor_task: Optional[asyncio.Task] = None
        self.pending_bundles: List[Dict[str, Any]] = []
        self.last_update = datetime.now()

    async def optimize_gas_params(
        self,
        tx_type: str,
        estimated_gas: int,
        urgency: str = 'normal'
    ) -> Dict[str, Any]:
        """Calculate optimal gas parameters for transaction"""
        try:
            # Get current market conditions
            base_fee = await self._get_current_base_fee()
            priority_fees = await self._get_priority_fee_range()
            congestion = await self._calculate_network_congestion()

            # Adjust for transaction urgency
            urgency_multipliers = {
                'low': 0.8,
                'normal': 1.0,
                'high': 1.2,
                'critical': 1.5
            }
            
            multiplier = urgency_multipliers.get(urgency, 1.0)

            # Calculate optimal parameters
            optimal_params = await self._calculate_optimal_params(
                base_fee=base_fee,
                priority_fees=priority_fees,
                congestion=congestion,
                multiplier=multiplier,
                tx_type=tx_type
            )

            # Add execution suggestions
            suggestions = self._get_execution_suggestions(
                optimal_params,
                tx_type,
                congestion
            )

            return {
                'gas_params': optimal_params,
                'suggestions': suggestions,
                'estimated_cost': self._estimate_tx_cost(
                    optimal_params,
                    estimated_gas
                ),
                'market_conditions': {
                    'congestion': congestion,
                    'base_fee': base_fee,
                    'priority_fee_range': priority_fees
                }
            }

        except Exception as e:
            logger.error(f"Error optimizing gas parameters: {str(e)}")
            return self._get_fallback_params(estimated_gas)

    async def _calculate_optimal_params(
        self,
        base_fee: int,
        priority_fees: List[int],
        congestion: float,
        multiplier: float,
        tx_type: str
    ) -> Dict[str, Any]:
        """Calculate optimal gas parameters"""
        try:
            # Adjust base fee based on congestion
            adjusted_base_fee = int(base_fee * (1 + congestion * 0.1))
            max_base_fee = int(adjusted_base_fee * self.config.base_fee_multiplier)

            # Calculate optimal priority fee
            optimal_priority_fee = self._calculate_priority_fee(
                priority_fees,
                congestion,
                multiplier
            )

            # Special handling for different transaction types
            if tx_type == 'bundle':
                return self._optimize_bundle_params(
                    adjusted_base_fee,
                    optimal_priority_fee
                )
            elif tx_type == 'flash':
                return self._optimize_flash_params(
                    adjusted_base_fee,
                    optimal_priority_fee
                )
            else:
                return {
                    'maxFeePerGas': max_base_fee,
                    'maxPriorityFeePerGas': optimal_priority_fee,
                    'type': '0x2'  # EIP-1559
                }

        except Exception as e:
            logger.error(f"Error calculating optimal parameters: {str(e)}")
            return self._get_fallback_params(0)

    def _calculate_priority_fee(
        self,
        priority_fees: List[int],
        congestion: float,
        multiplier: float
    ) -> int:
        """Calculate optimal priority fee"""
        if not priority_fees:
            return self.config.min_priority_fee

        recent_fees = sorted(priority_fees)
        median_fee = recent_fees[len(recent_fees) // 2]
        
        # Adjust for congestion and urgency
        adjusted_fee = int(median_fee * (1 + congestion * 0.2) * multiplier)
        
        # Apply bounds
        return min(
            max(adjusted_fee, self.config.min_priority_fee),
            self.config.max_priority_fee
        )

    async def _optimize_bundle_params(
        self,
        base_fee: int,
        priority_fee: int
    ) -> Dict[str, Any]:
        """Optimize parameters for bundle transactions"""
        return {
            'maxFeePerGas': base_fee,
            'maxPriorityFeePerGas': priority_fee,
            'type': '0x2',
            'bundle': {
                'enabled': True,
                'revertingTxHashes': [],
                'replacementUuid': None
            }
        }

    async def _optimize_flash_params(
        self,
        base_fee: int,
        priority_fee: int
    ) -> Dict[str, Any]:
        """Optimize parameters for flash transactions"""
        return {
            'maxFeePerGas': base_fee,
            'maxPriorityFeePerGas': priority_fee,
            'type': '0x2',
            'flash': {
                'enabled': True,
                'prefund': False,
                'maxGasPrice': base_fee + priority_fee
            }
        }

    def _estimate_tx_cost(
        self,
        params: Dict[str, Any],
        estimated_gas: int
    ) -> float:
        """Estimate transaction cost in native token"""
        max_fee = params.get('maxFeePerGas', 0)
        priority_fee = params.get('maxPriorityFeePerGas', 0)
        
        # Calculate worst-case cost
        worst_case_cost = (max_fee * estimated_gas) / 1e9
        
        # Calculate likely cost
        likely_cost = ((max_fee - priority_fee) * estimated_gas) / 1e9
        
        return {
            'worst_case': worst_case_cost,
            'likely': likely_cost,
            'estimated_gas': estimated_gas
        }

    def _get_execution_suggestions(
        self,
        params: Dict[str, Any],
        tx_type: str,
        congestion: float
    ) -> List[str]:
        """Get execution optimization suggestions"""
        suggestions = []
        
        if congestion > 0.8:
            suggestions.append("High network congestion - consider delayed execution")
            
        if tx_type == 'standard' and congestion > 0.5:
            suggestions.append("Consider using transaction bundles for better execution")
            
        if params['maxFeePerGas'] > self.config.max_base_fee:
            suggestions.append("Gas fees unusually high - may want to wait")
            
        return suggestions

    async def _get_current_base_fee(self) -> int:
        """Get current base fee from network"""
        try:
            block = await self.rpc._make_request(
                "eth_getBlockByNumber",
                ["latest", False]
            )
            return int(block['baseFeePerGas'], 16)
        except Exception as e:
            logger.error(f"Error getting base fee: {str(e)}")
            return 50  # Fallback value in GWEI

    async def _get_priority_fee_range(self) -> List[int]:
        """Get recent priority fee range"""
        try:
            fee_history = await self.rpc._make_request(
                "eth_feeHistory",
                ["0x4", "latest", [25, 50, 75]]
            )
            
            if not fee_history or 'reward' not in fee_history:
                return []
                
            fees = []
            for reward_list in fee_history['reward']:
                fees.extend([int(r, 16) for r in reward_list])
            
            return sorted(fees)
            
        except Exception as e:
            logger.error(f"Error getting priority fees: {str(e)}")
            return []

    async def _calculate_network_congestion(self) -> float:
        """Calculate current network congestion level"""
        try:
            block = await self.rpc._make_request(
                "eth_getBlockByNumber",
                ["latest", False]
            )
            
            gas_used = int(block['gasUsed'], 16)
            gas_limit = int(block['gasLimit'], 16)
            
            return min(gas_used / gas_limit, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating congestion: {str(e)}")
            return 0.5  # Moderate congestion as fallback

    def _get_fallback_params(self, estimated_gas: int) -> Dict[str, Any]:
        """Get fallback gas parameters"""
        return {
            'gas_params': {
                'maxFeePerGas': self.config.max_base_fee,
                'maxPriorityFeePerGas': self.config.min_priority_fee,
                'type': '0x2'
            },
            'suggestions': ["Using fallback gas parameters due to error"],
            'estimated_cost': {
                'worst_case': (self.config.max_base_fee * estimated_gas) / 1e9,
                'likely': (self.config.max_base_fee * estimated_gas) / 1e9,
                'estimated_gas': estimated_gas
            }
        }

    async def start_monitoring(self) -> None:
        """Start gas monitoring"""
        if self._monitor_task and not self._monitor_task.done():
            return

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Gas monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop gas monitoring"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("Gas monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while True:
            try:
                # Get current gas data
                base_fee = await self._get_current_base_fee()
                priority_fees = await self._get_priority_fee_range()
                
                # Update history
                self.gas_history.base_fees.append(base_fee)
                self.gas_history.priority_fees.extend(priority_fees)
                self.gas_history.timestamps.append(datetime.now())
                
                # Trim old data (keep last hour)
                cutoff = datetime.now() - timedelta(hours=1)
                cutoff_index = 0
                
                for i, ts in enumerate(self.gas_history.timestamps):
                    if ts > cutoff:
                        cutoff_index = i
                        break
                        
                self.gas_history.base_fees = self.gas_history.base_fees[cutoff_index:]
                self.gas_history.priority_fees = self.gas_history.priority_fees[cutoff_index:]
                self.gas_history.timestamps = self.gas_history.timestamps[cutoff_index:]
                
                await asyncio.sleep(12)  # Update every block
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                await asyncio.sleep(5)