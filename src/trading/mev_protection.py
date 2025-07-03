"""
Enhanced MEV protection system with comprehensive protection against sandwich attacks,
frontrunning, and other MEV-related risks.
"""

import logging
from typing import Dict, Optional, Any, List, Tuple, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
import random
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class MEVAnalysis:
    """Analysis results for MEV risk assessment"""
    frontrun_risk: float  # 0-1 scale
    backrun_risk: float   # 0-1 scale
    sandwich_risk: float  # 0-1 scale
    risk_factors: Dict[str, Any]
    timestamp: datetime

@dataclass
class RPCEndpoint:
    """RPC endpoint configuration and status"""
    url: str
    priority: int
    status: bool = True
    latency: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    error_count: int = 0

@dataclass
class ProtectedTransaction:
    """Transaction with MEV protection measures"""
    tx_hash: str
    token_address: str
    amount: Decimal
    timestamp: datetime
    protection_level: str
    private_routing: bool
    endpoints: List[str]
    status: str = "pending"

class MEVProtection:
    """Enhanced MEV protection system"""
    
    def __init__(self, settings: Any, jupiter_client: Any):
        self.settings = settings
        self.jupiter = jupiter_client
        self.mempool_cache: Dict[str, Any] = {}
        self.recent_txs: List[Dict[str, Any]] = []
        self.high_risk_tokens: set = set()
        self.last_analysis: Dict[str, MEVAnalysis] = {}
        
        # RPC endpoint management
        self.rpc_endpoints: Dict[str, RPCEndpoint] = {}
        self.current_endpoint: Optional[str] = None
        
        # Protection settings
        self.max_pending_txs = 3
        self.min_block_delay = 2
        self.max_price_deviation = 0.02  # 2%
        self.slippage_threshold = 0.01   # 1%
        
        # Initialize RPC endpoints
        self._initialize_rpc_endpoints()
        
    def _initialize_rpc_endpoints(self) -> None:
        """Initialize and validate RPC endpoints"""
        endpoints = self.settings.RPC_ENDPOINTS if hasattr(self.settings, 'RPC_ENDPOINTS') else {
            'primary': 'https://api.mainnet-beta.solana.com',
            'backup1': 'https://solana-api.projectserum.com',
            'backup2': 'https://rpc.ankr.com/solana'
        }
        
        for name, url in endpoints.items():
            self.rpc_endpoints[name] = RPCEndpoint(
                url=url,
                priority=len(self.rpc_endpoints)
            )
        
        # Set initial endpoint
        self.current_endpoint = next(iter(self.rpc_endpoints))

    async def analyze_mev_risk(self, token_address: str) -> Optional[MEVAnalysis]:
        """Analyze MEV risk for a token"""
        try:
            # Check mempool for pending transactions
            pending_txs = await self._get_pending_transactions(token_address)
            
            # Calculate risk metrics
            frontrun_risk = self._calculate_frontrun_risk(pending_txs)
            backrun_risk = self._calculate_backrun_risk(pending_txs)
            sandwich_risk = self._calculate_sandwich_risk(pending_txs)
            
            # Compile risk factors
            risk_factors = {
                'pending_tx_count': len(pending_txs),
                'high_value_pending': any(tx['value'] > 1 for tx in pending_txs),
                'recent_sandwich': self._check_recent_sandwich(token_address),
                'price_volatility': await self._get_price_volatility(token_address)
            }
            
            analysis = MEVAnalysis(
                frontrun_risk=frontrun_risk,
                backrun_risk=backrun_risk,
                sandwich_risk=sandwich_risk,
                risk_factors=risk_factors,
                timestamp=datetime.now()
            )
            
            self.last_analysis[token_address] = analysis
            return analysis
            
        except Exception as e:
            logger.error(f"MEV risk analysis error: {str(e)}")
            return None

    async def protect_transaction(
        self,
        token_address: str,
        amount: Decimal,
        slippage: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply MEV protection to a transaction"""
        try:
            # Get current mempool state
            pending = await self._get_pending_transactions(token_address)
            if len(pending) > self.max_pending_txs:
                return False, {"error": "Too many pending transactions"}
                
            # Calculate safe transaction parameters
            safe_params = await self._calculate_safe_parameters(
                token_address, amount, pending
            )
            
            if not safe_params:
                return False, {"error": "Could not determine safe parameters"}
                
            # Add protection measures
            protected_tx = await self._apply_protection_measures(
                token_address, safe_params
            )
            
            # Create protected transaction record
            tx_record = ProtectedTransaction(
                tx_hash=protected_tx.get('hash', ''),
                token_address=token_address,
                amount=amount,
                timestamp=datetime.now(),
                protection_level="high",
                private_routing=True,
                endpoints=list(self.rpc_endpoints.keys())
            )
            
            self.recent_txs.append(vars(tx_record))
            return True, protected_tx
            
        except Exception as e:
            logger.error(f"Transaction protection error: {str(e)}")
            return False, {"error": str(e)}

    async def _get_pending_transactions(self, token_address: str) -> List[Dict[str, Any]]:
        """Get pending transactions for token with multi-RPC support"""
        try:
            # Try primary endpoint if we have one set
            if self.current_endpoint and self.current_endpoint in self.rpc_endpoints:
                endpoint = self.rpc_endpoints[self.current_endpoint]
                try:
                    return await self._fetch_pending_txs(endpoint.url, token_address)
                except Exception as e:
                    logger.warning(f"Primary endpoint failed: {str(e)}")
            
            # Fallback to other endpoints
            for name, endpoint in self.rpc_endpoints.items():
                # Skip current endpoint as we already tried it
                if (self.current_endpoint is None or name != self.current_endpoint) and endpoint.status:
                    try:
                        txs = await self._fetch_pending_txs(endpoint.url, token_address)
                        self.current_endpoint = name
                        return txs
                    except Exception as e:
                        logger.warning(f"Endpoint {name} failed: {str(e)}")
                        endpoint.error_count += 1
                        
            logger.error("All RPC endpoints failed")
            return []
            
        except Exception as e:
            logger.error(f"Error getting pending transactions: {str(e)}")
            return []

    async def _fetch_pending_txs(self, rpc_url: str, token_address: str) -> List[Dict[str, Any]]:
        """Fetch pending transactions from specific RPC endpoint"""
        try:
            # Get mempool transactions
            response = await self.jupiter._make_request(
                "getSignaturesForAddress",
                [token_address, {"limit": 100}],
                rpc_url=rpc_url
            )
            
            if not response or 'result' not in response:
                return []
                
            # Filter and analyze transactions
            pending = []
            for tx in response['result']:
                if self._is_relevant_transaction(tx):
                    pending.append(tx)
                    
            return pending
            
        except Exception as e:
            logger.error(f"Error fetching from {rpc_url}: {str(e)}")
            raise

    def _is_relevant_transaction(self, tx: Dict[str, Any]) -> bool:
        """Check if transaction is relevant for MEV analysis"""
        return (
            tx.get('type') == 'swap' and
            not tx.get('confirmed', True) and
            datetime.fromtimestamp(tx.get('blockTime', 0)) > datetime.now() - timedelta(minutes=5)
        )

    def _calculate_frontrun_risk(self, pending_txs: List[Dict[str, Any]]) -> float:
        """Calculate risk of frontrunning"""
        if not pending_txs:
            return 0.0
            
        # Risk factors
        high_value = any(tx['value'] > 1 for tx in pending_txs)
        clustered = self._check_transaction_clustering(pending_txs)
        known_bots = any(
            tx['sender'] in self.high_risk_tokens for tx in pending_txs
        )
        
        # Weight and combine factors
        risk = 0.0
        if high_value:
            risk += 0.4
        if clustered:
            risk += 0.3
        if known_bots:
            risk += 0.3
            
        return min(risk, 1.0)

    def _calculate_backrun_risk(self, pending_txs: List[Dict[str, Any]]) -> float:
        """Calculate risk of backrunning"""
        if not pending_txs:
            return 0.0
            
        # Analyze transaction patterns
        pattern_risk = self._analyze_transaction_patterns(pending_txs)
        
        # Check for known backrunners
        backrunner_risk = any(
            tx['sender'] in self.high_risk_tokens for tx in pending_txs
        )
        
        return min(pattern_risk * 0.7 + (0.3 if backrunner_risk else 0), 1.0)

    def _calculate_sandwich_risk(self, pending_txs: List[Dict[str, Any]]) -> float:
        """Calculate risk of sandwich attacks"""
        if len(pending_txs) < 2:
            return 0.0
            
        # Look for sandwich patterns
        buy_sell_pairs = self._find_buy_sell_pairs(pending_txs)
        if not buy_sell_pairs:
            return 0.0
            
        # Calculate risk based on pattern strength
        pattern_strength = len(buy_sell_pairs) / len(pending_txs)
        known_attackers = any(
            pair['buyer'] in self.high_risk_tokens or 
            pair['seller'] in self.high_risk_tokens 
            for pair in buy_sell_pairs
        )
        
        return min(pattern_strength * 0.7 + (0.3 if known_attackers else 0), 1.0)

    async def _calculate_safe_parameters(
        self,
        token_address: str,
        amount: Decimal,
        pending_txs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Calculate safe transaction parameters"""
        try:
            # Get current price and market depth
            price_data = await self.jupiter.get_price(token_address)
            depth = await self.jupiter.get_market_depth(token_address)
            
            if not price_data or not depth:
                return None
                
            current_price = float(price_data['price'])
            
            # Calculate safe slippage
            slippage = self._calculate_safe_slippage(
                current_price,
                float(amount),
                depth,
                pending_txs
            )
            
            # Determine block submission
            block_delay = self._calculate_block_delay(pending_txs)
            
            return {
                'price': current_price,
                'slippage': slippage,
                'block_delay': block_delay,
                'priority_fee': self._calculate_priority_fee(pending_txs)
            }
            
        except Exception as e:
            logger.error(f"Error calculating safe parameters: {str(e)}")
            return None

    async def _apply_protection_measures(
        self,
        token_address: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply protection measures to transaction"""
        try:
            protected_tx = {
                'slippageBps': int(params['slippage'] * 10000),
                'deadline': int(
                    (datetime.now() + timedelta(seconds=30)).timestamp()
                ),
                'minTimestamp': int(
                    (datetime.now() + timedelta(
                        seconds=params['block_delay']
                    )).timestamp()
                ),
                'maxTimestamp': int(
                    (datetime.now() + timedelta(seconds=60)).timestamp()
                ),
                'priorityFee': params['priority_fee']
            }
            
            return protected_tx
            
        except Exception as e:
            logger.error(f"Error applying protection measures: {str(e)}")
            return {}

    def _check_transaction_clustering(self, txs: List[Dict[str, Any]]) -> bool:
        """Check for suspicious transaction clustering"""
        if len(txs) < 2:
            return False
            
        timestamps = [tx.get('blockTime', 0) for tx in txs]
        timestamps.sort()
        
        # Check for transactions too close together
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i-1] < 2:
                return True
                
        return False

    def _analyze_transaction_patterns(self, txs: List[Dict[str, Any]]) -> float:
        """Analyze transaction patterns for MEV risk"""
        if len(txs) < 2:
            return 0.0
            
        # Look for suspicious patterns
        risk_score = 0.0
        
        # Check for rapid succession transactions
        if self._check_transaction_clustering(txs):
            risk_score += 0.3
            
        # Check for size patterns
        if self._check_size_patterns(txs):
            risk_score += 0.3
            
        # Check for known MEV bots
        if any(tx['sender'] in self.high_risk_tokens for tx in txs):
            risk_score += 0.4
            
        return min(risk_score, 1.0)

    def _check_size_patterns(self, txs: List[Dict[str, Any]]) -> bool:
        """Check for suspicious transaction size patterns"""
        sizes = [tx.get('value', 0) for tx in txs]
        sizes.sort()
        
        # Look for sandwich pattern (small-large-small)
        if len(sizes) >= 3:
            for i in range(1, len(sizes)-1):
                if sizes[i] > sizes[i-1] * 3 and sizes[i] > sizes[i+1] * 3:
                    return True
                    
        return False

    def _find_buy_sell_pairs(self, txs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find potential sandwich attack patterns"""
        pairs = []
        for i in range(len(txs)-1):
            for j in range(i+1, len(txs)):
                if (
                    txs[i].get('type') == 'buy' and
                    txs[j].get('type') == 'sell' and
                    txs[j].get('blockTime', 0) - txs[i].get('blockTime', 0) < 2
                ):
                    pairs.append({
                        'buyer': txs[i].get('sender'),
                        'seller': txs[j].get('sender'),
                        'time_diff': txs[j].get('blockTime', 0) - txs[i].get('blockTime', 0)
                    })
        return pairs

    def _check_recent_sandwich(self, token_address: str) -> bool:
        """Check if token has been recently sandwiched"""
        recent_analysis = self.last_analysis.get(token_address)
        if not recent_analysis:
            return False
            
        # Check if there was high sandwich risk in last analysis
        return (recent_analysis.sandwich_risk > 0.7 and
                (datetime.now() - recent_analysis.timestamp) < timedelta(minutes=5))

    async def _get_price_volatility(self, token_address: str) -> float:
        """Get recent price volatility"""
        try:
            history = await self.jupiter.get_price_history(token_address)
            if not history:
                return 0.0
                
            prices = [float(price['price']) for price in history]
            if len(prices) < 2:
                return 0.0
                
            # Calculate volatility as standard deviation of returns
            returns = [
                (prices[i] - prices[i-1]) / prices[i-1]
                for i in range(1, len(prices))
            ]
            
            import numpy as np
            volatility = float(np.std(returns) * np.sqrt(365))  # Annualized
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating price volatility: {str(e)}")
            return 0.0

    def _calculate_safe_slippage(
        self,
        current_price: float,
        size: float,
        depth: Dict[str, Any],
        pending_txs: List[Dict[str, Any]]
    ) -> float:
        """Calculate safe slippage based on market conditions"""
        base_slippage = self.slippage_threshold
        
        # Adjust for pending transactions
        if len(pending_txs) > 0:
            base_slippage *= (1 + len(pending_txs) * 0.002)
            
        # Adjust for size relative to depth
        total_liquidity = float(depth.get('totalLiquidity', 0))
        if total_liquidity > 0:
            size_impact = (size / total_liquidity)
            base_slippage *= (1 + size_impact)
            
        return min(base_slippage, self.max_price_deviation)

    def _calculate_block_delay(self, pending_txs: List[Dict[str, Any]]) -> int:
        """Calculate safe block delay"""
        if len(pending_txs) == 0:
            return self.min_block_delay
            
        # Increase delay with more pending transactions
        return min(self.min_block_delay + len(pending_txs), 5)

    def _calculate_priority_fee(self, pending_txs: List[Dict[str, Any]]) -> int:
        """Calculate competitive priority fee"""
        if len(pending_txs) == 0:
            return 1
            
        # Get max priority fee from pending
        max_fee = max(
            (tx.get('priorityFee', 0) for tx in pending_txs),
            default=1
        )
        
        # Add small premium
        return int(max_fee * 1.1)

    async def get_protection_summary(self, token_address: str) -> Dict[str, Any]:
        """Get summary of MEV protection analysis"""
        analysis = await self.analyze_mev_risk(token_address)
        if not analysis:
            return {}
            
        return {
            "overall_risk": max(
                analysis.frontrun_risk,
                analysis.backrun_risk,
                analysis.sandwich_risk
            ),
            "risk_breakdown": {
                "frontrun": analysis.frontrun_risk,
                "backrun": analysis.backrun_risk,
                "sandwich": analysis.sandwich_risk
            },
            "risk_factors": analysis.risk_factors,
            "timestamp": analysis.timestamp.isoformat(),
            "recommendations": self._get_protection_recommendations(analysis)
        }
        
    def _get_protection_recommendations(self, analysis: MEVAnalysis) -> List[str]:
        """Get protection recommendations based on analysis"""
        recommendations = []
        
        if analysis.frontrun_risk > 0.7:
            recommendations.append(
                "High frontrun risk - Consider increasing block delay and priority fee"
            )
            
        if analysis.sandwich_risk > 0.7:
            recommendations.append(
                "High sandwich risk - Use tight slippage and avoid large trades"
            )
            
        if analysis.backrun_risk > 0.7:
            recommendations.append(
                "High backrun risk - Consider breaking trade into smaller sizes"
            )
            
        if len(analysis.risk_factors.get('pending_tx_count', 0)) > 5:
            recommendations.append(
                "High pending transaction count - Wait for clearer conditions"
            )
            
        return recommendations