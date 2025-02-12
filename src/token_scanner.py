import asyncio
import logging
from typing import List, Dict, Any, Optional, TypeVar, cast
from dataclasses import dataclass
from datetime import datetime, timedelta
try:
    import numpy as np
except ImportError:
    import pip
    pip.main(['install', 'numpy'])
    import numpy as np
    
from .trend_detector import TrendDetector, TrendAnalysis
from .meme_detector import MemeDetector
from config.settings import Settings  # Adjust import path based on your project structure

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class TokenMetrics:
    """Token market and risk metrics"""
    price: float
    liquidity: float
    volume_24h: float
    buy_pressure: float
    holder_count: int
    holder_growth: float
    price_change_24h: float
    transaction_count_24h: int
    average_transaction_size: float
    launch_time: Optional[datetime] = None
    is_new_launch: bool = False
    contract_verified: bool = False
    contract_score: float = 0.0

@dataclass
class ScreeningResult:
    """Results from token screening process"""
    metrics_score: float
    trend_score: float
    risk_score: float
    final_score: float
    passed_metrics: bool
    passed_trend: bool
    passed_risk: bool
    passed_overall: bool
    meme_analysis: Dict[str, Any]
    launch_analysis: Dict[str, Any]
    timestamp: datetime

class TokenScanner:
    """Advanced token scanner with launch detection and sniping capabilities"""
    
    def __init__(self, jupiter_client: Any, alchemy_client: Any, settings: Settings) -> None:
        self.trend_detector = TrendDetector(jupiter_client)
        self.meme_detector = MemeDetector()
        self.jupiter = jupiter_client
        self.alchemy = alchemy_client
        self.settings = settings
        
        # Token tracking
        self.known_tokens: set = set()
        self.potential_tokens: List[Dict[str, Any]] = []
        self.scan_history: List[Dict[str, Any]] = []
        self.screening_history: Dict[str, ScreeningResult] = {}
        
        # Launch monitoring
        self.pending_launches: Dict[str, Dict[str, Any]] = {}
        self.monitored_launches: Dict[str, Dict[str, Any]] = {}
        self.launch_monitor_interval: float = 1.0  # 1 second for quick monitoring
        self.launch_timeout: int = 300  # 5 minutes monitor window
        
        # Gas optimization
        self.gas_settings = {
            'priority_fee': 10,  # GWEI
            'max_fee': 100,      # GWEI
            'base_priority': 2   # Base priority multiplier
        }
        
        # Error tracking
        self.error_count: int = 0
        self.last_successful_scan: Optional[datetime] = None

    async def scan_new_listings(self) -> List[Dict[str, Any]]:
        """Scan for new token listings with launch detection"""
        try:
            start_time = datetime.now()
            tokens = await self._safe_api_call(
                self.jupiter.get_tokens_list,
                "Failed to get tokens list"
            )
            
            if not tokens:
                self.error_count += 1
                return []

            new_tokens = []
            for token in tokens:
                try:
                    if not self._validate_token_data(token):
                        continue
                        
                    if token['address'] in self.known_tokens:
                        continue

                    # Check if new launch
                    is_new = await self._is_new_launch(token)
                    if is_new:
                        await self._handle_new_launch(token)

                    # Comprehensive screening
                    if await self._comprehensive_screen(token, is_new):
                        new_tokens.append(token)
                        self.known_tokens.add(token['address'])

                except Exception as e:
                    logger.error(f"Token analysis error: {str(e)}")
                    continue

            self._update_scan_history(start_time, new_tokens)
            return new_tokens

        except Exception as e:
            logger.error(f"Scan error: {str(e)}")
            self.error_count += 1
            return []

    def _validate_token_data(self, token: Dict[str, Any]) -> bool:
        """Validate basic token data"""
        return all(
            token.get(field) is not None 
            for field in ['address', 'symbol', 'decimals']
        )

    async def _is_new_launch(self, token: Dict[str, Any]) -> bool:
        """Determine if token is a new launch"""
        try:
            # Check contract creation time
            creation_time = await self._get_contract_creation_time(token['address'])
            if not creation_time:
                return False

            # Consider new if less than 30 minutes old
            age = datetime.now() - creation_time
            if age <= timedelta(minutes=30):
                return True

            # Check first liquidity addition
            first_liq = await self._get_first_liquidity_time(token['address'])
            if first_liq and (datetime.now() - first_liq) <= timedelta(minutes=30):
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking launch status: {str(e)}")
            return False

    async def _handle_new_launch(self, token: Dict[str, Any]) -> None:
        """Process newly launched token and execute trades if conditions are met"""
        try:
            # Quick initial screening
            contract_score = await self._analyze_contract(token['address'])
            if contract_score < 50:  # Fail fast for suspicious contracts
                logger.warning(f"Suspicious contract detected: {token['address']}")
                return
                
            # Get initial metrics
            current_price = await self._get_price(token['address'])
            initial_liquidity = await self._get_liquidity(token['address'])
            
            if not current_price or not initial_liquidity:
                logger.warning(f"Could not get price/liquidity for {token['address']}")
                return
                
            initial_data = {
                'token': token,
                'discovery_time': datetime.now(),
                'initial_price': current_price,
                'initial_liquidity': initial_liquidity,
                'contract_score': contract_score,
                'monitoring_start': datetime.now()
            }

            # Execute buy if conditions are met
            if await self._should_execute_buy(token['address'], current_price, initial_liquidity, contract_score):
                entry_size = self._calculate_entry_size(current_price, initial_liquidity)
                if entry_size > 0:
                    success = await self._execute_buy(
                        token_address=token['address'],
                        size=entry_size,
                        price=current_price
                    )
                    if success:
                        logger.info(f"Successfully entered position for {token['address']}")
                        initial_data['entry_executed'] = True
                        initial_data['entry_size'] = entry_size
                        initial_data['entry_price'] = current_price

            # Start monitoring regardless of whether we entered
            self.pending_launches[token['address']] = initial_data
            asyncio.create_task(self._monitor_launch(token['address']))

        except Exception as e:
            logger.error(f"Error handling new launch: {str(e)}")

    async def _analyze_contract(self, token_address: str) -> float:
        """Analyze contract for security and risks"""
        try:
            contract_code = await self.alchemy.get_contract_code(token_address)
            if not contract_code:
                return 0.0

            score = 50.0  # Start neutral
            
            # Check for dangerous functions
            danger_patterns = {
                'selfdestruct': -20,
                'delegatecall': -15,
                'transfer.*(owner|creator)': -15,
                'mint.*unlimited': -20,
                'blacklist': -10,
                'pause': -5,
                'excludeFromFee': -5
            }
            
            # Check for security features
            security_patterns = {
                'onlyOwner': 5,
                'require': 5,
                'revert': 5,
                'SafeMath': 10,
                'maxTransaction': 5,
                'liquidityLock': 15
            }

            # Apply pattern scoring
            for pattern, penalty in danger_patterns.items():
                if pattern in contract_code.lower():
                    score += penalty

            for pattern, bonus in security_patterns.items():
                if pattern in contract_code:
                    score += bonus

            # Normalize score
            return max(0.0, min(100.0, score))

        except Exception as e:
            logger.error(f"Contract analysis error: {str(e)}")
            return 0.0

    async def _monitor_launch(self, token_address: str) -> None:
        """Monitor new token launch for suspicious activity"""
        try:
            launch_data = self.pending_launches[token_address]
            start_time = launch_data['monitoring_start']
            
            while (datetime.now() - start_time) < timedelta(seconds=self.launch_timeout):
                try:
                    # Get current metrics
                    current_price = await self._get_price(token_address)
                    current_liquidity = await self._get_liquidity(token_address)
                    
                    if not current_price or not current_liquidity:
                        break

                    # Calculate changes
                    price_change = (
                        (current_price - launch_data['initial_price']) 
                        / launch_data['initial_price']
                    )
                    liquidity_change = (
                        (current_liquidity - launch_data['initial_liquidity'])
                        / launch_data['initial_liquidity']
                    )

                    # Check for suspicious activity
                    if self._is_suspicious_activity(price_change, liquidity_change):
                        logger.warning(
                            f"Suspicious activity for {token_address}: "
                            f"Price change: {price_change:.2%}, "
                            f"Liquidity change: {liquidity_change:.2%}"
                        )
                        break

                    await asyncio.sleep(self.launch_monitor_interval)

                except Exception as e:
                    logger.error(f"Launch monitoring error: {str(e)}")
                    break

            # Store monitoring results
            self.monitored_launches[token_address] = {
                **launch_data,
                'final_price': await self._get_price(token_address),
                'final_liquidity': await self._get_liquidity(token_address),
                'monitoring_end': datetime.now()
            }
            
            # Cleanup
            if token_address in self.pending_launches:
                del self.pending_launches[token_address]

        except Exception as e:
            logger.error(f"Launch monitoring error: {str(e)}")

    def _is_suspicious_activity(self, price_change: float, liquidity_change: float) -> bool:
        """Detect suspicious trading activity"""
        return (
            abs(price_change) > 0.5 or  # 50% price change
            liquidity_change < -0.3 or  # 30% liquidity decrease
            (price_change > 0.3 and liquidity_change < -0.15)  # Pump and dump pattern
        )

    async def _get_contract_creation_time(self, token_address: str) -> Optional[datetime]:
        """Get contract creation timestamp"""
        try:
            tx = await self.alchemy.get_token_first_transaction(token_address)
            if tx and 'timestamp' in tx:
                return datetime.fromtimestamp(tx['timestamp'])
            return None
        except Exception as e:
            logger.error(f"Error getting contract creation time: {str(e)}")
            return None

    async def _get_first_liquidity_time(self, token_address: str) -> Optional[datetime]:
        """Get first liquidity addition timestamp"""
        try:
            tx = await self.alchemy.get_token_first_liquidity_tx(token_address)
            if tx and 'timestamp' in tx:
                return datetime.fromtimestamp(tx['timestamp'])
            return None
        except Exception as e:
            logger.error(f"Error getting first liquidity time: {str(e)}")
            return None

    async def _get_price(self, token_address: str) -> Optional[float]:
        """Get current token price"""
        try:
            price_data = await self.jupiter.get_price(token_address)
            return float(price_data['price']) if price_data else None
        except Exception as e:
            logger.error(f"Error getting price: {str(e)}")
            return None

    async def _get_liquidity(self, token_address: str) -> Optional[float]:
        """Get current token liquidity"""
        try:
            depth = await self.jupiter.get_market_depth(token_address)
            return float(depth['totalLiquidity']) if depth else None
        except Exception as e:
            logger.error(f"Error getting liquidity: {str(e)}")
            return None

    async def _comprehensive_screen(self, token: Dict[str, Any], is_new_launch: bool = False) -> bool:
        """Perform comprehensive token screening"""
        try:
            if not self._validate_token_data(token):
                return False

            metrics = await self._get_token_metrics(token['address'])
            if not metrics:
                return False

            trend = await self.trend_detector.analyze_trend(token['address'])
            if not trend:
                return False

            metrics_result = self._apply_screening_criteria(metrics, is_new_launch)
            trend_result = self._evaluate_trend_criteria(trend)
            risk_result = self._assess_risk_factors(metrics, trend)
            
            meme_result = self.meme_detector.analyze_token(
                token.get('name', ''),
                token.get('symbol', ''),
                {
                    'created_at': token.get('created_at'),
                    'is_verified': token.get('is_verified', False),
                    'total_supply': token.get('total_supply'),
                    'holder_distribution': token.get('holder_distribution', {})
                }
            )

            # Include launch analysis for new tokens
            launch_result = {}
            if is_new_launch:
                launch_result = self._get_launch_analysis(token['address'])

            final_score = self._calculate_final_score({
                'metrics': metrics_result,
                'trend': trend_result,
                'risk': risk_result,
                'launch': launch_result
            })

            result = ScreeningResult(
                metrics_score=metrics_result['score'],
                trend_score=trend_result['score'],
                risk_score=risk_result['score'],
                final_score=final_score,
                passed_metrics=metrics_result['passed'],
                passed_trend=trend_result['passed'],
                passed_risk=risk_result['passed'],
                passed_overall=self._passed_overall_screening(
                    metrics_result, trend_result, risk_result, final_score
                ),
                meme_analysis=meme_result,
                launch_analysis=launch_result,
                timestamp=datetime.now()
            )

            self.screening_history[token['address']] = result
            return result.passed_overall

        except Exception as e:
            logger.error(f"Screening error: {str(e)}")
            return False

    def _get_launch_analysis(self, token_address: str) -> Dict[str, Any]:
        """Get launch analysis data"""
        launch_data = self.monitored_launches.get(token_address, {})
        if not launch_data:
            return {}

        return {
            'launch_time': launch_data.get('discovery_time'),
            'initial_price': launch_data.get('initial_price'),
            'initial_liquidity': launch_data.get('initial_liquidity'),
            'contract_score': launch_data.get('contract_score'),
            'monitoring_duration': (
                launch_data.get('monitoring_end') - 
                launch_data.get('monitoring_start')
            ).total_seconds() if launch_data.get('monitoring_end') else 0,
            'price_change': (
                (launch_data.get('final_price', 0) - launch_data.get('initial_price', 0))
                / launch_data.get('initial_price', 1)
            ) if launch_data.get('initial_price') else 0,
            'liquidity_change': (
                (launch_data.get('final_liquidity', 0) - launch_data.get('initial_liquidity', 0))
                / launch_data.get('initial_liquidity', 1)
            ) if launch_data.get('initial_liquidity') else 0,
            'suspicious_activity': self._is_suspicious_activity(
                (launch_data.get('final_price', 0) - launch_data.get('initial_price', 0))
                / launch_data.get('initial_price', 1) if launch_data.get('initial_price') else 0,
                (launch_data.get('final_liquidity', 0) - launch_data.get('initial_liquidity', 0))
                / launch_data.get('initial_liquidity', 1) if launch_data.get('initial_liquidity') else 0
            )
        }

    async def _get_token_metrics(self, token_address: str) -> Optional[TokenMetrics]:
        """Get comprehensive token metrics"""
        try:
            price_data, market_data, holder_data = await asyncio.gather(
                self._safe_api_call(
                    lambda: self.jupiter.get_price(token_address),
                    "Price data fetch failed"
                ),
                self._safe_api_call(
                    lambda: self.jupiter.get_market_depth(token_address),
                    "Market depth fetch failed"
                ),
                self._get_holder_data(token_address)
            )

            if not all([price_data, market_data, holder_data]):
                return None

            return TokenMetrics(
                price=float(price_data.get('price', 0)),
                liquidity=float(market_data.get('totalLiquidity', 0)),
                volume_24h=float(market_data.get('volume24h', 0)),
                buy_pressure=self._calculate_buy_pressure(market_data),
                holder_count=int(holder_data.get('count', 0)),
                holder_growth=float(holder_data.get('growth', 0)),
                price_change_24h=float(price_data.get('change24h', 0)),
                transaction_count_24h=int(holder_data.get('transactions', 0)),
                average_transaction_size=float(holder_data.get('avg_size', 0)),
                launch_time=await self._get_contract_creation_time(token_address),
                is_new_launch=await self._is_new_launch({'address': token_address}),
                contract_verified=bool(holder_data.get('is_verified', False)),
                contract_score=await self._analyze_contract(token_address)
            )

        except Exception as e:
            logger.error(f"Error getting token metrics: {str(e)}")
            return None

    async def _get_holder_data(self, token_address: str) -> Dict[str, Any]:
        """Get token holder information"""
        try:
            response = await self.alchemy.get_token_holders(token_address)
            if not response:
                return {}

            holders = response.get('holders', [])
            total_holders = len(holders)
            if total_holders == 0:
                return {}

            growth_rate = 0.0
            if 'historical_holders' in response:
                prev_holders = len(response['historical_holders'])
                if prev_holders > 0:
                    growth_rate = (total_holders - prev_holders) / prev_holders

            return {
                'count': total_holders,
                'growth': growth_rate,
                'transactions': response.get('total_transactions', 0),
                'avg_size': response.get('average_transaction_size', 0),
                'is_verified': response.get('is_verified', False)
            }

        except Exception as e:
            logger.error(f"Error getting holder data: {str(e)}")
            return {}

    def _apply_screening_criteria(self, metrics: TokenMetrics, is_new_launch: bool) -> Dict[str, Any]:
        """Apply screening criteria with adjusted thresholds for new launches"""
        score = 0.0
        criteria_met = {}

        # Adjust thresholds for new launches
        liquidity_threshold = self.settings.MIN_LIQUIDITY * (0.5 if is_new_launch else 1.0)
        volume_threshold = self.settings.MIN_VOLUME_24H * (0.5 if is_new_launch else 1.0)

        if metrics.liquidity >= liquidity_threshold:
            score += 30
            criteria_met['high_liquidity'] = True
        elif metrics.liquidity >= liquidity_threshold * 0.5:
            score += 20
            criteria_met['medium_liquidity'] = True

        if metrics.volume_24h >= volume_threshold:
            score += 25
            criteria_met['high_volume'] = True
        elif metrics.volume_24h >= volume_threshold * 0.5:
            score += 15
            criteria_met['medium_volume'] = True

        if metrics.buy_pressure >= 1.5:
            score += 25
            criteria_met['high_pressure'] = True
        elif metrics.buy_pressure >= 1.2:
            score += 15
            criteria_met['medium_pressure'] = True

        holder_score = self._calculate_holder_score(metrics, is_new_launch)
        score += holder_score
        criteria_met['holder_score'] = holder_score >= 10

        # Add contract analysis score for new launches
        if is_new_launch:
            contract_score = metrics.contract_score / 2  # Scale to max 50 points
            score += contract_score
            criteria_met['contract_score'] = contract_score >= 25

        return {
            'passed': score >= (60 if is_new_launch else 70),
            'score': score,
            'criteria': criteria_met
        }

    def _calculate_holder_score(self, metrics: TokenMetrics, is_new_launch: bool) -> float:
        """Calculate holder-based score with adjustments for new launches"""
        score = 0.0
        
        if is_new_launch:
            # Different thresholds for new launches
            if metrics.holder_count >= 50:  # Lower threshold for new tokens
                score += 10
            elif metrics.holder_count >= 20:
                score += 5
                
            if metrics.holder_growth >= 0.2:  # Higher growth expectations
                score += 10
            elif metrics.holder_growth >= 0.1:
                score += 5
        else:
            # Standard thresholds
            if metrics.holder_count >= 500:
                score += 10
            elif metrics.holder_count >= 100:
                score += 5
                
            if metrics.holder_growth >= 0.1:
                score += 10
            elif metrics.holder_growth >= 0.05:
                score += 5

        return score

    def _evaluate_trend_criteria(self, trend: TrendAnalysis) -> Dict[str, Any]:
        """Evaluate trend analysis criteria"""
        score = 0.0
        criteria = {
            'uptrend': trend.trend_direction == "uptrend",
            'strong_momentum': trend.momentum_score > 0.5,
            'volume_increasing': trend.volume_trend == "increasing",
            'bullish_pattern': trend.price_pattern in ["breakout", "consolidation"],
            'high_confidence': trend.confidence >= 70
        }

        weights = {
            'uptrend': 30,
            'strong_momentum': 20,
            'volume_increasing': 20,
            'bullish_pattern': 15,
            'high_confidence': 15
        }

        for criterion, passed in criteria.items():
            if passed:
                score += weights[criterion]

        return {
            'passed': score >= 70,
            'score': score,
            'criteria': criteria
        }

    def _assess_risk_factors(self, metrics: TokenMetrics, trend: TrendAnalysis) -> Dict[str, Any]:
        """Assess risk factors with contract analysis"""
        risk_factors = {
            'low_liquidity': metrics.liquidity < self.settings.MIN_LIQUIDITY,
            'high_volatility': metrics.price_change_24h > 30,
            'low_holders': metrics.holder_count < 100,
            'unstable_trend': trend.confidence < 50,
            'low_volume': metrics.volume_24h < self.settings.MIN_VOLUME_24H,
            'contract_risk': metrics.contract_score < 50 if metrics.contract_score > 0 else False
        }

        # Weight contract risk more heavily for new launches
        if metrics.is_new_launch:
            contract_deduction = 30 if risk_factors['contract_risk'] else 0
            standard_deductions = sum(20 for risk in risk_factors.values() if risk) - (20 if risk_factors['contract_risk'] else 0)
            score = max(0, 100 - standard_deductions - contract_deduction)
        else:
            deductions = sum(20 for risk in risk_factors.values() if risk)
            score = max(0, 100 - deductions)

        return {
            'passed': score >= 60,
            'score': score,
            'risk_factors': risk_factors
        }

    def _calculate_final_score(self, results: Dict[str, Any]) -> float:
        """Calculate final score with launch considerations"""
        weights = {
            'metrics': 0.35,
            'trend': 0.35,
            'risk': 0.2,
            'launch': 0.1  # New launch analysis weight
        }
        
        weighted_sum = sum(
            results[key]['score'] * weight
            for key, weight in weights.items()
            if key in results and 'score' in results[key]
        )
        
        # Normalize weights if launch analysis is missing
        if 'launch' not in results:
            total_weight = sum(weight for key, weight in weights.items() if key != 'launch')
            weighted_sum = weighted_sum / total_weight * (total_weight + weights['launch'])
            
        return round(weighted_sum, 2)

    def _passed_overall_screening(
        self,
        metrics_result: Dict[str, Any],
        trend_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        final_score: float
    ) -> bool:
        """Determine if token passes overall screening"""
        return all([
            metrics_result['passed'],
            trend_result['passed'],
            risk_result['passed'],
            final_score >= 75
        ])

    async def _safe_api_call(self, func: Any, error_msg: str) -> Optional[Any]:
        """Make safe API call with retries"""
        retries = 3
        for attempt in range(retries):
            try:
                return await func()
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"{error_msg}: {str(e)}")
                    return None
                await asyncio.sleep(1 * (attempt + 1))

    def _update_scan_history(self, start_time: datetime, new_tokens: List[Dict[str, Any]]) -> None:
        """Update scan history"""
        duration = (datetime.now() - start_time).total_seconds()
        self.scan_history.append({
            'timestamp': start_time,
            'duration': duration,
            'tokens_found': len(new_tokens),
            'errors': self.error_count
        })
        self.last_successful_scan = datetime.now()
        self.error_count = 0
        
    def _calculate_buy_pressure(self, market_data: Optional[Dict[str, Any]]) -> float:
        """Calculate buy pressure from order book data"""
        try:
            if not market_data:
                return 0.0
                
            bids = market_data.get('bids', [])
            asks = market_data.get('asks', [])
            
            buy_volume = sum(float(bid.get('size', 0)) for bid in bids)
            sell_volume = sum(float(ask.get('size', 0)) for ask in asks)
            
            if sell_volume == 0:
                return 0.0
                
            return float(buy_volume / sell_volume)
            
        except Exception as e:
            logger.error(f"Error calculating buy pressure: {str(e)}")
            return 0.0

    async def _should_execute_buy(
        self, 
        token_address: str, 
        current_price: float,
        liquidity: float,
        contract_score: float
    ) -> bool:
        """Determine if buy should be executed"""
        try:
            # Minimum requirements
            if liquidity < self.settings.MIN_LIQUIDITY * 0.5:  # Lower threshold for new launches
                return False
                
            if contract_score < 70:  # Higher requirement for actual entry
                return False
                
            # Get market depth to check if entry is feasible
            depth = await self.jupiter.get_market_depth(token_address)
            if not depth:
                return False
                
            # Calculate potential price impact
            entry_size = self._calculate_entry_size(current_price, liquidity)
            impact = self._calculate_price_impact(depth, entry_size)
            
            if impact > self.settings.MAX_PRICE_IMPACT:
                return False
                
            # Additional entry criteria specific to launches
            holders = await self._get_holder_data(token_address)
            if holders.get('count', 0) < 10:  # Minimum holder requirement
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking buy conditions: {str(e)}")
            return False
            
    def _calculate_entry_size(self, price: float, liquidity: float) -> float:
        """Calculate optimal entry size"""
        try:
            # Base size on liquidity
            max_size = min(
                self.settings.MAX_TRADE_SIZE,
                liquidity * 0.02  # Max 2% of liquidity
            )
            
            # Adjust based on price
            adjusted_size = min(
                max_size,
                self.settings.MAX_POSITION_SIZE / price
            )
            
            return max(adjusted_size, self.settings.MIN_TRADE_SIZE)
            
        except Exception as e:
            logger.error(f"Error calculating entry size: {str(e)}")
            return 0.0
            
    def _calculate_price_impact(self, market_depth: Dict[str, Any], size: float) -> float:
        """Calculate expected price impact percentage"""
        try:
            bids = market_depth.get('bids', [])
            total_liquidity = sum(float(bid.get('size', 0)) for bid in bids)
            
            if total_liquidity == 0:
                return 100.0
                
            return (size / total_liquidity) * 100
            
        except Exception as e:
            logger.error(f"Error calculating price impact: {str(e)}")
            return 100.0
            
    async def _execute_buy(self, token_address: str, size: float, price: float) -> bool:
        """Execute buy order with optimized gas"""
        try:
            # Get optimal gas price
            base_fee = await self.jupiter._make_request("getBaseFee", [])
            priority_fee = self.gas_settings['priority_fee']
            max_fee = min(
                base_fee + priority_fee * self.gas_settings['base_priority'],
                self.gas_settings['max_fee']
            )
            
            # Execute swap through Jupiter
            success = await self.jupiter.execute_swap(
                input_token="So11111111111111111111111111111111111111112",  # SOL
                output_token=token_address,
                amount=size,
                slippage=self.settings.MAX_SLIPPAGE,
                priority_fee=priority_fee,
                max_fee=max_fee
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing buy: {str(e)}")
            return False