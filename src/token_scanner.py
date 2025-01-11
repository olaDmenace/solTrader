import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, TypeVar, cast
from dataclasses import dataclass
import numpy as np
from .trend_detector import TrendDetector, TrendAnalysis
from .meme_detector import MemeDetector

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class TokenMetrics:
    price: float
    liquidity: float
    volume_24h: float
    buy_pressure: float
    holder_count: int
    holder_growth: float
    price_change_24h: float
    transaction_count_24h: int
    average_transaction_size: float

@dataclass
class ScreeningResult:
    metrics_score: float
    trend_score: float
    risk_score: float
    final_score: float
    passed_metrics: bool
    passed_trend: bool
    passed_risk: bool
    passed_overall: bool
    meme_analysis: Dict[str, Any]
    timestamp: datetime

class TokenScanner:
    def __init__(self, jupiter_client: Any, alchemy_client: Any) -> None:
        self.trend_detector = TrendDetector(jupiter_client)
        self.meme_detector = MemeDetector()
        self.jupiter = jupiter_client
        self.alchemy = alchemy_client
        self.known_tokens: set = set()
        self.potential_tokens: List[Dict[str, Any]] = []
        self.scan_history: List[Dict[str, Any]] = []
        self.error_count: int = 0
        self.last_successful_scan: Optional[datetime] = None
        self.screening_history: Dict[str, ScreeningResult] = {}

    async def scan_new_listings(self) -> List[Dict[str, Any]]:
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
                if not isinstance(token, dict) or 'address' not in token:
                    continue
                    
                if token['address'] in self.known_tokens:
                    continue

                try:
                    if await self._comprehensive_screen(token):
                        new_tokens.append(token)
                        self.known_tokens.add(token['address'])
                except Exception as e:
                    logger.error(f"Token screening error: {str(e)}")
                    continue

            self._update_scan_history(start_time, new_tokens)
            return new_tokens

        except Exception as e:
            logger.error(f"Scan error: {str(e)}")
            self.error_count += 1
            return []

    async def _comprehensive_screen(self, token: Dict[str, Any]) -> bool:
        try:
            if not self._validate_token_data(token):
                return False

            metrics = await self._get_token_metrics(token['address'])
            if not metrics:
                return False

            trend = await self.trend_detector.analyze_trend(token['address'])
            if not trend:
                return False

            metrics_result = self._apply_screening_criteria(metrics)
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
            ) or {}

            final_score = self._calculate_final_score({
                'metrics': metrics_result,
                'trend': trend_result,
                'risk': risk_result
            })

            screening_result = ScreeningResult(
                metrics_score=metrics_result['score'],
                trend_score=trend_result['score'],
                risk_score=risk_result['score'],
                final_score=final_score,
                passed_metrics=metrics_result['passed'],
                passed_trend=trend_result['passed'],
                passed_risk=risk_result['passed'],
                passed_overall=all([
                    metrics_result['passed'],
                    trend_result['passed'],
                    risk_result['passed'],
                    final_score >= 75
                ]),
                meme_analysis=meme_result,
                timestamp=datetime.now()
            )

            self.screening_history[token['address']] = screening_result
            return screening_result.passed_overall

        except Exception as e:
            logger.error(f"Screening error: {str(e)}")
            return False

    async def _get_token_metrics(self, token_address: str) -> Optional[TokenMetrics]:
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
                average_transaction_size=float(holder_data.get('avg_size', 0))
            )

        except Exception as e:
            logger.error(f"Metrics error: {str(e)}")
            return None

    def _apply_screening_criteria(self, metrics: TokenMetrics) -> Dict[str, Any]:
        score = 0.0
        criteria_met = {}

        if metrics.liquidity >= 10000:
            score += 30
            criteria_met['high_liquidity'] = True
        elif metrics.liquidity >= 5000:
            score += 20
            criteria_met['medium_liquidity'] = True

        if metrics.volume_24h >= 5000:
            score += 25
            criteria_met['high_volume'] = True
        elif metrics.volume_24h >= 1000:
            score += 15
            criteria_met['medium_volume'] = True

        if metrics.buy_pressure >= 1.5:
            score += 25
            criteria_met['high_pressure'] = True
        elif metrics.buy_pressure >= 1.2:
            score += 15
            criteria_met['medium_pressure'] = True

        holder_score = self._calculate_holder_score(metrics)
        score += holder_score
        criteria_met['holder_score'] = holder_score >= 10

        return {
            'passed': score >= 70,
            'score': score,
            'criteria': criteria_met
        }

    def _calculate_holder_score(self, metrics: TokenMetrics) -> float:
        score = 0.0
        
        if metrics.holder_growth >= 0.1:
            score += 10
        elif metrics.holder_growth >= 0.05:
            score += 5

        if metrics.holder_count >= 500:
            score += 10
        elif metrics.holder_count >= 100:
            score += 5

        return score

    def _evaluate_trend_criteria(self, trend: TrendAnalysis) -> Dict[str, Any]:
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
        risk_factors = {
            'low_liquidity': metrics.liquidity < 10000,
            'high_volatility': metrics.price_change_24h > 30,
            'low_holders': metrics.holder_count < 100,
            'unstable_trend': trend.confidence < 50,
            'low_volume': metrics.volume_24h < 5000
        }

        deductions = sum(20 for risk in risk_factors.values() if risk)
        score = max(0, 100 - deductions)

        return {
            'passed': score >= 60,
            'score': score,
            'risk_factors': risk_factors
        }

    def _calculate_final_score(self, results: Dict[str, Any]) -> float:
        weights = {'metrics': 0.4, 'trend': 0.4, 'risk': 0.2}
        weighted_sum = sum(
            results[key]['score'] * weight
            for key, weight in weights.items()
        )
        return round(float(weighted_sum), 2)

    async def _safe_api_call(self, func: Any, error_msg: str) -> Optional[Any]:
        retries = 3
        for attempt in range(retries):
            try:
                return await func()
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"{error_msg}: {str(e)}")
                    return None
                await asyncio.sleep(1 * (attempt + 1))

    def _validate_token_data(self, token: Dict[str, Any]) -> bool:
        return all(
            token.get(field) is not None
            for field in ['address', 'symbol', 'decimals']
        )

    async def _get_holder_data(self, token_address: str) -> Dict[str, Any]:
        # Placeholder implementation
        return {
            'count': 100,
            'growth': 0.05,
            'transactions': 150,
            'avg_size': 0.5
        }

    def _calculate_buy_pressure(self, market_data: Optional[Dict[str, Any]]) -> float:
        if not market_data:
            return 0.0

        try:
            bids = market_data.get('bids', [])
            asks = market_data.get('asks', [])

            buy_volume = sum(float(bid.get('size', 0)) for bid in bids)
            sell_volume = sum(float(ask.get('size', 0)) for ask in asks)

            return buy_volume / sell_volume if sell_volume > 0 else 0.0

        except Exception as e:
            logger.error(f"Buy pressure calculation error: {str(e)}")
            return 0.0

    def _update_scan_history(self, start_time: datetime, new_tokens: List[Dict[str, Any]]) -> None:
        duration = (datetime.now() - start_time).total_seconds()
        self.scan_history.append({
            'timestamp': start_time,
            'duration': duration,
            'tokens_found': len(new_tokens),
            'errors': self.error_count
        })
        self.last_successful_scan = datetime.now()
        self.error_count = 0
