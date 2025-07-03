# src/trading/advanced_risk.py

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    var_95: float
    es_95: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: Dict[str, Dict[str, float]]


@dataclass
class RiskLimits:
    max_drawdown: float = 0.05  # 5%
    max_position_size: float = 0.15  # 15%
    max_correlated_exposure: float = 0.25  # 25%
    max_portfolio_vol: float = 0.30  # 30%
    max_single_vol: float = 0.50  # 50%
    min_liquidity: float = 10000  # $10k
    var_limit: float = 0.02  # 2%


class EnhancedRiskManager:
    def __init__(self, settings):
        self.settings = settings
        self.limits = RiskLimits()
        self.historical_data = {}
        self.position_metrics = {}
        self.last_update = None
        self.monitoring_task = None

    async def start_monitoring(self):
        self.monitoring_task = asyncio.create_task(self._risk_monitor_loop())

    def _check_risk_breaches(self) -> None:
        """Check for risk limit breaches"""
        for token, position in self.position_metrics.items():
            if position.get('volatility', 0) > self.limits.max_single_vol:
                logger.warning(f"Volatility breach for {token}")
            if position.get('var', 0) > self.limits.var_limit:
                logger.warning(f"VaR breach for {token}")

    async def _get_current_price(self, token: str) -> Optional[float]:
        """Get current token price from market data"""
        try:
            price_data = await self.settings.jupiter.get_price(token)
            if isinstance(price_data, dict) and 'price' in price_data:
                return float(price_data['price'])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {token}: {str(e)}")
            return None

    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate period returns from price series"""
        if len(prices) < 2:
            return []
        return [((prices[i+1] - prices[i]) / prices[i]) for i in range(len(prices)-1)]

    async def _get_historical_data(self, token: str) -> List[float]:
        """Get historical price data for a token"""
        try:
            history = await self.settings.jupiter.get_price_history(token)
            if not history:
                return []

            prices = []
            for data_point in history:
                if isinstance(data_point, dict) and 'price' in data_point:
                    prices.append(float(data_point['price']))
            return prices
        except Exception as e:
            logger.error(f"Error getting historical data for {token}: {str(e)}")
            return []

    async def _calculate_liquidity_score(self, token: str) -> float:
        """Calculate liquidity score based on market depth"""
        depth = await self._get_market_depth(token)
        return min(depth / self.limits.min_liquidity, 1.0)
    
    async def _calculate_price_impact(self, token: str, size: float) -> float:
        """Calculate expected price impact"""
        try:
            market_depth = await self._get_market_depth(token)
            current_price = await self._get_current_price(token)

            if current_price is None or market_depth <= 0:
                return float('inf')

            impact = (size * current_price) / market_depth
            return impact
        except Exception as e:
            logger.error(f"Error calculating price impact: {str(e)}")
            return float('inf')

    async def _calculate_correlation_risk(self, token: str) -> float:
        """Calculate correlation-based risk score"""
        try:
            correlations = await self._calculate_correlations(token)
            if not correlations:
                return 0.0
            return max(correlations.values())
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {str(e)}")
            return 0.0
    
    def _calculate_portfolio_impact(self, size: float, entry_price: float) -> float:
        """Calculate trade's impact on portfolio value"""
        trade_value = size * entry_price
        return trade_value / self.settings.PORTFOLIO_VALUE    

    async def _calculate_correlations(self, token: str) -> Dict[str, float]:
        """Calculate correlations with existing positions"""
        correlations = {}
        try:
            token_prices = await self._get_historical_data(token)
            if not token_prices:
                return correlations

            token_returns = self._calculate_returns(token_prices)

            for pos_token in self.position_metrics:
                pos_prices = await self._get_historical_data(pos_token)
                if not pos_prices:
                    continue

                pos_returns = self._calculate_returns(pos_prices)

                # Ensure both return series have the same length
                min_len = min(len(token_returns), len(pos_returns))
                if min_len > 1:  # Need at least 2 points for correlation
                    token_returns_trimmed = token_returns[:min_len]
                    pos_returns_trimmed = pos_returns[:min_len]

                    correlation = np.corrcoef(token_returns_trimmed, pos_returns_trimmed)[0, 1]
                    if not np.isnan(correlation):  # Check for NaN values
                        correlations[pos_token] = float(correlation)
                    else:
                        correlations[pos_token] = 0.0

            return correlations
        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            return correlations

    async def _get_market_depth(self, token: str) -> float:
        """Get market depth from exchange"""
        try:
            depth = await self.settings.jupiter.get_market_depth(token)
            return float(depth.get("totalLiquidity", 0))
        except Exception as e:
            logger.error(f"Error getting market depth: {str(e)}")
            return 0.0

    def _get_correlation_adjustment(self, token: str) -> float:
        """Get position size adjustment based on correlations"""
        max_correlation = max(
            self.position_metrics.get(token, {}).get("correlations", {}).values(),
            default=0,
        )
        return 1.0 - (
            max_correlation * 0.5
        )  # Reduce size by up to 50% based on correlation

    async def _risk_monitor_loop(self):
        while True:
            try:
                await self._update_risk_metrics()
                await self._check_risk_breaches()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Risk monitoring error: {str(e)}")
                await asyncio.sleep(5)

    async def _update_risk_metrics(self):
        for token_address, position in self.position_metrics.items():
            price_data = position.get('price_history', [])
            returns = self._calculate_returns(price_data)
            position.update({
                'volatility': self._calculate_volatility(returns),
                'var': self._calculate_var(returns),
                'es': self._calculate_expected_shortfall(returns)
            })

    async def evaluate_trade(self, token: str, size: float, price: float) -> Dict:
        """Evaluate trade risk before execution"""
        try:
            metrics = await self._calculate_risk_metrics(token, size, price)

            checks = {
                "position_size": size <= self.limits.max_position_size,
                "portfolio_var": metrics["portfolio_var"] <= self.limits.var_limit,
                "correlation": await self._check_correlation_limits(token),
                "liquidity": await self._check_liquidity(token, size),
                "volatility": metrics["volatility"] <= self.limits.max_single_vol,
            }

            return {
                "approved": all(checks.values()),
                "metrics": metrics,
                "checks": checks,
            }

        except Exception as e:
            logger.error(f"Trade evaluation error: {str(e)}")
            return {"approved": False, "error": str(e)}

    async def _calculate_risk_metrics(
        self, token: str, size: float, price: float
    ) -> Dict:
        """Calculate comprehensive risk metrics"""
        historical_data = await self._get_historical_data(token)
        returns = self._calculate_returns(historical_data)

        metrics = {
            "volatility": self._calculate_volatility(returns),
            "var_95": self._calculate_var(returns),
            "es_95": self._calculate_expected_shortfall(returns),
            "liquidity_score": await self._calculate_liquidity_score(token),
            "correlation_risk": await self._calculate_correlation_risk(token),
            "portfolio_impact": self._calculate_portfolio_impact(size, price),
        }

        return metrics

    def _calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility"""
        return float(np.std(returns) * np.sqrt(252))

    def _calculate_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return float(np.percentile(returns, (1 - confidence) * 100))
    
    def _should_update_historical_data(self, token: str) -> bool:
        last_update = self.historical_data.get(token, {}).get('timestamp')
        return not last_update or (datetime.now() - last_update) > timedelta(minutes=5)
    
    def _check_liquidation_thresholds(self, position: Dict[str, Any]) -> bool:
        return (position.get('unrealized_pnl', 0) / position.get('size', 1) 
                < -self.limits.max_drawdown)

    def _calculate_expected_shortfall(
        self, returns: List[float], confidence: float = 0.95
    ) -> float:
        """Calculate Expected Shortfall (CVaR)"""
        var = self._calculate_var(returns, confidence)
        return float(np.mean([r for r in returns if r <= var]))

    async def _check_correlation_limits(self, token: str) -> bool:
        """Check correlation with existing positions"""
        correlations = await self._calculate_correlations(token)
        return (
            max(correlations.values(), default=0) <= self.limits.max_correlated_exposure
        )

    async def _check_liquidity(self, token: str, size: float) -> bool:
        """Verify sufficient liquidity for the trade"""
        market_depth = await self._get_market_depth(token)
        return market_depth >= self.limits.min_liquidity

    def calculate_position_sizing(self, token: str, account_size: float) -> float:
        """Calculate optimal position size"""
        vol = self.position_metrics.get(token, {}).get("volatility", 1.0)
        correlation_factor = self._get_correlation_adjustment(token)

        base_size = account_size * self.limits.max_position_size
        adjusted_size = base_size * (1 / vol) * correlation_factor

        return min(adjusted_size, account_size * self.limits.max_position_size)
