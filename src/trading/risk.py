from .position import Position
import logging
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from .market_analyzer import MarketConditions



logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    daily_trades: int = 0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_risk: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    market_volatility: float = 0.0
    portfolio_beta: float = 1.0

@dataclass
class MarketCondition:
    volatility: float
    trend_strength: float
    liquidity_score: float
    market_regime: str  # 'trending', 'ranging', 'volatile'

class RiskManager:
    def __init__(self, settings):
        self.settings = settings
        self.metrics = RiskMetrics()
        self.position_risks: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.correlation_lookback = 20  # days for correlation calculation
        self.market_conditions: Optional[MarketCondition] = None
        self.max_drawdown = 0.0
        self.var_95 = 0.0
        self.sharpe_ratio = 0.0
        self.positions_correlation = {}
        self.portfolio_beta = 1.0

    def should_reset_daily(self) -> bool:
        """Check if daily metrics should be reset"""
        now = datetime.now()
        return (now - self.metrics.last_reset).days > 0

    def reset_daily_metrics(self) -> None:
        """Reset daily trading metrics"""
        self.metrics.daily_trades = 0
        self.metrics.daily_pnl = 0.0
        self.metrics.last_reset = datetime.now()

    def calculate_portfolio_metrics(self, positions: Dict[str, Position], 
        price_history: Dict[str, List[float]]) -> Dict[str, Any]:
        return {
            "var_95": self._calculate_var(price_history),
            "sharpe_ratio": self._calculate_sharpe(price_history),
            "max_drawdown": self._calculate_drawdown(price_history),
            "portfolio_beta": self._calculate_portfolio_beta(positions, self.market_conditions),
            "position_correlations": self._calculate_correlations(positions, price_history)
        }
    
    def _calculate_drawdown(self, price_history: Dict[str, List[float]]) -> float:
        try:
            prices = price_history.get('1h', [])
            if len(prices) < 2:
                return 0.0
            
            peak = prices[0]
            max_dd = 0.0
            
            for price in prices:
                if price > peak:
                    peak = price
                dd = (peak - price) / peak
                max_dd = max(max_dd, dd)
            
            return float(max_dd)
        except Exception as e:
            logger.error(f"Drawdown calculation error: {str(e)}")
            return 0.0

    def _calculate_var(self, price_history: Dict[str, List[float]], confidence: float = 0.95) -> float:
        try:
            prices = price_history.get('1h', [])
            if len(prices) < 2:
                return 0.0
                
            returns = np.diff(prices) / prices[:-1]
            return float(np.percentile(returns, (1 - confidence) * 100))
        except Exception as e:
            logger.error(f"VaR calculation error: {str(e)}")
            return 0.0

    def _calculate_sharpe(self, price_history: Dict[str, List[float]], risk_free_rate: float = 0.02) -> float:
        try:
            prices = price_history.get('1h', [])
            if len(prices) < 2:
                return 0.0

            returns = np.diff(prices) / prices[:-1]
            excess_returns = returns - (risk_free_rate / 365)
            
            if len(excess_returns) == 0:
                return 0.0
                
            return float(np.sqrt(365) * np.mean(excess_returns) / np.std(excess_returns))
        except Exception as e:
            logger.error(f"Sharpe ratio calculation error: {str(e)}")
            return 0.0

    def _calculate_correlations(self, positions: Dict[str, Position], 
                              price_history: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        correlations = {}
        for addr1 in positions:
            correlations[addr1] = {}
            for addr2 in positions:
                if addr1 != addr2:
                    corr = self._calculate_pair_correlation(addr1, addr2, price_history)
                    correlations[addr1][addr2] = corr
        return correlations

    def _calculate_pair_correlation(self, token1: str, token2: str, 
                                  price_history: Dict[str, List[float]]) -> float:
        try:
            prices1 = price_history.get(token1, [])
            prices2 = price_history.get(token2, [])
            
            if len(prices1) < 2 or len(prices2) < 2:
                return 0.0
                
            returns1 = np.diff(prices1) / prices1[:-1]
            returns2 = np.diff(prices2) / prices2[:-1]
            
            if len(returns1) != len(returns2):
                min_len = min(len(returns1), len(returns2))
                returns1 = returns1[-min_len:]
                returns2 = returns2[-min_len:]
                
            return float(np.corrcoef(returns1, returns2)[0, 1])
        except Exception as e:
            logger.error(f"Correlation calculation error: {str(e)}")
            return 0.0

    # In risk.py, update the signature and usage:

    def _calculate_portfolio_beta(self, positions: Dict[str, Position], 
        market_conditions: Optional[MarketCondition]) -> float:
        try:
            base_beta = 1.0
            
            if market_conditions:
                if market_conditions.market_regime == "volatile":
                    base_beta *= 0.7
                elif market_conditions.market_regime == "trending_up":
                    base_beta *= 1.2
                elif market_conditions.market_regime == "trending_down":
                    base_beta *= 0.8
                    
            return float(base_beta)
        except Exception as e:
            logger.error(f"Portfolio beta calculation error: {str(e)}")
            return 1.0

    def check_correlation_limits(self, token_address: str) -> bool:
        """Check correlation with existing positions"""
        if not self.price_history.get(token_address):
            return True  # No history available
            
        for pos_addr in self.position_risks.keys():
            if pos_addr == token_address:
                continue
                
            correlation = self._calculate_correlation(token_address, pos_addr)
            if correlation > 0.7:  # High correlation threshold
                return False
                
        return True

    def calculate_adjusted_portfolio_risk(self, new_position_risk: float) -> float:
        """Calculate risk-adjusted portfolio risk"""
        try:
            base_risk = self.metrics.current_risk + new_position_risk
            
            # Adjust for market conditions
            if self.market_conditions:
                volatility_factor = 1 + (self.market_conditions.volatility - 0.5)
                liquidity_factor = 1 - (self.market_conditions.liquidity_score - 0.5)
                base_risk *= volatility_factor * liquidity_factor
                
            # Adjust for portfolio beta
            base_risk *= self.metrics.portfolio_beta
            
            return base_risk
            
        except Exception as e:
            logger.error(f"Error calculating adjusted portfolio risk: {str(e)}")
            return float('inf')  # Return maximum risk on error

    def get_correlation_factor(self, token_address: str) -> float:
        """Calculate size adjustment based on correlations"""
        try:
            correlations = [
                self._calculate_correlation(token_address, addr)
                for addr in self.position_risks.keys()
            ]
            if not correlations:
                return 1.0
            
            avg_correlation = float(np.mean(correlations))
            return 1 - (avg_correlation * 0.5)  # Reduce size for high correlations
            
        except Exception as e:
            logger.error(f"Error calculating correlation factor: {str(e)}")
            return 1.0

    def get_regime_factor(self) -> float:
        """Calculate size adjustment based on market regime"""
        if not self.market_conditions:
            return 1.0
            
        regime_factors = {
            'volatile': 0.7,   # Reduce size in volatile markets
            'trending': 1.2,   # Increase size in trending markets
            'ranging': 1.0     # Normal size in ranging markets
        }
        
        return regime_factors.get(self.market_conditions.market_regime, 1.0)
    
    def calculate_risk_score(self, token_data: Dict[str, Any]) -> float:
        try:
            volume = float(token_data.get('volume24h', 0))
            liquidity = float(token_data.get('liquidity', 0))
            vol_score = min(volume / self.settings.MIN_VOLUME_24H, 1.0) * 40
            liq_score = min(liquidity / self.settings.MIN_LIQUIDITY, 1.0) * 40
            age_score = 20 if token_data.get('created_at') else 0
            return vol_score + liq_score + age_score
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 100  # Maximum risk on error
        

    def calculate_position_risk(self, 
                              position_size: float, 
                              entry_price: float, 
                              market_conditions: Optional[MarketCondition]) -> float:
        """Calculate risk percentage for a position"""
        try:
            # Basic risk calculation based on position size relative to total portfolio
            portfolio_value = self.settings.PORTFOLIO_VALUE
            position_value = position_size * entry_price
            base_risk = (position_value / portfolio_value) * 100

            # Adjust risk based on market conditions
            if market_conditions:
                volatility_adjustment = 1 + (market_conditions.volatility - 0.5)
                liquidity_adjustment = 1 - (market_conditions.liquidity_score - 0.5)
                return float(base_risk * volatility_adjustment * liquidity_adjustment)

            return float(base_risk)

        except Exception as e:
            logger.error(f"Error calculating position risk: {str(e)}")
            return 100.0  # Return max risk on error

    def can_open_position(self, token_address: str, position_size: float, entry_price: float) -> bool:
        """Enhanced position validation with correlation and market condition checks"""
        try:
            # Basic checks
            if self.should_reset_daily():
                self.reset_daily_metrics()
                
            if self.metrics.daily_trades >= self.settings.MAX_DAILY_TRADES:
                logger.warning("Daily trade limit reached")
                return False
                
            if self.metrics.daily_pnl <= -self.settings.MAX_DAILY_LOSS:
                logger.warning("Daily loss limit reached")
                return False

            # Check correlation with existing positions
            if not self.check_correlation_limits(token_address):  # Remove underscore
                logger.warning("Correlation limits exceeded")
                return False

            # Calculate position risk with market conditions
            position_risk = self.calculate_position_risk(
                position_size, 
                entry_price,
                self.market_conditions
            )
            
            # Calculate adjusted portfolio risk
            total_risk = self.calculate_adjusted_portfolio_risk(position_risk)  
            if total_risk > self.settings.MAX_PORTFOLIO_RISK:
                logger.warning(f"Adjusted portfolio risk too high: {total_risk:.2f}%")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking position risk: {str(e)}")
            return False
        
    async def calculate_market_volatility(self) -> float:
        """Calculate current market volatility"""
        try:
            if self.market_conditions:
                return self.market_conditions.volatility

            # If no market conditions available, use stored metric
            return self.metrics.market_volatility

        except Exception as e:
            logger.error(f"Error calculating market volatility: {str(e)}")
            return 0.5  # Return moderate volatility as default

    # ... [rest of your methods]

    def _calculate_correlation(self, token1: str, token2: str) -> float:
        """Calculate price correlation between two tokens"""
        try:
            prices1 = self.price_history.get(token1, [])
            prices2 = self.price_history.get(token2, [])
            
            if len(prices1) < 2 or len(prices2) < 2:
                return 0.0
                
            # Convert to numpy arrays and ensure same length
            prices1_array = np.array(prices1[-20:])
            prices2_array = np.array(prices2[-20:])
            
            # Calculate correlation
            correlation = float(np.corrcoef(prices1_array, prices2_array)[0, 1])
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0

    def calculate_position_size(self, 
                              token_address: str, 
                              max_size: float,
                              volatility: float,
                              market_impact: float) -> float:
        """Calculate optimal position size based on multiple factors"""
        try:
            # Base size from settings
            base_size = float(min(max_size, self.settings.MAX_TRADE_SIZE))
            
            # Adjust for volatility
            volatility_factor = float(1 - (volatility * 0.5))
            
            # Adjust for market impact
            impact_factor = float(1 - (market_impact * 2))
            
            # Adjust for correlation
            correlation_factor = self.get_correlation_factor(token_address)  # Remove underscore
            
            # Adjust for market regime
            regime_factor = self.get_regime_factor()  # Remove underscore
            
            final_size = float(base_size * volatility_factor * impact_factor * 
                             correlation_factor * regime_factor)
            
            # Apply minimum size constraint
            return float(max(final_size, self.settings.MIN_TRADE_SIZE))
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return float(self.settings.MIN_TRADE_SIZE)

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get enhanced risk metrics"""
        metrics = {
            "daily_trades": self.metrics.daily_trades,
            "daily_pnl": float(self.metrics.daily_pnl),
            "max_drawdown": float(self.metrics.max_drawdown),
            "current_risk": float(self.metrics.current_risk),
            "remaining_trades": self.settings.MAX_DAILY_TRADES - self.metrics.daily_trades,
            "risk_capacity": float(self.settings.MAX_PORTFOLIO_RISK - self.metrics.current_risk)
        }
        
        if self.market_conditions:
            metrics.update({
                "market_regime": self.market_conditions.market_regime,
                "market_volatility": float(self.metrics.market_volatility),
                "portfolio_beta": float(self.metrics.portfolio_beta),
                "liquidity_score": float(self.market_conditions.liquidity_score)
            })
            
        return metrics
    