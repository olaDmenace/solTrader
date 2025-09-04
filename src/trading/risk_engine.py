import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.database.db_manager import DatabaseManager
from src.utils.trading_time import trading_time


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskEngineConfig:
    max_position_size: float = 0.05  # 5% max per position
    max_portfolio_risk: float = 0.15  # 15% max portfolio risk
    max_daily_loss: float = 0.10  # 10% max daily loss
    max_drawdown: float = 0.20  # 20% max drawdown
    min_liquidity_threshold: float = 1000.0
    max_correlation: float = 0.7
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    volatility_adjustment: bool = True
    risk_free_rate: float = 0.02  # 2% annual risk-free rate


@dataclass
class TradeRisk:
    symbol: str
    position_size: float
    risk_level: RiskLevel
    risk_score: float
    max_loss: float
    confidence: float
    liquidity_risk: float
    concentration_risk: float
    volatility_risk: float
    recommendation: str
    metadata: Dict[str, Any]


class RiskEngine:
    def __init__(self, db_manager: DatabaseManager, config: RiskEngineConfig):
        self.db_manager = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk tracking
        self.position_risks: Dict[str, TradeRisk] = {}
        self.portfolio_risk_metrics: Dict[str, float] = {}
        self.daily_pnl_history: List[float] = []
        self.portfolio_value_history: List[float] = []
        
    async def initialize(self):
        """Initialize risk engine"""
        try:
            # Initialize risk tracking tables
            await self._create_risk_tables()
            
            # Load historical data
            await self._load_historical_data()
            
            self.logger.info("Risk engine initialized")
            
        except Exception as e:
            self.logger.error(f"Risk engine initialization failed: {e}")
            raise
            
    async def _create_risk_tables(self):
        """Create risk management tables"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            # Risk assessments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_assessments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    max_loss REAL NOT NULL,
                    confidence REAL NOT NULL,
                    recommendation TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # Portfolio risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to create risk tables: {e}")
            raise
            
    async def _load_historical_data(self):
        """Load historical risk and performance data"""
        try:
            # Load recent portfolio metrics
            recent_metrics = await self.db_manager.get_metrics("portfolio_value", hours=24)
            self.portfolio_value_history = [m['metric_value'] for m in recent_metrics[-100:]]
            
            # Load recent PnL data
            pnl_metrics = await self.db_manager.get_metrics("daily_pnl", hours=24*7)
            self.daily_pnl_history = [m['metric_value'] for m in pnl_metrics[-30:]]
            
        except Exception as e:
            self.logger.warning(f"Could not load historical risk data: {e}")
            
    async def assess_trade_risk(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        strategy_name: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> TradeRisk:
        """Comprehensive trade risk assessment"""
        try:
            # Calculate position size as percentage of portfolio
            position_value = quantity * price
            portfolio_value = self._get_current_portfolio_value()
            position_size = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Check multi-strategy cumulative risk FIRST
            cumulative_risk = await self._check_multi_strategy_risk(strategy_name, position_value)
            if cumulative_risk > self.config.max_portfolio_risk:
                # Immediately reject if multi-strategy risk too high
                return TradeRisk(
                    symbol=symbol,
                    position_size=position_size,
                    risk_level=RiskLevel.CRITICAL,
                    risk_score=100.0,
                    max_loss=position_value,
                    confidence=0.0,
                    liquidity_risk=50.0,
                    concentration_risk=100.0,  # Max because of multi-strategy risk
                    volatility_risk=50.0,
                    recommendation="REJECT - Multi-strategy risk limit exceeded",
                    metadata={
                        "cumulative_risk": cumulative_risk,
                        "max_allowed": self.config.max_portfolio_risk,
                        "strategy_name": strategy_name
                    }
                )
            
            # Individual risk components
            liquidity_risk = await self._assess_liquidity_risk(symbol, position_value)
            concentration_risk = await self._assess_concentration_risk(symbol, position_size)
            volatility_risk = await self._assess_volatility_risk(symbol, position_size)
            
            # Combined risk score (0-100)
            risk_score = (
                liquidity_risk * 0.3 +
                concentration_risk * 0.4 +
                volatility_risk * 0.3
            )
            
            # Risk level classification
            if risk_score < 25:
                risk_level = RiskLevel.LOW
            elif risk_score < 50:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 75:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
                
            # Calculate maximum potential loss
            volatility = await self._get_symbol_volatility(symbol)
            max_loss = position_value * min(volatility * 2, 0.5)  # Cap at 50%
            
            # Trading confidence based on risk
            confidence = max(0.1, 1.0 - (risk_score / 100))
            
            # Generate recommendation
            recommendation = self._generate_recommendation(risk_level, risk_score, position_size)
            
            trade_risk = TradeRisk(
                symbol=symbol,
                position_size=position_size,
                risk_level=risk_level,
                risk_score=risk_score,
                max_loss=max_loss,
                confidence=confidence,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                volatility_risk=volatility_risk,
                recommendation=recommendation,
                metadata=metadata or {}
            )
            
            # Store risk assessment
            self.position_risks[symbol] = trade_risk
            await self._store_risk_assessment(trade_risk)
            
            return trade_risk
            
        except Exception as e:
            self.logger.error(f"Trade risk assessment failed: {e}")
            
            # Return conservative risk assessment on error
            return TradeRisk(
                symbol=symbol,
                position_size=0.0,
                risk_level=RiskLevel.CRITICAL,
                risk_score=100.0,
                max_loss=position_value,
                confidence=0.0,
                liquidity_risk=100.0,
                concentration_risk=100.0,
                volatility_risk=100.0,
                recommendation="REJECT - Risk assessment failed",
                metadata=metadata or {}
            )
            
    async def _assess_liquidity_risk(self, symbol: str, position_value: float) -> float:
        """Assess liquidity risk for a symbol"""
        try:
            # This would integrate with real liquidity data
            # For now, use conservative estimates based on position size
            
            if position_value < 1000:
                return 10.0  # Low liquidity risk for small positions
            elif position_value < 10000:
                return 25.0  # Medium liquidity risk
            else:
                return 50.0  # High liquidity risk for large positions
                
        except Exception as e:
            self.logger.error(f"Liquidity risk assessment failed: {e}")
            return 100.0  # Maximum risk on error
            
    async def _assess_concentration_risk(self, symbol: str, position_size: float) -> float:
        """Assess position concentration risk"""
        try:
            # Check against max position size
            if position_size > self.config.max_position_size:
                return 100.0  # Critical risk - exceeds max position size
                
            # Calculate concentration risk based on position size
            concentration_ratio = position_size / self.config.max_position_size
            return min(100.0, concentration_ratio * 60.0)  # Scale to max 60% risk
            
        except Exception as e:
            self.logger.error(f"Concentration risk assessment failed: {e}")
            return 100.0
            
    async def _assess_volatility_risk(self, symbol: str, position_size: float) -> float:
        """Assess volatility-based risk"""
        try:
            # Get symbol volatility
            volatility = await self._get_symbol_volatility(symbol)
            
            # High volatility increases risk, especially for large positions
            volatility_risk = min(100.0, (volatility * 200) * (1 + position_size * 2))
            
            return volatility_risk
            
        except Exception as e:
            self.logger.error(f"Volatility risk assessment failed: {e}")
            return 75.0  # Conservative default
            
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get estimated volatility for a symbol"""
        try:
            # This would integrate with real volatility calculations
            # For now, return conservative estimates based on asset type
            
            if "BTC" in symbol or "ETH" in symbol:
                return 0.15  # 15% daily volatility for major crypto
            elif "SOL" in symbol:
                return 0.20  # 20% daily volatility for SOL
            else:
                return 0.30  # 30% daily volatility for other tokens
                
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}")
            return 0.35  # Conservative high volatility default
            
    def _get_current_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            if self.portfolio_value_history:
                return self.portfolio_value_history[-1]
            else:
                return 10000.0  # Default portfolio value
                
        except Exception as e:
            return 10000.0
            
    def _generate_recommendation(self, risk_level: RiskLevel, risk_score: float, position_size: float) -> str:
        """Generate trading recommendation based on risk assessment"""
        if risk_level == RiskLevel.CRITICAL:
            return "REJECT - Critical risk level"
        elif risk_level == RiskLevel.HIGH:
            return "CAUTION - High risk, reduce position size"
        elif risk_level == RiskLevel.MEDIUM:
            if position_size > self.config.max_position_size * 0.8:
                return "REDUCE - Position too large for risk level"
            else:
                return "PROCEED - Acceptable risk with monitoring"
        else:  # LOW risk
            return "APPROVE - Low risk trade"
            
    async def check_portfolio_risk(self) -> Dict[str, Any]:
        """Check overall portfolio risk metrics"""
        try:
            portfolio_value = self._get_current_portfolio_value()
            
            # Calculate portfolio metrics
            total_risk_exposure = sum([
                risk.position_size * portfolio_value * (risk.risk_score / 100)
                for risk in self.position_risks.values()
            ])
            
            risk_percentage = total_risk_exposure / portfolio_value if portfolio_value > 0 else 0
            
            # Check against limits
            risk_violations = []
            
            if risk_percentage > self.config.max_portfolio_risk:
                risk_violations.append(f"Portfolio risk ({risk_percentage:.1%}) exceeds maximum ({self.config.max_portfolio_risk:.1%})")
                
            # Daily loss check
            if self.daily_pnl_history:
                current_daily_loss = min(self.daily_pnl_history[-1:] + [0])
                daily_loss_pct = abs(current_daily_loss) / portfolio_value if portfolio_value > 0 else 0
                
                if daily_loss_pct > self.config.max_daily_loss:
                    risk_violations.append(f"Daily loss ({daily_loss_pct:.1%}) exceeds maximum ({self.config.max_daily_loss:.1%})")
                    
            # Drawdown check
            if len(self.portfolio_value_history) > 10:
                peak_value = max(self.portfolio_value_history[-30:])
                current_drawdown = (peak_value - portfolio_value) / peak_value
                
                if current_drawdown > self.config.max_drawdown:
                    risk_violations.append(f"Drawdown ({current_drawdown:.1%}) exceeds maximum ({self.config.max_drawdown:.1%})")
                    
            portfolio_risk = {
                "portfolio_value": portfolio_value,
                "total_risk_exposure": total_risk_exposure,
                "risk_percentage": risk_percentage,
                "active_positions": len(self.position_risks),
                "risk_violations": risk_violations,
                "overall_risk_level": "CRITICAL" if risk_violations else "NORMAL",
                "timestamp": trading_time.now().isoformat()
            }
            
            # Store portfolio risk metrics
            await self.db_manager.log_metric("portfolio_risk_percentage", risk_percentage)
            await self.db_manager.log_metric("portfolio_risk_exposure", total_risk_exposure)
            
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"Portfolio risk check failed: {e}")
            return {
                "portfolio_value": 0,
                "risk_percentage": 100,
                "overall_risk_level": "CRITICAL",
                "error": str(e)
            }
            
    async def should_halt_trading(self) -> tuple[bool, str]:
        """Determine if trading should be halted due to risk"""
        try:
            portfolio_risk = await self.check_portfolio_risk()
            
            # Critical conditions that halt trading
            if portfolio_risk["overall_risk_level"] == "CRITICAL":
                return True, "Critical portfolio risk level reached"
                
            if portfolio_risk["risk_percentage"] > self.config.max_portfolio_risk * 1.2:
                return True, f"Portfolio risk ({portfolio_risk['risk_percentage']:.1%}) critically high"
                
            # Check for rapid losses
            if len(self.daily_pnl_history) >= 3:
                recent_losses = self.daily_pnl_history[-3:]
                if all(pnl < 0 for pnl in recent_losses):
                    total_loss = sum(recent_losses)
                    portfolio_value = self._get_current_portfolio_value()
                    loss_pct = abs(total_loss) / portfolio_value if portfolio_value > 0 else 1
                    
                    if loss_pct > self.config.max_daily_loss * 1.5:
                        return True, f"Consecutive losses ({loss_pct:.1%}) exceed threshold"
                        
            return False, "Risk levels acceptable"
            
        except Exception as e:
            self.logger.error(f"Trading halt check failed: {e}")
            return True, f"Risk check failed: {e}"
            
    async def update_position_risk(self, symbol: str, current_price: float):
        """Update risk metrics for existing position"""
        if symbol in self.position_risks:
            try:
                risk = self.position_risks[symbol]
                
                # Recalculate risk based on current price
                updated_risk = await self.assess_trade_risk(
                    symbol, "UPDATE", risk.position_size, current_price,
                    metadata=risk.metadata
                )
                
                self.position_risks[symbol] = updated_risk
                
            except Exception as e:
                self.logger.error(f"Position risk update failed for {symbol}: {e}")
                
    async def _store_risk_assessment(self, risk: TradeRisk):
        """Store risk assessment in database"""
        try:
            import sqlite3
            import json
            
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_assessments
                (symbol, risk_level, risk_score, max_loss, confidence, recommendation, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                risk.symbol, risk.risk_level.value, risk.risk_score, risk.max_loss,
                risk.confidence, risk.recommendation,
                json.dumps(risk.metadata), trading_time.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store risk assessment: {e}")
            
    async def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            portfolio_risk = await self.check_portfolio_risk()
            
            return {
                "timestamp": trading_time.now().isoformat(),
                "portfolio_risk": portfolio_risk,
                "position_risks": {
                    symbol: {
                        "risk_level": risk.risk_level.value,
                        "risk_score": risk.risk_score,
                        "recommendation": risk.recommendation,
                        "confidence": risk.confidence
                    }
                    for symbol, risk in self.position_risks.items()
                },
                "risk_config": {
                    "max_position_size": self.config.max_position_size,
                    "max_portfolio_risk": self.config.max_portfolio_risk,
                    "max_daily_loss": self.config.max_daily_loss,
                    "max_drawdown": self.config.max_drawdown
                }
            }
            
        except Exception as e:
            self.logger.error(f"Risk report generation failed: {e}")
            return {"error": str(e)}
    
    async def _check_multi_strategy_risk(self, strategy_name: str, position_value: float) -> float:
        """Check cumulative risk across all strategies"""
        try:
            import sqlite3
            
            # Get current portfolio value
            portfolio_value = self._get_current_portfolio_value()
            if portfolio_value <= 0:
                return 1.0  # Max risk if no portfolio value
                
            # Calculate current exposure from all strategies
            conn = sqlite3.connect(self.db_manager.db_path, timeout=5.0)
            cursor = conn.cursor()
            
            # Get current open positions across all strategies
            cursor.execute("""
                SELECT SUM(quantity * current_price) as total_exposure
                FROM positions 
                WHERE status = 'open'
            """)
            
            result = cursor.fetchone()
            current_exposure = result[0] if result[0] else 0.0
            
            # Add the new position exposure
            total_exposure = current_exposure + position_value
            
            # Calculate risk as percentage of portfolio
            cumulative_risk = total_exposure / portfolio_value
            
            conn.close()
            
            self.logger.debug(f"Multi-strategy risk check: {cumulative_risk:.2%} "
                            f"(current: {current_exposure}, new: {position_value}, portfolio: {portfolio_value})")
            
            return cumulative_risk
            
        except Exception as e:
            self.logger.error(f"Multi-strategy risk check failed: {e}")
            # Return high risk on error to be safe
            return 1.0
            
    async def shutdown(self):
        """Shutdown risk engine"""
        try:
            self.logger.info("Risk engine shutdown")
            
        except Exception as e:
            self.logger.error(f"Risk engine shutdown error: {e}")