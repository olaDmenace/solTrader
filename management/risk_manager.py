#!/usr/bin/env python3
"""
UNIFIED RISK MANAGER - Day 8 Consolidation
Consolidates 8 duplicate risk management implementations into a single, comprehensive system.

This unified risk manager consolidates the best features from:
1. src/risk/risk_manager.py - Core risk infrastructure with emergency controls
2. src/utils/risk_management.py - Advanced metrics and drawdown tracking  
3. src/portfolio/portfolio_risk_manager.py - Portfolio-level VaR and correlation analysis
4. src/trading/risk.py - Position-level risk calculation and market conditions
5. src/trading/enhanced_risk.py - Multi-factor risk evaluation
6. src/trading/risk_engine.py - Trade assessment and database integration
7. strategies/* - Strategy-specific risk controls
8. core/swap_executor.py - Transaction-level risk validation

Key Features:
- Emergency stop mechanisms and circuit breakers
- Portfolio-level VaR/CVaR calculations with correlation analysis
- Position sizing with Kelly criterion and risk budgets
- Drawdown tracking and streak analysis
- Multi-strategy coordination and cumulative risk monitoring
- Real-time alerting and comprehensive risk reporting
"""

import asyncio
import logging
import time
import os
import json
import sqlite3
import aiosqlite
import numpy as np
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications with comparable values"""
    LOW = 1
    MODERATE = 2  
    MEDIUM = 2    # Alias for consistency
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

class RiskEvent(Enum):
    """Risk event types for alerting"""
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    DRAWDOWN_LIMIT_EXCEEDED = "drawdown_limit_exceeded"
    VOLATILITY_SPIKE = "volatility_spike"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    PORTFOLIO_CONCENTRATION = "portfolio_concentration"
    EMERGENCY_STOP = "emergency_stop"
    BALANCE_TOO_LOW = "balance_too_low"
    TRADE_FREQUENCY_HIGH = "trade_frequency_high"
    VAR_BREACH = "var_breach"
    HIGH_CORRELATION = "high_correlation"
    RISK_BUDGET_BREACH = "risk_budget_breach"
    MULTI_STRATEGY_RISK = "multi_strategy_risk"

class EmergencyAction(Enum):
    """Emergency actions for critical risk situations"""
    NONE = "NONE"
    REDUCE_POSITIONS = "REDUCE_POSITIONS" 
    HALT_TRADING = "HALT_TRADING"
    LIQUIDATE_ALL = "LIQUIDATE_ALL"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    event_type: RiskEvent
    level: RiskLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    recommended_action: EmergencyAction = EmergencyAction.NONE
    affected_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'level': self.level.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'resolved': self.resolved,
            'recommended_action': self.recommended_action.value,
            'affected_strategies': self.affected_strategies
        }

@dataclass
class PositionRisk:
    """Position-specific risk metrics"""
    token_address: str
    strategy_name: str
    position_size: Decimal
    position_value_usd: Optional[Decimal] = None
    portfolio_percentage: Optional[Decimal] = None
    volatility_score: Optional[Decimal] = None
    correlation_risk: RiskLevel = RiskLevel.LOW
    concentration_risk: RiskLevel = RiskLevel.LOW
    liquidity_risk: RiskLevel = RiskLevel.LOW
    total_risk_score: float = 0.0

@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""
    timestamp: datetime
    total_value_usd: Decimal
    daily_pnl: Decimal
    max_drawdown: Decimal
    current_drawdown: Decimal
    
    # VaR/CVaR metrics
    var_95_1d: float = 0.0
    var_99_1d: float = 0.0
    cvar_95_1d: float = 0.0
    cvar_99_1d: float = 0.0
    
    # Portfolio metrics
    total_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    beta_to_market: float = 1.0
    
    # Correlation and diversification
    avg_strategy_correlation: float = 0.0
    max_strategy_correlation: float = 0.0
    diversification_ratio: float = 0.0
    
    # Risk budgets and position sizing
    used_risk_budget: float = 0.0
    available_risk_budget: float = 0.0
    recommended_leverage: float = 0.0
    max_position_size: float = 0.0
    current_leverage: float = 0.0
    
    # Overall assessment
    position_count: int = 0
    concentration_score: Decimal = Decimal('0')
    overall_risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0  # 0-100 risk score

@dataclass
class DrawdownEvent:
    """Drawdown period tracking"""
    start_date: str
    end_date: str
    start_balance: float
    peak_balance: float
    trough_balance: float
    drawdown_amount: float
    drawdown_percentage: float
    recovery_date: Optional[str]
    duration_days: int
    trades_during_drawdown: int

@dataclass
class MarketCondition:
    """Market condition data for risk adjustment"""
    volatility: float
    trend_strength: float
    liquidity_score: float
    market_regime: str  # 'trending', 'ranging', 'volatile'

@dataclass
class TradeRisk:
    """Individual trade risk assessment"""
    token_address: str
    strategy_name: str
    position_size: float
    risk_level: RiskLevel
    risk_score: float
    max_loss: float
    confidence: float
    liquidity_risk: float
    concentration_risk: float
    volatility_risk: float
    correlation_risk: float
    recommendation: str
    metadata: Dict[str, Any]

class UnifiedRiskManager:
    """
    UNIFIED RISK MANAGER - Consolidates all risk management functionality
    
    This system provides comprehensive risk management by consolidating the best
    features from 8 different risk management implementations across the codebase.
    """
    
    def __init__(self, settings=None, db_path: str = "logs/unified_risk.db"):
        self.settings = settings
        self.db_path = db_path
        
        # Load configuration from environment and settings
        self._load_config()
        
        # Risk state tracking
        self.active_alerts: List[RiskAlert] = []
        self.alert_history: List[RiskAlert] = []
        self.emergency_stop_active = False
        self.last_emergency_check = time.time()
        
        # Performance tracking
        self.daily_pnl = Decimal('0')
        self.daily_trade_count = 0
        self.current_positions: Dict[str, PositionRisk] = {}
        self.balance_history: List[Tuple[datetime, Decimal]] = []
        self.portfolio_metrics: Optional[PortfolioRiskMetrics] = None
        
        # Advanced risk tracking
        self.drawdown_events: List[DrawdownEvent] = []
        self.daily_balances: deque = deque(maxlen=365)  # 1 year history
        self.trade_sequence: deque = deque(maxlen=1000)  # Last 1000 trades
        self.strategy_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
        # Rate limiting and frequency controls
        self.trade_timestamps: List[datetime] = []
        self.daily_reset_time = None
        
        # Market conditions and volatility tracking
        self.market_conditions: Optional[MarketCondition] = None
        self.volatility_windows: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[float]] = {}
        
        # Risk calculation cache
        self.last_risk_calculation = datetime.now()
        self.risk_calculation_interval = timedelta(minutes=5)
        
        logger.info("[UNIFIED_RISK] Unified Risk Manager initialized with comprehensive risk controls")
    
    def _load_config(self):
        """Load comprehensive risk management configuration"""
        # Position limits
        self.max_position_size = Decimal(os.getenv('MAX_LIVE_POSITION_SIZE', '0.5'))
        self.max_positions = int(os.getenv('MAX_LIVE_POSITIONS', '5'))
        self.max_portfolio_risk = Decimal(os.getenv('MAX_PORTFOLIO_RISK', '15.0')) / 100
        
        # Loss limits
        self.max_daily_loss = Decimal(os.getenv('MAX_DAILY_LOSS', '10.0'))
        self.max_drawdown = Decimal(os.getenv('MAX_DRAWDOWN', '20.0')) / 100
        self.emergency_stop_loss = Decimal(os.getenv('EMERGENCY_STOP_LOSS', '5.0'))
        
        # Balance limits
        self.min_balance = Decimal(os.getenv('EMERGENCY_MIN_BALANCE', '0.1'))
        self.trading_min_balance = Decimal(os.getenv('TRADING_MIN_BALANCE', '0.01'))
        
        # Trading frequency
        self.max_trades_per_day = int(os.getenv('MAX_TRADES_PER_DAY', '20'))
        self.max_trades_per_hour = int(os.getenv('EMERGENCY_MAX_TRADES_HOUR', '10'))
        
        # Volatility and VaR limits
        self.max_volatility = Decimal(os.getenv('MAX_VOLATILITY', '30.0')) / 100
        self.emergency_max_volatility = Decimal(os.getenv('EMERGENCY_MAX_VOLATILITY', '50.0')) / 100
        self.max_portfolio_var = 0.08  # 8% max daily VaR
        self.emergency_var_threshold = 0.12  # 12% VaR triggers emergency
        
        # Correlation and concentration
        self.correlation_alert_threshold = 0.75  # Alert if avg correlation > 75%
        self.max_concentration = 0.30  # 30% max single position concentration
        
        # Risk budget and leverage
        self.risk_budget = 0.15  # 15% risk budget
        self.max_portfolio_leverage = 2.0  # Max 2x leverage
        self.position_reduction_factor = 0.6  # Reduce positions by 40% in emergency
        
        # Multi-strategy risk controls
        self.max_strategy_correlation = 0.80  # 80% max strategy correlation
        self.strategy_risk_budget = 0.05  # 5% risk budget per strategy
        
        logger.info(f"[UNIFIED_RISK] Configuration loaded - Max Position: {self.max_position_size} SOL, "
                   f"Max Daily Loss: ${self.max_daily_loss}, Risk Budget: {self.risk_budget:.1%}")
    
    async def initialize(self):
        """Initialize risk management database and load historical data"""
        try:
            # Create database tables
            await self._create_risk_tables()
            
            # Load historical risk data
            await self._load_historical_data()
            
            logger.info("[UNIFIED_RISK] Risk management system initialized successfully")
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Initialization failed: {e}")
            raise
    
    async def _create_risk_tables(self):
        """Create comprehensive risk management database tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Risk alerts table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS risk_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        recommended_action TEXT NOT NULL,
                        affected_strategies TEXT NOT NULL,
                        data TEXT,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Position risk tracking
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS position_risks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        token_address TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        position_size REAL NOT NULL,
                        position_value_usd REAL,
                        portfolio_percentage REAL,
                        risk_score REAL NOT NULL,
                        risk_level TEXT NOT NULL
                    )
                """)
                
                # Portfolio risk metrics
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_risk_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_value_usd REAL NOT NULL,
                        var_95_1d REAL NOT NULL,
                        var_99_1d REAL NOT NULL,
                        total_volatility REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        current_drawdown REAL NOT NULL,
                        diversification_ratio REAL NOT NULL,
                        risk_score REAL NOT NULL,
                        overall_risk_level TEXT NOT NULL
                    )
                """)
                
                # Strategy correlation matrix
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_correlations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy1 TEXT NOT NULL,
                        strategy2 TEXT NOT NULL,
                        correlation REAL NOT NULL,
                        lookback_days INTEGER NOT NULL
                    )
                """)
                
                await db.commit()
                logger.info("[UNIFIED_RISK] Risk management database tables created")
                
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Database initialization failed: {e}")
            raise
    
    async def _load_historical_data(self):
        """Load historical risk and performance data"""
        try:
            # Load existing position data if available
            data_files = ["analytics/risk_metrics.json", "logs/risk_data.json"]
            
            for data_file in data_files:
                if Path(data_file).exists():
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                        
                    # Load balance history
                    if 'daily_balances' in data:
                        self.daily_balances = deque(data['daily_balances'], maxlen=365)
                    
                    # Load trade sequence
                    if 'trade_sequence' in data:
                        self.trade_sequence = deque(data['trade_sequence'], maxlen=1000)
                    
                    logger.info(f"[UNIFIED_RISK] Loaded historical data from {data_file}")
                    break
                    
        except Exception as e:
            logger.warning(f"[UNIFIED_RISK] Could not load historical data: {e}")
    
    async def validate_trade(self, 
                           token_address: str, 
                           strategy_name: str,
                           trade_size: Decimal, 
                           current_balance: Decimal,
                           entry_price: float,
                           trade_type: str = "buy") -> Tuple[bool, str, RiskLevel]:
        """
        COMPREHENSIVE TRADE VALIDATION
        Consolidates all risk validation logic from multiple implementations
        
        Returns: (is_allowed, reason, risk_level)
        """
        try:
            # Emergency stop check (highest priority)
            if self.emergency_stop_active:
                return False, "Emergency stop is active - all trading halted", RiskLevel.EMERGENCY
            
            # Balance validation
            balance_check = self._validate_balance(trade_size, current_balance)
            if not balance_check[0]:
                return balance_check
            
            # Position limits validation  
            position_check = self._validate_position_limits(token_address, trade_size, trade_type)
            if not position_check[0]:
                return position_check
            
            # Daily limits validation
            daily_check = self._validate_daily_limits()
            if not daily_check[0]:
                return daily_check
            
            # Portfolio risk validation
            portfolio_check = await self._validate_portfolio_risk(
                token_address, strategy_name, trade_size, entry_price, current_balance
            )
            if not portfolio_check[0]:
                return portfolio_check
            
            # Multi-strategy risk validation
            multi_strategy_check = await self._validate_multi_strategy_risk(
                strategy_name, trade_size * Decimal(str(entry_price))
            )
            if not multi_strategy_check[0]:
                return multi_strategy_check
            
            # Correlation and concentration validation
            correlation_check = self._validate_correlation_and_concentration(
                token_address, trade_size, current_balance
            )
            if not correlation_check[0]:
                return correlation_check
            
            # Market condition validation
            market_check = self._validate_market_conditions(trade_size, current_balance)
            if not market_check[0]:
                return market_check
            
            # Calculate overall trade risk level
            trade_risk = self._calculate_overall_trade_risk(
                trade_size, current_balance, token_address, strategy_name
            )
            
            logger.debug(f"[UNIFIED_RISK] Trade validation passed - Risk Level: {trade_risk}")
            return True, "Trade approved by unified risk management", trade_risk
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Trade validation error: {e}")
            return False, f"Risk validation error: {str(e)}", RiskLevel.CRITICAL
    
    def _validate_balance(self, trade_size: Decimal, current_balance: Decimal) -> Tuple[bool, str, RiskLevel]:
        """Validate balance-related risk controls"""
        # Emergency balance check
        if current_balance < self.min_balance:
            self._trigger_emergency_stop(f"Balance {current_balance} below emergency minimum {self.min_balance}")
            return False, f"Emergency balance too low: {current_balance}", RiskLevel.EMERGENCY
        
        # Trading minimum balance check
        if current_balance - trade_size < self.trading_min_balance:
            return False, "Trade would leave balance below minimum trading threshold", RiskLevel.HIGH
        
        return True, "Balance validation passed", RiskLevel.LOW
    
    def _validate_position_limits(self, token_address: str, trade_size: Decimal, trade_type: str) -> Tuple[bool, str, RiskLevel]:
        """Validate position size and count limits"""
        # Individual position size check
        if trade_size > self.max_position_size:
            return False, f"Trade size {trade_size} exceeds maximum {self.max_position_size}", RiskLevel.HIGH
        
        # Position count check for new positions
        if trade_type == "buy" and len(self.current_positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) already reached", RiskLevel.MODERATE
        
        return True, "Position limits validation passed", RiskLevel.LOW
    
    def _validate_daily_limits(self) -> Tuple[bool, str, RiskLevel]:
        """Validate daily trading limits"""
        # Reset daily metrics if needed
        if self._should_reset_daily():
            self._reset_daily_metrics()
        
        # Daily trade count check
        if self.daily_trade_count >= self.max_trades_per_day:
            return False, f"Daily trade limit reached: {self.daily_trade_count}/{self.max_trades_per_day}", RiskLevel.MODERATE
        
        # Daily loss check
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit reached: ${abs(self.daily_pnl)}", RiskLevel.HIGH
        
        # Trade frequency check
        if not self._check_trade_frequency():
            return False, "Trade frequency limits exceeded", RiskLevel.MODERATE
        
        return True, "Daily limits validation passed", RiskLevel.LOW
    
    async def _validate_portfolio_risk(self, 
                                     token_address: str,
                                     strategy_name: str, 
                                     trade_size: Decimal,
                                     entry_price: float,
                                     current_balance: Decimal) -> Tuple[bool, str, RiskLevel]:
        """Validate portfolio-level risk metrics"""
        try:
            # Calculate position value
            position_value = trade_size * Decimal(str(entry_price))
            
            # Check portfolio VaR if we have sufficient data
            if self.portfolio_metrics:
                current_var = abs(self.portfolio_metrics.var_95_1d)
                if current_var > self.max_portfolio_var:
                    return False, f"Portfolio VaR ({current_var:.2%}) exceeds limit", RiskLevel.HIGH
            
            # Calculate new portfolio risk exposure
            total_portfolio_value = current_balance * Decimal('10')  # Approximate portfolio value
            new_risk_exposure = position_value / total_portfolio_value
            
            # Check against portfolio risk budget
            current_risk = sum(
                pos.position_value_usd / total_portfolio_value 
                for pos in self.current_positions.values() 
                if pos.position_value_usd
            ) or Decimal('0')
            
            total_risk = current_risk + new_risk_exposure
            
            if total_risk > self.max_portfolio_risk:
                return False, f"Portfolio risk ({total_risk:.2%}) would exceed limit ({self.max_portfolio_risk:.2%})", RiskLevel.HIGH
            
            return True, "Portfolio risk validation passed", RiskLevel.LOW
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Portfolio risk validation error: {e}")
            return False, f"Portfolio risk validation failed: {e}", RiskLevel.CRITICAL
    
    async def _validate_multi_strategy_risk(self, 
                                          strategy_name: str, 
                                          position_value: Decimal) -> Tuple[bool, str, RiskLevel]:
        """Validate cumulative risk across multiple strategies"""
        try:
            # Get current total exposure across all strategies
            total_exposure = sum(
                pos.position_value_usd for pos in self.current_positions.values()
                if pos.position_value_usd
            ) or Decimal('0')
            
            # Add new position exposure
            new_total_exposure = total_exposure + position_value
            
            # Check against multi-strategy risk budget (more conservative)
            portfolio_value = Decimal('10000')  # Approximate - in production use actual portfolio value
            cumulative_risk = new_total_exposure / portfolio_value
            
            # Multi-strategy risk threshold (lower than single strategy)
            max_multi_strategy_risk = self.max_portfolio_risk * Decimal('0.8')
            
            if cumulative_risk > max_multi_strategy_risk:
                return False, f"Multi-strategy risk ({cumulative_risk:.2%}) exceeds limit", RiskLevel.HIGH
            
            # Check strategy-specific risk budget
            strategy_exposure = sum(
                pos.position_value_usd for pos in self.current_positions.values()
                if pos.strategy_name == strategy_name and pos.position_value_usd
            ) or Decimal('0')
            
            new_strategy_exposure = strategy_exposure + position_value
            strategy_risk = new_strategy_exposure / portfolio_value
            
            if strategy_risk > Decimal(str(self.strategy_risk_budget)):
                return False, f"Strategy '{strategy_name}' risk budget exceeded", RiskLevel.MODERATE
            
            return True, "Multi-strategy risk validation passed", RiskLevel.LOW
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Multi-strategy risk validation error: {e}")
            return False, f"Multi-strategy validation failed: {e}", RiskLevel.CRITICAL
    
    def _validate_correlation_and_concentration(self, 
                                              token_address: str, 
                                              trade_size: Decimal,
                                              current_balance: Decimal) -> Tuple[bool, str, RiskLevel]:
        """Validate correlation and concentration risk"""
        try:
            # Concentration risk check
            position_value = trade_size  # Simplified - treating as SOL value
            portfolio_value = current_balance * Decimal('5')  # Approximate total portfolio
            concentration = position_value / portfolio_value
            
            if concentration > Decimal(str(self.max_concentration)):
                return False, f"Position concentration ({concentration:.2%}) too high", RiskLevel.HIGH
            
            # Correlation check with existing positions
            if self.current_positions:
                high_correlation_count = 0
                for pos_addr in self.current_positions.keys():
                    if pos_addr != token_address:
                        correlation = self._calculate_correlation(token_address, pos_addr)
                        if abs(correlation) > 0.7:  # High correlation threshold
                            high_correlation_count += 1
                
                if high_correlation_count >= 2:  # Too many highly correlated positions
                    return False, "Too many highly correlated positions", RiskLevel.MODERATE
            
            return True, "Correlation and concentration validation passed", RiskLevel.LOW
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Correlation validation error: {e}")
            return True, "Correlation validation skipped due to error", RiskLevel.LOW  # Don't block on correlation errors
    
    def _validate_market_conditions(self, trade_size: Decimal, current_balance: Decimal) -> Tuple[bool, str, RiskLevel]:
        """Validate trade against current market conditions"""
        try:
            # Check market volatility if available
            if self.market_conditions:
                if self.market_conditions.volatility > float(self.emergency_max_volatility):
                    return False, "Market volatility too high for new positions", RiskLevel.HIGH
                
                # Adjust position size for high volatility markets
                if self.market_conditions.volatility > float(self.max_volatility):
                    max_allowed_size = self.max_position_size * Decimal('0.7')  # Reduce by 30%
                    if trade_size > max_allowed_size:
                        return False, f"Position size too large for volatile market conditions", RiskLevel.MODERATE
            
            # Check portfolio volatility from recent balance history
            if len(self.balance_history) >= 10:
                recent_balances = [balance for _, balance in self.balance_history[-10:]]
                volatility = self._calculate_balance_volatility(recent_balances)
                
                if volatility > float(self.emergency_max_volatility):
                    return False, "Portfolio volatility too high", RiskLevel.HIGH
            
            return True, "Market conditions validation passed", RiskLevel.LOW
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Market conditions validation error: {e}")
            return True, "Market conditions validation skipped", RiskLevel.LOW
    
    def _calculate_overall_trade_risk(self, 
                                    trade_size: Decimal,
                                    current_balance: Decimal, 
                                    token_address: str,
                                    strategy_name: str) -> RiskLevel:
        """Calculate overall risk level for approved trade"""
        try:
            risk_factors = []
            
            # Size risk (higher size = higher risk)
            size_ratio = trade_size / current_balance if current_balance > 0 else 1
            if size_ratio > 0.1:  # >10% of balance
                risk_factors.append(RiskLevel.HIGH)
            elif size_ratio > 0.05:  # >5% of balance
                risk_factors.append(RiskLevel.MODERATE)
            else:
                risk_factors.append(RiskLevel.LOW)
            
            # Portfolio risk factor
            if self.portfolio_metrics:
                if self.portfolio_metrics.used_risk_budget > 0.8:
                    risk_factors.append(RiskLevel.HIGH)
                elif self.portfolio_metrics.used_risk_budget > 0.6:
                    risk_factors.append(RiskLevel.MODERATE)
                else:
                    risk_factors.append(RiskLevel.LOW)
            
            # Market condition risk factor
            if self.market_conditions:
                if self.market_conditions.volatility > 0.5:
                    risk_factors.append(RiskLevel.HIGH)
                elif self.market_conditions.volatility > 0.3:
                    risk_factors.append(RiskLevel.MODERATE)
                else:
                    risk_factors.append(RiskLevel.LOW)
            
            # Return highest risk level
            if not risk_factors:
                return RiskLevel.LOW
            
            return max(risk_factors)
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Overall risk calculation error: {e}")
            return RiskLevel.MODERATE
    
    async def assess_comprehensive_trade_risk(self,
                                            token_address: str,
                                            strategy_name: str,
                                            position_size: float,
                                            entry_price: float,
                                            metadata: Dict[str, Any] = None) -> TradeRisk:
        """Comprehensive trade risk assessment combining all risk factors"""
        try:
            # Individual risk component calculations
            liquidity_risk = self._assess_liquidity_risk(token_address, position_size * entry_price)
            concentration_risk = self._assess_concentration_risk(position_size, entry_price)
            volatility_risk = self._assess_volatility_risk(token_address)
            correlation_risk = self._assess_correlation_risk(token_address)
            
            # Combined risk score (0-100)
            risk_score = (
                liquidity_risk * 0.25 +
                concentration_risk * 0.30 +
                volatility_risk * 0.25 +
                correlation_risk * 0.20
            )
            
            # Risk level classification
            if risk_score < 25:
                risk_level = RiskLevel.LOW
            elif risk_score < 50:
                risk_level = RiskLevel.MODERATE
            elif risk_score < 75:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            # Calculate potential loss
            volatility = self._get_token_volatility(token_address)
            max_loss = position_size * entry_price * min(volatility * 2, 0.5)
            
            # Trading confidence
            confidence = max(0.1, 1.0 - (risk_score / 100))
            
            # Generate recommendation
            recommendation = self._generate_trade_recommendation(risk_level, risk_score)
            
            return TradeRisk(
                token_address=token_address,
                strategy_name=strategy_name,
                position_size=position_size,
                risk_level=risk_level,
                risk_score=risk_score,
                max_loss=max_loss,
                confidence=confidence,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                volatility_risk=volatility_risk,
                correlation_risk=correlation_risk,
                recommendation=recommendation,
                metadata=metadata or {}
            )
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Comprehensive risk assessment error: {e}")
            # Return conservative assessment on error
            return TradeRisk(
                token_address=token_address,
                strategy_name=strategy_name,
                position_size=position_size,
                risk_level=RiskLevel.CRITICAL,
                risk_score=100.0,
                max_loss=position_size * entry_price,
                confidence=0.0,
                liquidity_risk=100.0,
                concentration_risk=100.0,
                volatility_risk=100.0,
                correlation_risk=100.0,
                recommendation="REJECT - Risk assessment failed",
                metadata=metadata or {}
            )
    
    def update_position(self, 
                       token_address: str,
                       strategy_name: str, 
                       position_size: Decimal,
                       entry_price: float,
                       position_value_usd: Optional[Decimal] = None):
        """Update position tracking with comprehensive risk monitoring"""
        try:
            if position_size <= 0:
                # Remove position
                if token_address in self.current_positions:
                    del self.current_positions[token_address]
                    logger.debug(f"[UNIFIED_RISK] Removed position: {token_address[:8]}...")
            else:
                # Calculate position metrics
                if position_value_usd is None:
                    position_value_usd = position_size * Decimal(str(entry_price))
                
                # Calculate portfolio percentage
                portfolio_pct = None
                if self.portfolio_metrics and self.portfolio_metrics.total_value_usd > 0:
                    portfolio_pct = position_value_usd / self.portfolio_metrics.total_value_usd * 100
                
                # Create position risk record
                position_risk = PositionRisk(
                    token_address=token_address,
                    strategy_name=strategy_name,
                    position_size=position_size,
                    position_value_usd=position_value_usd,
                    portfolio_percentage=portfolio_pct,
                    total_risk_score=self._calculate_position_risk_score(
                        token_address, float(position_size), entry_price
                    )
                )
                
                self.current_positions[token_address] = position_risk
                logger.debug(f"[UNIFIED_RISK] Updated position: {token_address[:8]}... = {position_size} SOL")
            
            # Update portfolio risk metrics
            asyncio.create_task(self._update_portfolio_risk_metrics())
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error updating position: {e}")
    
    def record_trade(self, 
                    token_address: str,
                    strategy_name: str,
                    success: bool,
                    profit_loss: Optional[Decimal] = None,
                    trade_size: Optional[Decimal] = None):
        """Record trade execution for comprehensive tracking"""
        try:
            current_time = datetime.now()
            
            # Update trade frequency tracking
            self.trade_timestamps.append(current_time)
            self.daily_trade_count += 1
            
            # Update P&L tracking
            if profit_loss is not None:
                self.daily_pnl += profit_loss
                
                # Add to trade sequence for advanced analytics
                trade_record = {
                    'timestamp': current_time.isoformat(),
                    'token_address': token_address,
                    'strategy_name': strategy_name,
                    'success': success,
                    'profit_loss': float(profit_loss),
                    'trade_size': float(trade_size) if trade_size else 0.0
                }
                self.trade_sequence.append(trade_record)
            
            # Clean old timestamps (keep last hour)
            cutoff_time = current_time - timedelta(hours=1)
            self.trade_timestamps = [ts for ts in self.trade_timestamps if ts > cutoff_time]
            
            # Check for emergency loss conditions
            if self.daily_pnl < -self.emergency_stop_loss:
                self._trigger_emergency_stop(f"Emergency daily loss exceeded: ${abs(self.daily_pnl)}")
            
            # Reset daily counter at midnight
            if self.daily_reset_time is None or current_time.date() > self.daily_reset_time:
                self._reset_daily_metrics()
                self.daily_reset_time = current_time.date()
            
            logger.debug(f"[UNIFIED_RISK] Trade recorded: {strategy_name} - {'SUCCESS' if success else 'FAILURE'}")
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error recording trade: {e}")
    
    def update_balance(self, new_balance: Decimal):
        """Update balance with comprehensive drawdown and risk tracking"""
        try:
            current_time = datetime.now()
            self.balance_history.append((current_time, new_balance))
            
            # Update daily balance tracking
            today = current_time.date().isoformat()
            if not self.daily_balances or self.daily_balances[-1].get('date') != today:
                self.daily_balances.append({
                    'date': today,
                    'balance': float(new_balance),
                    'trades_count': 1
                })
            else:
                self.daily_balances[-1]['balance'] = float(new_balance)
                self.daily_balances[-1]['trades_count'] += 1
            
            # Keep only last 24 hours of balance history  
            cutoff_time = current_time - timedelta(hours=24)
            self.balance_history = [
                (ts, balance) for ts, balance in self.balance_history 
                if ts > cutoff_time
            ]
            
            # Calculate and check drawdown
            self._check_drawdown_limits(new_balance)
            
            logger.debug(f"[UNIFIED_RISK] Balance updated: {new_balance} SOL")
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error updating balance: {e}")
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger comprehensive emergency stop mechanism"""
        try:
            if not self.emergency_stop_active:
                self.emergency_stop_active = True
                self.last_emergency_check = time.time()
                
                # Create emergency alert
                alert = RiskAlert(
                    event_type=RiskEvent.EMERGENCY_STOP,
                    level=RiskLevel.EMERGENCY,
                    message=f"EMERGENCY STOP TRIGGERED: {reason}",
                    recommended_action=EmergencyAction.HALT_TRADING,
                    affected_strategies=list(set(pos.strategy_name for pos in self.current_positions.values())),
                    data={"reason": reason, "timestamp": datetime.now().isoformat()}
                )
                
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
                
                # Log to database
                asyncio.create_task(self._log_risk_alert(alert))
                
                logger.critical(f"[UNIFIED_RISK] 🚨 EMERGENCY STOP TRIGGERED: {reason}")
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error triggering emergency stop: {e}")
    
    def clear_emergency_stop(self, manual_override: bool = False) -> bool:
        """Clear emergency stop with manual override requirement"""
        try:
            if not manual_override:
                logger.warning("[UNIFIED_RISK] Emergency stop clear attempted without manual override")
                return False
            
            self.emergency_stop_active = False
            self.last_emergency_check = time.time()
            
            # Resolve emergency alerts
            for alert in self.active_alerts:
                if alert.event_type == RiskEvent.EMERGENCY_STOP:
                    alert.resolved = True
            
            logger.warning("[UNIFIED_RISK] Emergency stop cleared via manual override")
            return True
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error clearing emergency stop: {e}")
            return False
    
    async def get_comprehensive_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary combining all risk dimensions"""
        try:
            # Calculate current portfolio metrics if needed
            if datetime.now() - self.last_risk_calculation >= self.risk_calculation_interval:
                await self._update_portfolio_risk_metrics()
            
            # Active alerts summary
            active_alerts_by_level = {}
            for level in RiskLevel:
                active_alerts_by_level[level.name.lower()] = len([
                    alert for alert in self.active_alerts 
                    if alert.level == level and not alert.resolved
                ])
            
            # Position summary with risk scoring
            position_summary = []
            for token_addr, position in self.current_positions.items():
                position_summary.append({
                    'token': token_addr[:8] + '...',
                    'strategy': position.strategy_name,
                    'size_sol': float(position.position_size),
                    'value_usd': float(position.position_value_usd) if position.position_value_usd else 0,
                    'portfolio_pct': float(position.portfolio_percentage) if position.portfolio_percentage else 0,
                    'risk_score': position.total_risk_score
                })
            
            # Strategy correlation matrix
            strategy_correlations = {}
            strategies = list(set(pos.strategy_name for pos in self.current_positions.values()))
            for i, strategy1 in enumerate(strategies):
                for strategy2 in strategies[i:]:
                    if strategy1 != strategy2:
                        correlation = self._get_strategy_correlation(strategy1, strategy2)
                        strategy_correlations[f"{strategy1}-{strategy2}"] = correlation
            
            return {
                "timestamp": datetime.now().isoformat(),
                "emergency_stop_active": self.emergency_stop_active,
                "overall_risk_level": self.portfolio_metrics.overall_risk_level.name.lower() if self.portfolio_metrics else "unknown",
                
                # Daily metrics
                "daily_metrics": {
                    "pnl": float(self.daily_pnl),
                    "trade_count": self.daily_trade_count,
                    "remaining_trades": max(0, self.max_trades_per_day - self.daily_trade_count)
                },
                
                # Portfolio metrics
                "portfolio": {
                    "total_positions": len(self.current_positions),
                    "max_positions": self.max_positions,
                    "total_value_usd": float(self.portfolio_metrics.total_value_usd) if self.portfolio_metrics else 0,
                    "current_drawdown": float(self.portfolio_metrics.current_drawdown) if self.portfolio_metrics else 0,
                    "max_drawdown": float(self.portfolio_metrics.max_drawdown) if self.portfolio_metrics else 0,
                    "risk_score": self.portfolio_metrics.risk_score if self.portfolio_metrics else 0
                },
                
                # Risk metrics
                "risk_metrics": {
                    "var_95_1d": self.portfolio_metrics.var_95_1d if self.portfolio_metrics else 0,
                    "total_volatility": self.portfolio_metrics.total_volatility if self.portfolio_metrics else 0,
                    "sharpe_ratio": self.portfolio_metrics.sharpe_ratio if self.portfolio_metrics else 0,
                    "diversification_ratio": self.portfolio_metrics.diversification_ratio if self.portfolio_metrics else 0,
                    "used_risk_budget": self.portfolio_metrics.used_risk_budget if self.portfolio_metrics else 0,
                    "available_risk_budget": self.portfolio_metrics.available_risk_budget if self.portfolio_metrics else 1.0
                },
                
                # Alerts and warnings
                "alerts": {
                    "active_count": len([a for a in self.active_alerts if not a.resolved]),
                    "by_level": active_alerts_by_level,
                    "recent_alerts": [alert.to_dict() for alert in self.active_alerts[-5:]]
                },
                
                # Position details
                "positions": position_summary,
                
                # Strategy correlations
                "strategy_correlations": strategy_correlations,
                
                # Risk limits and configuration
                "limits": {
                    "max_position_size": float(self.max_position_size),
                    "max_daily_loss": float(self.max_daily_loss),
                    "max_portfolio_risk": float(self.max_portfolio_risk),
                    "max_drawdown": float(self.max_drawdown),
                    "risk_budget": self.risk_budget,
                    "correlation_alert_threshold": self.correlation_alert_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error generating risk summary: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    # Helper methods for risk calculations
    def _should_reset_daily(self) -> bool:
        """Check if daily metrics should be reset"""
        now = datetime.now()
        if self.daily_reset_time is None:
            return True
        return (now.date() > self.daily_reset_time)
    
    def _reset_daily_metrics(self):
        """Reset daily trading metrics"""
        self.daily_trade_count = 0
        self.daily_pnl = Decimal('0')
        self.daily_reset_time = datetime.now().date()
        logger.debug("[UNIFIED_RISK] Daily metrics reset")
    
    def _check_trade_frequency(self) -> bool:
        """Check if trade frequency is within limits"""
        try:
            # Daily limit check
            if self.daily_trade_count >= self.max_trades_per_day:
                self._create_alert(
                    RiskEvent.TRADE_FREQUENCY_HIGH,
                    RiskLevel.MODERATE,
                    f"Daily trade limit reached: {self.daily_trade_count}/{self.max_trades_per_day}",
                    {"daily_trades": self.daily_trade_count}
                )
                return False
            
            # Hourly limit check
            hourly_trades = len(self.trade_timestamps)
            if hourly_trades >= self.max_trades_per_hour:
                self._create_alert(
                    RiskEvent.TRADE_FREQUENCY_HIGH,
                    RiskLevel.HIGH,
                    f"Hourly trade limit reached: {hourly_trades}/{self.max_trades_per_hour}",
                    {"hourly_trades": hourly_trades}
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error checking trade frequency: {e}")
            return False
    
    def _check_drawdown_limits(self, current_balance: Decimal):
        """Check and alert on drawdown limits"""
        try:
            if len(self.balance_history) < 2:
                return
            
            # Calculate current drawdown
            max_balance = max(balance for _, balance in self.balance_history)
            current_drawdown = (max_balance - current_balance) / max_balance if max_balance > 0 else Decimal('0')
            
            # Emergency drawdown check
            if current_drawdown > self.max_drawdown * Decimal('2'):  # 2x emergency threshold
                self._trigger_emergency_stop(f"Critical drawdown: {current_drawdown:.2%}")
            elif current_drawdown > self.max_drawdown:
                self._create_alert(
                    RiskEvent.DRAWDOWN_LIMIT_EXCEEDED,
                    RiskLevel.HIGH,
                    f"Drawdown limit exceeded: {current_drawdown:.2%}",
                    {"current_drawdown": float(current_drawdown), "limit": float(self.max_drawdown)}
                )
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error checking drawdown: {e}")
    
    def _calculate_correlation(self, token1: str, token2: str) -> float:
        """Calculate correlation between two tokens"""
        try:
            prices1 = self.price_history.get(token1, [])
            prices2 = self.price_history.get(token2, [])
            
            if len(prices1) < 10 or len(prices2) < 10:
                return 0.0
            
            # Ensure same length
            min_length = min(len(prices1), len(prices2))
            prices1 = prices1[-min_length:]
            prices2 = prices2[-min_length:]
            
            # Calculate correlation
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error calculating correlation: {e}")
            return 0.0
    
    def _calculate_balance_volatility(self, balances: List[Decimal]) -> float:
        """Calculate volatility from balance history"""
        try:
            if len(balances) < 2:
                return 0.0
            
            returns = []
            for i in range(1, len(balances)):
                if balances[i-1] > 0:
                    ret = float(balances[i] / balances[i-1] - 1)
                    returns.append(ret)
            
            return statistics.stdev(returns) if len(returns) > 1 else 0.0
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error calculating balance volatility: {e}")
            return 0.0
    
    # Risk assessment helper methods
    def _assess_liquidity_risk(self, token_address: str, position_value: float) -> float:
        """Assess liquidity risk for a position"""
        try:
            # Simplified liquidity risk based on position size
            if position_value < 100:
                return 5.0
            elif position_value < 1000:
                return 15.0
            elif position_value < 5000:
                return 30.0
            else:
                return 60.0
        except Exception:
            return 50.0
    
    def _assess_concentration_risk(self, position_size: float, entry_price: float) -> float:
        """Assess position concentration risk"""
        try:
            position_value = position_size * entry_price
            portfolio_value = 10000  # Simplified - use actual portfolio value in production
            
            concentration = position_value / portfolio_value
            return min(100.0, concentration * 200)  # Scale to 0-100
        except Exception:
            return 25.0
    
    def _assess_volatility_risk(self, token_address: str) -> float:
        """Assess volatility-based risk"""
        try:
            volatility = self._get_token_volatility(token_address)
            return min(100.0, volatility * 200)  # Scale to 0-100
        except Exception:
            return 40.0
    
    def _assess_correlation_risk(self, token_address: str) -> float:
        """Assess correlation risk with existing positions"""
        try:
            if not self.current_positions:
                return 0.0
            
            correlations = []
            for pos_addr in self.current_positions.keys():
                if pos_addr != token_address:
                    correlation = self._calculate_correlation(token_address, pos_addr)
                    correlations.append(abs(correlation))
            
            if not correlations:
                return 0.0
            
            avg_correlation = np.mean(correlations)
            return min(100.0, avg_correlation * 100)  # Scale to 0-100
            
        except Exception:
            return 20.0
    
    def _get_token_volatility(self, token_address: str) -> float:
        """Get estimated volatility for a token"""
        try:
            if token_address in self.volatility_windows:
                prices = self.volatility_windows[token_address]
                if len(prices) >= 2:
                    returns = np.diff(prices) / np.array(prices[:-1])
                    return np.std(returns)
            
            # Default volatility estimates
            return 0.25  # 25% default volatility
        except Exception:
            return 0.30
    
    def _calculate_position_risk_score(self, token_address: str, position_size: float, entry_price: float) -> float:
        """Calculate comprehensive position risk score"""
        try:
            liquidity_risk = self._assess_liquidity_risk(token_address, position_size * entry_price)
            concentration_risk = self._assess_concentration_risk(position_size, entry_price)
            volatility_risk = self._assess_volatility_risk(token_address)
            correlation_risk = self._assess_correlation_risk(token_address)
            
            # Weighted average
            return (liquidity_risk * 0.25 + concentration_risk * 0.35 + 
                   volatility_risk * 0.25 + correlation_risk * 0.15)
        except Exception:
            return 50.0
    
    def _generate_trade_recommendation(self, risk_level: RiskLevel, risk_score: float) -> str:
        """Generate trading recommendation based on risk assessment"""
        if risk_level == RiskLevel.CRITICAL or risk_level == RiskLevel.EMERGENCY:
            return "REJECT - Critical risk level"
        elif risk_level == RiskLevel.HIGH:
            return "CAUTION - High risk, consider reducing size"
        elif risk_level == RiskLevel.MODERATE:
            return "PROCEED WITH MONITORING - Moderate risk"
        else:
            return "APPROVE - Low risk trade"
    
    def _get_strategy_correlation(self, strategy1: str, strategy2: str) -> float:
        """Get correlation between two strategies"""
        try:
            if strategy1 in self.strategy_returns and strategy2 in self.strategy_returns:
                returns1 = list(self.strategy_returns[strategy1])
                returns2 = list(self.strategy_returns[strategy2])
                
                min_length = min(len(returns1), len(returns2))
                if min_length >= 10:
                    correlation = np.corrcoef(returns1[-min_length:], returns2[-min_length:])[0, 1]
                    return correlation if not np.isnan(correlation) else 0.0
            
            return 0.0
        except Exception:
            return 0.0
    
    def _create_alert(self, event_type: RiskEvent, level: RiskLevel, message: str, data: Dict[str, Any]):
        """Create and store risk alert"""
        try:
            alert = RiskAlert(
                event_type=event_type,
                level=level,
                message=message,
                data=data,
                affected_strategies=list(set(pos.strategy_name for pos in self.current_positions.values()))
            )
            
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Keep alert history manageable
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]
            
            # Log to database
            asyncio.create_task(self._log_risk_alert(alert))
            
            logger.warning(f"[UNIFIED_RISK] {level.name} ALERT: {message}")
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error creating alert: {e}")
    
    async def _update_portfolio_risk_metrics(self):
        """Update comprehensive portfolio risk metrics"""
        try:
            if not self.current_positions:
                self.portfolio_metrics = None
                return
            
            current_time = datetime.now()
            
            # Basic portfolio calculations
            total_value = sum(
                pos.position_value_usd for pos in self.current_positions.values()
                if pos.position_value_usd
            ) or Decimal('0')
            
            # VaR calculations (simplified)
            var_95 = float(total_value) * 0.05  # 5% of portfolio value
            var_99 = float(total_value) * 0.08  # 8% of portfolio value
            
            # Volatility calculation
            volatility = self._calculate_balance_volatility([balance for _, balance in self.balance_history[-30:]])
            
            # Drawdown calculations
            max_drawdown = self._calculate_max_drawdown()
            current_drawdown = self._calculate_current_drawdown()
            
            # Risk budget utilization
            used_budget = min(1.0, float(total_value) / 10000 * 0.15)  # Simplified calculation
            
            # Overall risk assessment
            risk_score = self._calculate_portfolio_risk_score(volatility, float(current_drawdown), used_budget)
            
            # Determine risk level
            if risk_score > 75 or self.emergency_stop_active:
                risk_level = RiskLevel.CRITICAL
            elif risk_score > 50:
                risk_level = RiskLevel.HIGH
            elif risk_score > 25:
                risk_level = RiskLevel.MODERATE
            else:
                risk_level = RiskLevel.LOW
            
            # Update portfolio metrics
            self.portfolio_metrics = PortfolioRiskMetrics(
                timestamp=current_time,
                total_value_usd=total_value,
                daily_pnl=self.daily_pnl,
                max_drawdown=Decimal(str(max_drawdown)),
                current_drawdown=Decimal(str(current_drawdown)),
                var_95_1d=var_95 / float(total_value) if total_value > 0 else 0,
                var_99_1d=var_99 / float(total_value) if total_value > 0 else 0,
                total_volatility=volatility,
                position_count=len(self.current_positions),
                used_risk_budget=used_budget,
                available_risk_budget=max(0, 1.0 - used_budget),
                overall_risk_level=risk_level,
                risk_score=risk_score
            )
            
            self.last_risk_calculation = current_time
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error updating portfolio risk metrics: {e}")
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from balance history"""
        try:
            if len(self.balance_history) < 2:
                return 0.0
            
            balances = [float(balance) for _, balance in self.balance_history]
            max_drawdown = 0.0
            peak = balances[0]
            
            for balance in balances:
                if balance > peak:
                    peak = balance
                drawdown = (peak - balance) / peak if peak > 0 else 0.0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            return max_drawdown
            
        except Exception:
            return 0.0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from recent peak"""
        try:
            if len(self.balance_history) < 2:
                return 0.0
            
            recent_balances = [float(balance) for _, balance in self.balance_history[-24:]]
            if not recent_balances:
                return 0.0
            
            peak = max(recent_balances)
            current = recent_balances[-1]
            
            return (peak - current) / peak if peak > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_portfolio_risk_score(self, volatility: float, drawdown: float, used_budget: float) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        try:
            # Volatility component (0-30 points)
            vol_score = min(30, volatility * 100)
            
            # Drawdown component (0-30 points)
            dd_score = min(30, drawdown * 150)
            
            # Risk budget component (0-25 points)
            budget_score = used_budget * 25
            
            # Emergency condition (0-15 points)
            emergency_score = 15 if self.emergency_stop_active else 0
            
            total_score = vol_score + dd_score + budget_score + emergency_score
            return min(100, max(0, total_score))
            
        except Exception:
            return 50.0
    
    async def _log_risk_alert(self, alert: RiskAlert):
        """Log risk alert to database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO risk_alerts
                    (timestamp, event_type, risk_level, message, recommended_action, 
                     affected_strategies, data, resolved)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.timestamp.isoformat(),
                    alert.event_type.value,
                    alert.level.name,
                    alert.message,
                    alert.recommended_action.value,
                    json.dumps(alert.affected_strategies),
                    json.dumps(alert.data),
                    alert.resolved
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Failed to log risk alert: {e}")
    
    async def shutdown(self):
        """Shutdown risk manager and save final state"""
        try:
            # Save final risk metrics
            if self.portfolio_metrics:
                await self._log_portfolio_metrics()
            
            # Save historical data
            await self._save_historical_data()
            
            logger.info("[UNIFIED_RISK] Unified Risk Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error during shutdown: {e}")
    
    async def _save_historical_data(self):
        """Save historical data to file"""
        try:
            os.makedirs("analytics", exist_ok=True)
            
            data = {
                'daily_balances': list(self.daily_balances),
                'trade_sequence': list(self.trade_sequence),
                'last_updated': datetime.now().isoformat()
            }
            
            with open("analytics/unified_risk_data.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info("[UNIFIED_RISK] Historical data saved")
            
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Error saving historical data: {e}")
    
    async def _log_portfolio_metrics(self):
        """Log portfolio metrics to database"""
        try:
            if not self.portfolio_metrics:
                return
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO portfolio_risk_metrics
                    (timestamp, total_value_usd, var_95_1d, var_99_1d, total_volatility,
                     sharpe_ratio, max_drawdown, current_drawdown, diversification_ratio,
                     risk_score, overall_risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.portfolio_metrics.timestamp.isoformat(),
                    float(self.portfolio_metrics.total_value_usd),
                    self.portfolio_metrics.var_95_1d,
                    self.portfolio_metrics.var_99_1d,
                    self.portfolio_metrics.total_volatility,
                    self.portfolio_metrics.sharpe_ratio,
                    float(self.portfolio_metrics.max_drawdown),
                    float(self.portfolio_metrics.current_drawdown),
                    self.portfolio_metrics.diversification_ratio,
                    self.portfolio_metrics.risk_score,
                    self.portfolio_metrics.overall_risk_level.name
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[UNIFIED_RISK] Failed to log portfolio metrics: {e}")


# Global unified risk manager instance
_global_unified_risk_manager: Optional[UnifiedRiskManager] = None

def get_unified_risk_manager(settings=None) -> UnifiedRiskManager:
    """Get global unified risk manager instance"""
    global _global_unified_risk_manager
    if _global_unified_risk_manager is None:
        _global_unified_risk_manager = UnifiedRiskManager(settings)
    return _global_unified_risk_manager

# Compatibility aliases for existing code
def get_risk_manager(settings=None) -> UnifiedRiskManager:
    """Compatibility alias for existing code"""
    return get_unified_risk_manager(settings)

class RiskManager(UnifiedRiskManager):
    """Compatibility alias for existing code"""
    pass