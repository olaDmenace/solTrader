#!/usr/bin/env python3
"""
Portfolio-Level Risk Management System
Comprehensive risk management system that monitors and controls portfolio-wide risk
through VaR/CVaR calculations, correlation analysis, stress testing, and emergency controls.

Key Features:
1. Portfolio-wide VaR and CVaR calculations
2. Cross-strategy correlation tracking and analysis
3. Dynamic position sizing based on portfolio risk
4. Stress testing and scenario analysis
5. Risk budget enforcement and emergency position reduction
6. Real-time risk monitoring and alerting
7. Portfolio optimization under risk constraints
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
import aiosqlite
from collections import deque, defaultdict
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EmergencyAction(Enum):
    NONE = "NONE"
    REDUCE_POSITIONS = "REDUCE_POSITIONS"
    HALT_TRADING = "HALT_TRADING"
    LIQUIDATE_ALL = "LIQUIDATE_ALL"

@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""
    timestamp: datetime
    
    # VaR/CVaR metrics
    var_95_1d: float = 0.0      # 1-day VaR at 95% confidence
    var_99_1d: float = 0.0      # 1-day VaR at 99% confidence
    cvar_95_1d: float = 0.0     # 1-day CVaR at 95% confidence
    cvar_99_1d: float = 0.0     # 1-day CVaR at 99% confidence
    
    # Portfolio metrics
    total_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    beta_to_market: float = 0.0
    
    # Correlation metrics
    avg_strategy_correlation: float = 0.0
    max_strategy_correlation: float = 0.0
    diversification_ratio: float = 0.0
    
    # Risk budgets
    used_risk_budget: float = 0.0      # Percentage of risk budget used
    available_risk_budget: float = 0.0  # Remaining risk budget
    
    # Position sizing
    recommended_leverage: float = 0.0
    max_position_size: float = 0.0
    current_leverage: float = 0.0
    
    # Risk level assessment
    overall_risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0  # 0-100 risk score

@dataclass
class StressTestScenario:
    """Stress testing scenario definition"""
    name: str
    description: str
    market_shock: float          # Market movement percentage
    volatility_shock: float      # Volatility increase multiplier
    correlation_shock: float     # Correlation increase factor
    duration_days: int = 1       # Scenario duration
    probability: float = 0.05    # Estimated probability

@dataclass
class StressTestResult:
    """Results from stress testing"""
    scenario_name: str
    portfolio_pnl: float
    portfolio_return: float
    var_impact: float
    strategy_impacts: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario_name': self.scenario_name,
            'portfolio_pnl': self.portfolio_pnl,
            'portfolio_return': self.portfolio_return,
            'var_impact': self.var_impact,
            'strategy_impacts': self.strategy_impacts,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class RiskAlert:
    """Risk alert for critical situations"""
    alert_type: str
    risk_level: RiskLevel
    message: str
    recommended_action: EmergencyAction
    affected_strategies: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'recommended_action': self.recommended_action.value,
            'affected_strategies': self.affected_strategies,
            'timestamp': self.timestamp.isoformat()
        }

class RiskDatabase:
    """SQLite database for risk metrics and history"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def initialize(self):
        """Initialize database tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Portfolio risk metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS portfolio_risk_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        var_95_1d REAL NOT NULL,
                        var_99_1d REAL NOT NULL,
                        cvar_95_1d REAL NOT NULL,
                        cvar_99_1d REAL NOT NULL,
                        total_volatility REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        avg_strategy_correlation REAL NOT NULL,
                        diversification_ratio REAL NOT NULL,
                        used_risk_budget REAL NOT NULL,
                        overall_risk_level TEXT NOT NULL,
                        risk_score REAL NOT NULL
                    )
                """)
                
                # Correlation matrix table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS correlation_matrix (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy1 TEXT NOT NULL,
                        strategy2 TEXT NOT NULL,
                        correlation REAL NOT NULL,
                        lookback_days INTEGER NOT NULL
                    )
                """)
                
                # Stress test results table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS stress_test_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        scenario_name TEXT NOT NULL,
                        portfolio_pnl REAL NOT NULL,
                        portfolio_return REAL NOT NULL,
                        var_impact REAL NOT NULL,
                        strategy_impacts TEXT NOT NULL  -- JSON string
                    )
                """)
                
                # Risk alerts table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS risk_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        recommended_action TEXT NOT NULL,
                        affected_strategies TEXT NOT NULL  -- JSON string
                    )
                """)
                
                await db.commit()
                
            logger.info("[RISK_DB] Risk management database initialized")
            
        except Exception as e:
            logger.error(f"[RISK_DB] Database initialization failed: {e}")
            raise
    
    async def log_risk_metrics(self, metrics: PortfolioRiskMetrics):
        """Log portfolio risk metrics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO portfolio_risk_metrics
                    (timestamp, var_95_1d, var_99_1d, cvar_95_1d, cvar_99_1d,
                     total_volatility, sharpe_ratio, max_drawdown, avg_strategy_correlation,
                     diversification_ratio, used_risk_budget, overall_risk_level, risk_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp.isoformat(),
                    metrics.var_95_1d,
                    metrics.var_99_1d,
                    metrics.cvar_95_1d,
                    metrics.cvar_99_1d,
                    metrics.total_volatility,
                    metrics.sharpe_ratio,
                    metrics.max_drawdown,
                    metrics.avg_strategy_correlation,
                    metrics.diversification_ratio,
                    metrics.used_risk_budget,
                    metrics.overall_risk_level.value,
                    metrics.risk_score
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[RISK_DB] Failed to log risk metrics: {e}")
    
    async def log_correlation_matrix(self, correlations: Dict[Tuple[str, str], float], lookback_days: int):
        """Log correlation matrix"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                timestamp = datetime.now().isoformat()
                
                for (strategy1, strategy2), correlation in correlations.items():
                    await db.execute("""
                        INSERT INTO correlation_matrix
                        (timestamp, strategy1, strategy2, correlation, lookback_days)
                        VALUES (?, ?, ?, ?, ?)
                    """, (timestamp, strategy1, strategy2, correlation, lookback_days))
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"[RISK_DB] Failed to log correlation matrix: {e}")
    
    async def log_stress_test(self, result: StressTestResult):
        """Log stress test result"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO stress_test_results
                    (timestamp, scenario_name, portfolio_pnl, portfolio_return, var_impact, strategy_impacts)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.timestamp.isoformat(),
                    result.scenario_name,
                    result.portfolio_pnl,
                    result.portfolio_return,
                    result.var_impact,
                    json.dumps(result.strategy_impacts)
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[RISK_DB] Failed to log stress test: {e}")
    
    async def log_risk_alert(self, alert: RiskAlert):
        """Log risk alert"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO risk_alerts
                    (timestamp, alert_type, risk_level, message, recommended_action, affected_strategies)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    alert.timestamp.isoformat(),
                    alert.alert_type,
                    alert.risk_level.value,
                    alert.message,
                    alert.recommended_action.value,
                    json.dumps(alert.affected_strategies)
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[RISK_DB] Failed to log risk alert: {e}")

class CorrelationAnalyzer:
    """Analyzes correlations between strategy returns"""
    
    def __init__(self, lookback_window: int = 30):
        self.lookback_window = lookback_window
        self.strategy_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_window))
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
    
    def update_strategy_return(self, strategy_name: str, return_value: float):
        """Update strategy return data"""
        try:
            self.strategy_returns[strategy_name].append(return_value)
            
            # Recalculate correlations if we have enough data
            if len(self.strategy_returns[strategy_name]) >= 10:
                self._calculate_correlations()
                
        except Exception as e:
            logger.error(f"[CORRELATION] Error updating strategy return: {e}")
    
    def _calculate_correlations(self):
        """Calculate pairwise correlations between strategies"""
        try:
            strategies = list(self.strategy_returns.keys())
            
            for i, strategy1 in enumerate(strategies):
                for j, strategy2 in enumerate(strategies[i:], i):
                    if strategy1 == strategy2:
                        self.correlation_matrix[(strategy1, strategy2)] = 1.0
                        continue
                    
                    # Get aligned returns
                    returns1 = list(self.strategy_returns[strategy1])
                    returns2 = list(self.strategy_returns[strategy2])
                    
                    min_length = min(len(returns1), len(returns2))
                    if min_length < 10:
                        continue
                    
                    # Calculate correlation
                    correlation = np.corrcoef(returns1[-min_length:], returns2[-min_length:])[0, 1]
                    
                    # Handle NaN correlations
                    if np.isnan(correlation):
                        correlation = 0.0
                    
                    self.correlation_matrix[(strategy1, strategy2)] = correlation
                    self.correlation_matrix[(strategy2, strategy1)] = correlation
                    
        except Exception as e:
            logger.error(f"[CORRELATION] Error calculating correlations: {e}")
    
    def get_correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """Get current correlation matrix"""
        return self.correlation_matrix.copy()
    
    def get_average_correlation(self) -> float:
        """Get average correlation across all strategy pairs"""
        try:
            correlations = []
            strategies = list(self.strategy_returns.keys())
            
            for i, strategy1 in enumerate(strategies):
                for strategy2 in strategies[i+1:]:
                    correlation = self.correlation_matrix.get((strategy1, strategy2), 0.0)
                    correlations.append(abs(correlation))  # Use absolute correlation
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"[CORRELATION] Error calculating average correlation: {e}")
            return 0.0
    
    def get_diversification_ratio(self) -> float:
        """Calculate diversification ratio (1 - average correlation)"""
        try:
            avg_correlation = self.get_average_correlation()
            return max(0.0, 1.0 - avg_correlation)
            
        except Exception as e:
            logger.error(f"[CORRELATION] Error calculating diversification ratio: {e}")
            return 0.0

class PortfolioRiskManager:
    """
    Comprehensive portfolio-level risk management system
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        
        # Initialize components
        import os
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.db = RiskDatabase(f"{log_dir}/portfolio_risk.db")
        self.correlation_analyzer = CorrelationAnalyzer()
        
        # Risk parameters
        self.max_portfolio_var = 0.05    # 5% max daily VaR
        self.max_portfolio_leverage = 1.5 # Max 1.5x leverage
        self.correlation_alert_threshold = 0.8  # Alert if avg correlation > 80%
        self.risk_budget = 0.10          # 10% risk budget
        self.emergency_var_threshold = 0.08  # 8% VaR triggers emergency action
        
        # Stress test scenarios
        self.stress_scenarios = [
            StressTestScenario("Market Crash", "30% market decline", -0.30, 2.0, 1.5, 1, 0.02),
            StressTestScenario("Flash Crash", "15% sudden drop", -0.15, 3.0, 2.0, 1, 0.05),
            StressTestScenario("Volatility Spike", "Volatility doubles", 0.0, 2.0, 1.3, 1, 0.10),
            StressTestScenario("Correlation Breakdown", "All correlations -> 0.9", 0.0, 1.2, 3.0, 1, 0.05),
            StressTestScenario("Liquidity Crisis", "20% decline + vol spike", -0.20, 2.5, 1.8, 3, 0.03)
        ]
        
        # History tracking
        self.risk_metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.alerts_history: List[RiskAlert] = []
        self.last_risk_calculation = datetime.now()
        self.risk_calculation_interval = timedelta(minutes=5)
        
        # Emergency controls
        self.emergency_mode = False
        self.position_reduction_factor = 0.5  # Reduce positions by 50% in emergency
        
        logger.info("[RISK_MANAGER] Portfolio Risk Manager initialized")
    
    async def start(self):
        """Start the portfolio risk manager"""
        try:
            await self.db.initialize()
            logger.info("[RISK_MANAGER] Portfolio Risk Manager started")
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Failed to start risk manager: {e}")
            raise
    
    async def stop(self):
        """Stop the portfolio risk manager"""
        try:
            # Save final risk metrics
            if self.risk_metrics_history:
                await self.db.log_risk_metrics(self.risk_metrics_history[-1])
                
            logger.info("[RISK_MANAGER] Portfolio Risk Manager stopped")
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error during risk manager shutdown: {e}")
    
    async def update_portfolio_data(
        self,
        strategy_allocations: Dict[str, float],
        strategy_returns: Dict[str, List[float]],
        strategy_metrics: Dict[str, Any]
    ) -> PortfolioRiskMetrics:
        """Update portfolio data and calculate risk metrics"""
        try:
            # Update correlation data
            for strategy_name, returns_list in strategy_returns.items():
                if returns_list:
                    latest_return = returns_list[-1]
                    self.correlation_analyzer.update_strategy_return(strategy_name, latest_return)
            
            # Calculate portfolio risk metrics
            risk_metrics = await self._calculate_portfolio_risk_metrics(
                strategy_allocations, strategy_returns, strategy_metrics
            )
            
            # Store metrics
            self.risk_metrics_history.append(risk_metrics)
            
            # Check for risk alerts
            await self._check_risk_alerts(risk_metrics)
            
            # Log to database
            if datetime.now() - self.last_risk_calculation >= self.risk_calculation_interval:
                await self.db.log_risk_metrics(risk_metrics)
                await self.db.log_correlation_matrix(
                    self.correlation_analyzer.get_correlation_matrix(),
                    self.correlation_analyzer.lookback_window
                )
                self.last_risk_calculation = datetime.now()
            
            logger.debug(f"[RISK_MANAGER] Risk metrics updated: VaR95={risk_metrics.var_95_1d:.3f}, Risk Level={risk_metrics.overall_risk_level.value}")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error updating portfolio data: {e}")
            # Return default metrics on error
            return PortfolioRiskMetrics(timestamp=datetime.now())
    
    async def _calculate_portfolio_risk_metrics(
        self,
        allocations: Dict[str, float],
        returns: Dict[str, List[float]],
        metrics: Dict[str, Any]
    ) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            timestamp = datetime.now()
            
            # Get portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(allocations, returns)
            
            if len(portfolio_returns) < 10:
                logger.warning("[RISK_MANAGER] Insufficient return data for risk calculations")
                return PortfolioRiskMetrics(timestamp=timestamp)
            
            # VaR calculations
            var_95_1d = np.percentile(portfolio_returns, 5)  # 5th percentile
            var_99_1d = np.percentile(portfolio_returns, 1)  # 1st percentile
            
            # CVaR calculations (expected shortfall)
            cvar_95_1d = np.mean([r for r in portfolio_returns if r <= var_95_1d])
            cvar_99_1d = np.mean([r for r in portfolio_returns if r <= var_99_1d])
            
            # Portfolio volatility
            total_volatility = np.std(portfolio_returns)
            
            # Sharpe ratio
            sharpe_ratio = np.mean(portfolio_returns) / total_volatility if total_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + np.array(portfolio_returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (running_max - cumulative_returns) / running_max
            max_drawdown = np.max(drawdowns)
            
            # Correlation metrics
            avg_correlation = self.correlation_analyzer.get_average_correlation()
            max_correlation = max(self.correlation_analyzer.get_correlation_matrix().values()) if self.correlation_analyzer.get_correlation_matrix() else 0.0
            diversification_ratio = self.correlation_analyzer.get_diversification_ratio()
            
            # Risk budget utilization
            used_risk_budget = abs(var_95_1d) / self.risk_budget if self.risk_budget > 0 else 0
            available_risk_budget = max(0, 1.0 - used_risk_budget)
            
            # Position sizing recommendations
            recommended_leverage = min(self.max_portfolio_leverage, self.risk_budget / max(abs(var_95_1d), 0.01))
            max_position_size = self.risk_budget / (2 * max(total_volatility, 0.01))  # 2-sigma rule
            current_leverage = sum(allocations.values())
            
            # Risk level assessment
            risk_score = self._calculate_risk_score(var_95_1d, total_volatility, avg_correlation, max_drawdown)
            overall_risk_level = self._determine_risk_level(risk_score, var_95_1d)
            
            return PortfolioRiskMetrics(
                timestamp=timestamp,
                var_95_1d=var_95_1d,
                var_99_1d=var_99_1d,
                cvar_95_1d=cvar_95_1d,
                cvar_99_1d=cvar_99_1d,
                total_volatility=total_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_strategy_correlation=avg_correlation,
                max_strategy_correlation=max_correlation,
                diversification_ratio=diversification_ratio,
                used_risk_budget=used_risk_budget,
                available_risk_budget=available_risk_budget,
                recommended_leverage=recommended_leverage,
                max_position_size=max_position_size,
                current_leverage=current_leverage,
                overall_risk_level=overall_risk_level,
                risk_score=risk_score
            )
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error calculating portfolio risk metrics: {e}")
            return PortfolioRiskMetrics(timestamp=datetime.now())
    
    def _calculate_portfolio_returns(
        self,
        allocations: Dict[str, float],
        strategy_returns: Dict[str, List[float]]
    ) -> List[float]:
        """Calculate portfolio returns based on allocations and strategy returns"""
        try:
            # Find minimum return series length
            min_length = float('inf')
            for strategy_name, returns_list in strategy_returns.items():
                if strategy_name in allocations and allocations[strategy_name] > 0:
                    min_length = min(min_length, len(returns_list))
            
            if min_length == float('inf') or min_length == 0:
                return []
            
            portfolio_returns = []
            for i in range(min_length):
                portfolio_return = 0.0
                total_allocation = 0.0
                
                for strategy_name, allocation in allocations.items():
                    if allocation > 0 and strategy_name in strategy_returns:
                        if i < len(strategy_returns[strategy_name]):
                            portfolio_return += allocation * strategy_returns[strategy_name][i]
                            total_allocation += allocation
                
                if total_allocation > 0:
                    portfolio_returns.append(portfolio_return / total_allocation)
            
            return portfolio_returns
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error calculating portfolio returns: {e}")
            return []
    
    def _calculate_risk_score(self, var_95: float, volatility: float, avg_correlation: float, max_drawdown: float) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            # VaR component (0-40 points)
            var_score = min(40, abs(var_95) / 0.10 * 40)
            
            # Volatility component (0-25 points)
            vol_score = min(25, volatility / 0.30 * 25)
            
            # Correlation component (0-20 points)
            corr_score = avg_correlation * 20
            
            # Drawdown component (0-15 points)
            dd_score = min(15, max_drawdown / 0.20 * 15)
            
            total_score = var_score + vol_score + corr_score + dd_score
            return min(100, max(0, total_score))
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error calculating risk score: {e}")
            return 50.0
    
    def _determine_risk_level(self, risk_score: float, var_95: float) -> RiskLevel:
        """Determine overall risk level"""
        try:
            # Critical level if VaR exceeds emergency threshold
            if abs(var_95) > self.emergency_var_threshold:
                return RiskLevel.CRITICAL
            
            # Risk level based on score
            if risk_score >= 75:
                return RiskLevel.HIGH
            elif risk_score >= 50:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception:
            return RiskLevel.MEDIUM
    
    async def _check_risk_alerts(self, metrics: PortfolioRiskMetrics):
        """Check for risk alert conditions"""
        try:
            alerts = []
            
            # VaR alert
            if abs(metrics.var_95_1d) > self.max_portfolio_var:
                alerts.append(RiskAlert(
                    alert_type="VAR_BREACH",
                    risk_level=RiskLevel.HIGH if abs(metrics.var_95_1d) < self.emergency_var_threshold else RiskLevel.CRITICAL,
                    message=f"Portfolio VaR ({metrics.var_95_1d:.2%}) exceeds limit ({self.max_portfolio_var:.2%})",
                    recommended_action=EmergencyAction.REDUCE_POSITIONS if abs(metrics.var_95_1d) < self.emergency_var_threshold else EmergencyAction.HALT_TRADING,
                    affected_strategies=list(self.correlation_analyzer.strategy_returns.keys())
                ))
            
            # Correlation alert
            if metrics.avg_strategy_correlation > self.correlation_alert_threshold:
                alerts.append(RiskAlert(
                    alert_type="HIGH_CORRELATION",
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Average strategy correlation ({metrics.avg_strategy_correlation:.2%}) is high",
                    recommended_action=EmergencyAction.REDUCE_POSITIONS,
                    affected_strategies=list(self.correlation_analyzer.strategy_returns.keys())
                ))
            
            # Risk budget alert
            if metrics.used_risk_budget > 0.9:
                alerts.append(RiskAlert(
                    alert_type="RISK_BUDGET_BREACH",
                    risk_level=RiskLevel.HIGH,
                    message=f"Risk budget utilization at {metrics.used_risk_budget:.1%}",
                    recommended_action=EmergencyAction.REDUCE_POSITIONS,
                    affected_strategies=list(self.correlation_analyzer.strategy_returns.keys())
                ))
            
            # Log and store alerts
            for alert in alerts:
                await self.db.log_risk_alert(alert)
                self.alerts_history.append(alert)
                logger.warning(f"[RISK_ALERT] {alert.alert_type}: {alert.message}")
                
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error checking risk alerts: {e}")
    
    async def run_stress_tests(
        self,
        allocations: Dict[str, float],
        strategy_returns: Dict[str, List[float]]
    ) -> List[StressTestResult]:
        """Run stress tests on the portfolio"""
        try:
            results = []
            
            for scenario in self.stress_scenarios:
                result = await self._run_stress_scenario(scenario, allocations, strategy_returns)
                results.append(result)
                await self.db.log_stress_test(result)
            
            logger.info(f"[RISK_MANAGER] Completed {len(results)} stress tests")
            return results
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error running stress tests: {e}")
            return []
    
    async def _run_stress_scenario(
        self,
        scenario: StressTestScenario,
        allocations: Dict[str, float],
        strategy_returns: Dict[str, List[float]]
    ) -> StressTestResult:
        """Run a single stress test scenario"""
        try:
            strategy_impacts = {}
            total_portfolio_impact = 0.0
            
            for strategy_name, allocation in allocations.items():
                if allocation <= 0 or strategy_name not in strategy_returns:
                    strategy_impacts[strategy_name] = 0.0
                    continue
                
                # Get strategy volatility
                returns = strategy_returns[strategy_name][-30:]  # Last 30 returns
                if not returns:
                    strategy_impacts[strategy_name] = 0.0
                    continue
                
                strategy_vol = np.std(returns)
                
                # Apply scenario shocks
                base_impact = scenario.market_shock
                vol_impact = strategy_vol * (scenario.volatility_shock - 1.0)
                
                # Correlation effect (higher correlation amplifies impact)
                avg_correlation = self.correlation_analyzer.get_average_correlation()
                correlation_multiplier = 1.0 + (avg_correlation * (scenario.correlation_shock - 1.0))
                
                strategy_impact = (base_impact + vol_impact) * correlation_multiplier
                strategy_impacts[strategy_name] = strategy_impact
                
                # Weight by allocation
                total_portfolio_impact += allocation * strategy_impact
            
            # Calculate portfolio metrics
            total_capital = sum(allocations.values()) * 10000  # Assume $10k base
            portfolio_pnl = total_portfolio_impact * total_capital
            portfolio_return = total_portfolio_impact
            
            # VaR impact (approximate)
            current_var = self.risk_metrics_history[-1].var_95_1d if self.risk_metrics_history else 0.05
            var_impact = abs(total_portfolio_impact) - abs(current_var)
            
            return StressTestResult(
                scenario_name=scenario.name,
                portfolio_pnl=portfolio_pnl,
                portfolio_return=portfolio_return,
                var_impact=var_impact,
                strategy_impacts=strategy_impacts,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error running stress scenario {scenario.name}: {e}")
            return StressTestResult(
                scenario_name=scenario.name,
                portfolio_pnl=0.0,
                portfolio_return=0.0,
                var_impact=0.0,
                strategy_impacts={},
                timestamp=datetime.now()
            )
    
    async def get_position_sizing_recommendation(
        self,
        strategy_name: str,
        expected_return: float,
        expected_volatility: float
    ) -> Dict[str, float]:
        """Get position sizing recommendation for a strategy"""
        try:
            if not self.risk_metrics_history:
                return {'recommended_size': 0.02, 'max_size': 0.05, 'confidence': 0.5}
            
            latest_metrics = self.risk_metrics_history[-1]
            
            # Risk-adjusted position sizing
            risk_contribution = expected_volatility * latest_metrics.avg_strategy_correlation
            available_risk = latest_metrics.available_risk_budget
            
            # Kelly criterion approximation
            if expected_volatility > 0:
                kelly_fraction = expected_return / (expected_volatility ** 2)
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            else:
                kelly_fraction = 0.02
            
            # Risk budget constraint
            risk_budget_size = available_risk / max(risk_contribution, 0.01)
            
            # Final recommendation
            recommended_size = min(kelly_fraction, risk_budget_size, latest_metrics.max_position_size)
            max_size = min(0.10, latest_metrics.max_position_size * 1.5)  # Max 10%
            
            # Confidence based on risk level
            confidence_map = {
                RiskLevel.LOW: 0.9,
                RiskLevel.MEDIUM: 0.7,
                RiskLevel.HIGH: 0.4,
                RiskLevel.CRITICAL: 0.1
            }
            confidence = confidence_map.get(latest_metrics.overall_risk_level, 0.5)
            
            return {
                'recommended_size': recommended_size,
                'max_size': max_size,
                'confidence': confidence,
                'risk_contribution': risk_contribution,
                'available_risk_budget': available_risk
            }
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error calculating position sizing: {e}")
            return {'recommended_size': 0.02, 'max_size': 0.05, 'confidence': 0.5}
    
    async def trigger_emergency_action(self, action: EmergencyAction, affected_strategies: List[str]):
        """Trigger emergency risk management action"""
        try:
            logger.critical(f"[RISK_MANAGER] EMERGENCY ACTION TRIGGERED: {action.value}")
            
            if action == EmergencyAction.REDUCE_POSITIONS:
                # Set emergency mode and reduction factor
                self.emergency_mode = True
                logger.warning(f"[RISK_MANAGER] Reducing positions by {(1-self.position_reduction_factor):.0%}")
                
            elif action == EmergencyAction.HALT_TRADING:
                # Halt all new position opening
                self.emergency_mode = True
                logger.critical("[RISK_MANAGER] TRADING HALTED - No new positions allowed")
                
            elif action == EmergencyAction.LIQUIDATE_ALL:
                # Emergency liquidation
                self.emergency_mode = True
                self.position_reduction_factor = 0.0  # Liquidate everything
                logger.critical("[RISK_MANAGER] EMERGENCY LIQUIDATION - Closing all positions")
            
            # Create alert
            alert = RiskAlert(
                alert_type="EMERGENCY_ACTION",
                risk_level=RiskLevel.CRITICAL,
                message=f"Emergency action triggered: {action.value}",
                recommended_action=action,
                affected_strategies=affected_strategies
            )
            
            await self.db.log_risk_alert(alert)
            self.alerts_history.append(alert)
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error triggering emergency action: {e}")
    
    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get current risk status summary"""
        try:
            if not self.risk_metrics_history:
                return {'status': 'insufficient_data'}
            
            latest_metrics = self.risk_metrics_history[-1]
            recent_alerts = [alert for alert in self.alerts_history[-10:]]  # Last 10 alerts
            
            return {
                'timestamp': datetime.now().isoformat(),
                'risk_level': latest_metrics.overall_risk_level.value,
                'risk_score': latest_metrics.risk_score,
                'var_95_1d': latest_metrics.var_95_1d,
                'portfolio_volatility': latest_metrics.total_volatility,
                'diversification_ratio': latest_metrics.diversification_ratio,
                'used_risk_budget': latest_metrics.used_risk_budget,
                'emergency_mode': self.emergency_mode,
                'recent_alerts_count': len(recent_alerts),
                'correlation_status': 'HIGH' if latest_metrics.avg_strategy_correlation > 0.6 else 'NORMAL'
            }
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error getting risk status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def log_system_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """
        Log system metric (compatibility method for portfolio monitoring)
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            metadata: Optional metadata dict
        """
        try:
            # Simple logging implementation for compatibility
            logger = logging.getLogger(__name__)
            logger.debug(f"[PORTFOLIO_METRIC] {metric_name}: {value}")
            
            # Could extend to store in risk database if needed
            # await self.db.store_metric(metric_name, value, metadata)
            
        except Exception as e:
            logger.error(f"[RISK_MANAGER] Error logging system metric {metric_name}: {e}")