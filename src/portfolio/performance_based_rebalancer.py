#!/usr/bin/env python3
"""
Performance-Based Strategy Rebalancer
Advanced rebalancing system that dynamically adjusts strategy allocations based on 
rolling performance windows, momentum detection, and adaptive performance metrics.

Key Features:
1. Multi-timeframe performance analysis (1d, 7d, 30d)
2. Strategy momentum and mean reversion detection
3. Performance attribution analysis
4. Dynamic strategy weights based on recent performance
5. Adaptive rebalancing thresholds
6. Performance-based allocation scaling
7. Risk-adjusted performance scoring
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics
import sqlite3
import aiosqlite

logger = logging.getLogger(__name__)

class PerformanceRegime(Enum):
    STRONG_MOMENTUM = "STRONG_MOMENTUM"      # Strategy outperforming consistently
    WEAK_MOMENTUM = "WEAK_MOMENTUM"          # Strategy underperforming consistently  
    MEAN_REVERTING = "MEAN_REVERTING"        # Strategy showing mean reversion
    VOLATILE = "VOLATILE"                     # High volatility, inconsistent performance
    STABLE = "STABLE"                        # Consistent, low volatility performance

class RebalanceSignal(Enum):
    STRONG_BUY = "STRONG_BUY"               # Increase allocation significantly
    BUY = "BUY"                             # Increase allocation moderately
    HOLD = "HOLD"                           # Maintain current allocation
    SELL = "SELL"                           # Decrease allocation moderately
    STRONG_SELL = "STRONG_SELL"             # Decrease allocation significantly

@dataclass
class StrategyPerformanceWindow:
    """Performance metrics over a specific time window"""
    strategy_name: str
    window_days: int
    
    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0  # Return vs benchmark
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    trades_count: int = 0
    
    # Performance consistency
    performance_consistency: float = 0.0  # 0-1, higher is more consistent
    trend_strength: float = 0.0           # -1 to 1, negative for downtrend
    
    # Timestamps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    
    def performance_score(self) -> float:
        """Calculate overall performance score"""
        try:
            # Base score from risk-adjusted return
            base_score = self.sharpe_ratio * 10  # Scale Sharpe ratio
            
            # Consistency bonus
            consistency_bonus = self.performance_consistency * 5
            
            # Win rate bonus
            win_rate_bonus = max(0, (self.win_rate - 0.5) * 10)
            
            # Drawdown penalty
            drawdown_penalty = self.max_drawdown * 20
            
            # Trend strength bonus/penalty
            trend_bonus = self.trend_strength * 3
            
            score = base_score + consistency_bonus + win_rate_bonus - drawdown_penalty + trend_bonus
            return max(-20, min(20, score))  # Clamp between -20 and 20
            
        except Exception:
            return 0.0

@dataclass
class PerformanceAnalysisResult:
    """Result of performance analysis for a strategy"""
    strategy_name: str
    
    # Performance windows
    window_1d: StrategyPerformanceWindow
    window_7d: StrategyPerformanceWindow  
    window_30d: StrategyPerformanceWindow
    
    # Performance regime
    current_regime: PerformanceRegime
    regime_confidence: float  # 0-1
    
    # Rebalancing recommendation
    rebalance_signal: RebalanceSignal
    recommended_allocation_change: float  # +/- percentage points
    
    # Attribution analysis
    alpha: float = 0.0          # Strategy alpha vs market
    beta: float = 1.0           # Strategy beta vs market
    tracking_error: float = 0.0  # Volatility of excess returns
    
    # Momentum indicators
    momentum_1d: float = 0.0    # 1-day momentum
    momentum_7d: float = 0.0    # 7-day momentum
    momentum_30d: float = 0.0   # 30-day momentum
    
    # Quality metrics
    strategy_quality_score: float = 0.0  # Overall strategy quality (0-100)
    recommendation_confidence: float = 0.0  # Confidence in rebalance signal (0-1)
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'strategy_name': self.strategy_name,
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'rebalance_signal': self.rebalance_signal.value,
            'recommended_allocation_change': self.recommended_allocation_change,
            'alpha': self.alpha,
            'beta': self.beta,
            'momentum_1d': self.momentum_1d,
            'momentum_7d': self.momentum_7d,
            'momentum_30d': self.momentum_30d,
            'strategy_quality_score': self.strategy_quality_score,
            'recommendation_confidence': self.recommendation_confidence,
            'window_1d_score': self.window_1d.performance_score(),
            'window_7d_score': self.window_7d.performance_score(),
            'window_30d_score': self.window_30d.performance_score(),
            'timestamp': self.timestamp.isoformat()
        }

class PerformanceDatabase:
    """Database for storing performance analysis results"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def initialize(self):
        """Initialize database tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Performance windows table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS performance_windows (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        window_days INTEGER NOT NULL,
                        total_return REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        performance_score REAL NOT NULL,
                        trend_strength REAL NOT NULL
                    )
                """)
                
                # Performance analysis results table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS performance_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        current_regime TEXT NOT NULL,
                        rebalance_signal TEXT NOT NULL,
                        recommended_change REAL NOT NULL,
                        alpha REAL NOT NULL,
                        beta REAL NOT NULL,
                        quality_score REAL NOT NULL,
                        confidence REAL NOT NULL
                    )
                """)
                
                await db.commit()
                
            logger.info("[PERFORMANCE_DB] Performance database initialized")
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_DB] Database initialization failed: {e}")
            raise
    
    async def log_performance_window(self, window: StrategyPerformanceWindow):
        """Log performance window data"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO performance_windows
                    (timestamp, strategy_name, window_days, total_return, sharpe_ratio,
                     max_drawdown, win_rate, performance_score, trend_strength)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    window.strategy_name,
                    window.window_days,
                    window.total_return,
                    window.sharpe_ratio,
                    window.max_drawdown,
                    window.win_rate,
                    window.performance_score(),
                    window.trend_strength
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[PERFORMANCE_DB] Failed to log performance window: {e}")
    
    async def log_analysis_result(self, result: PerformanceAnalysisResult):
        """Log performance analysis result"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO performance_analysis
                    (timestamp, strategy_name, current_regime, rebalance_signal,
                     recommended_change, alpha, beta, quality_score, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.timestamp.isoformat(),
                    result.strategy_name,
                    result.current_regime.value,
                    result.rebalance_signal.value,
                    result.recommended_allocation_change,
                    result.alpha,
                    result.beta,
                    result.strategy_quality_score,
                    result.recommendation_confidence
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"[PERFORMANCE_DB] Failed to log analysis result: {e}")

class PerformanceBasedRebalancer:
    """
    Advanced rebalancer that adjusts allocations based on multi-timeframe performance analysis
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        
        # Initialize database
        import os
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.db = PerformanceDatabase(f"{log_dir}/performance_rebalancer.db")
        
        # Performance tracking
        self.strategy_returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=720))  # 30 days of hourly data
        self.strategy_trades: Dict[str, List[Dict]] = defaultdict(list)
        self.benchmark_returns: deque = deque(maxlen=720)  # Benchmark returns
        
        # Rebalancing parameters
        self.rebalance_thresholds = {
            RebalanceSignal.STRONG_BUY: 0.05,    # 5% allocation increase
            RebalanceSignal.BUY: 0.02,           # 2% allocation increase
            RebalanceSignal.HOLD: 0.0,           # No change
            RebalanceSignal.SELL: -0.02,         # 2% allocation decrease
            RebalanceSignal.STRONG_SELL: -0.05   # 5% allocation decrease
        }
        
        # Analysis history
        self.analysis_history: Dict[str, List[PerformanceAnalysisResult]] = defaultdict(list)
        self.last_analysis_time = datetime.now()
        
        # Performance benchmarks
        self.performance_percentiles = {
            'excellent': 90,  # Top 10%
            'good': 70,       # Top 30%
            'average': 50,    # Median
            'poor': 30,       # Bottom 30%
            'terrible': 10    # Bottom 10%
        }
        
        logger.info("[PERFORMANCE_REBALANCER] Performance-Based Rebalancer initialized")
    
    async def initialize(self):
        """Initialize the performance-based rebalancer and database"""
        try:
            await self.db.initialize()
            logger.info("[PERFORMANCE_REBALANCER] Performance-Based Rebalancer initialized")
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Failed to initialize: {e}")
            raise
            
    async def start(self):
        """Start the performance-based rebalancer"""
        try:
            await self.db.initialize()
            logger.info("[PERFORMANCE_REBALANCER] Performance-Based Rebalancer started")
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Failed to start: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the performance-based rebalancer"""
        await self.stop()
        
    async def stop(self):
        """Stop the performance-based rebalancer"""
        try:
            logger.info("[PERFORMANCE_REBALANCER] Performance-Based Rebalancer stopped")
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error during shutdown: {e}")
    
    def update_strategy_return(self, strategy_name: str, return_value: float, trade_data: Optional[Dict] = None):
        """Update strategy return data"""
        try:
            # Store return with timestamp
            timestamped_return = {
                'return': return_value,
                'timestamp': datetime.now()
            }
            
            self.strategy_returns[strategy_name].append(timestamped_return)
            
            # Store trade data if provided
            if trade_data:
                trade_data['timestamp'] = datetime.now()
                self.strategy_trades[strategy_name].append(trade_data)
                
                # Keep only last 1000 trades per strategy
                if len(self.strategy_trades[strategy_name]) > 1000:
                    self.strategy_trades[strategy_name] = self.strategy_trades[strategy_name][-1000:]
            
            logger.debug(f"[PERFORMANCE_REBALANCER] Updated {strategy_name} return: {return_value:.4f}")
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error updating strategy return: {e}")
    
    def update_benchmark_return(self, return_value: float):
        """Update benchmark return (e.g., SOL price return)"""
        try:
            timestamped_return = {
                'return': return_value,
                'timestamp': datetime.now()
            }
            
            self.benchmark_returns.append(timestamped_return)
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error updating benchmark return: {e}")
    
    async def analyze_strategy_performance(self, strategy_name: str) -> Optional[PerformanceAnalysisResult]:
        """Analyze strategy performance across multiple timeframes"""
        try:
            if strategy_name not in self.strategy_returns or len(self.strategy_returns[strategy_name]) < 24:
                logger.warning(f"[PERFORMANCE_REBALANCER] Insufficient data for {strategy_name}")
                return None
            
            # Calculate performance windows
            window_1d = self._calculate_performance_window(strategy_name, 1)
            window_7d = self._calculate_performance_window(strategy_name, 7)
            window_30d = self._calculate_performance_window(strategy_name, 30)
            
            # Determine performance regime
            regime, regime_confidence = self._determine_performance_regime(window_1d, window_7d, window_30d)
            
            # Calculate momentum indicators
            momentum_1d = self._calculate_momentum(strategy_name, 1)
            momentum_7d = self._calculate_momentum(strategy_name, 7)
            momentum_30d = self._calculate_momentum(strategy_name, 30)
            
            # Calculate alpha and beta
            alpha, beta, tracking_error = self._calculate_alpha_beta(strategy_name)
            
            # Generate rebalancing signal
            rebalance_signal, allocation_change, confidence = self._generate_rebalance_signal(
                window_1d, window_7d, window_30d, regime, momentum_1d, momentum_7d, momentum_30d
            )
            
            # Calculate strategy quality score
            quality_score = self._calculate_strategy_quality_score(
                window_1d, window_7d, window_30d, alpha, beta, tracking_error
            )
            
            # Create analysis result
            result = PerformanceAnalysisResult(
                strategy_name=strategy_name,
                window_1d=window_1d,
                window_7d=window_7d,
                window_30d=window_30d,
                current_regime=regime,
                regime_confidence=regime_confidence,
                rebalance_signal=rebalance_signal,
                recommended_allocation_change=allocation_change,
                alpha=alpha,
                beta=beta,
                tracking_error=tracking_error,
                momentum_1d=momentum_1d,
                momentum_7d=momentum_7d,
                momentum_30d=momentum_30d,
                strategy_quality_score=quality_score,
                recommendation_confidence=confidence
            )
            
            # Store analysis result
            self.analysis_history[strategy_name].append(result)
            
            # Keep only last 100 analysis results
            if len(self.analysis_history[strategy_name]) > 100:
                self.analysis_history[strategy_name] = self.analysis_history[strategy_name][-100:]
            
            # Log to database
            await self.db.log_analysis_result(result)
            
            logger.info(f"[PERFORMANCE_REBALANCER] {strategy_name}: {regime.value}, "
                       f"Signal: {rebalance_signal.value}, Change: {allocation_change:+.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error analyzing {strategy_name}: {e}")
            return None
    
    def _calculate_performance_window(self, strategy_name: str, days: int) -> StrategyPerformanceWindow:
        """Calculate performance metrics for a specific time window"""
        try:
            # Get returns for the time window
            cutoff_time = datetime.now() - timedelta(days=days)
            returns_data = self.strategy_returns[strategy_name]
            
            # Filter returns by time window
            window_returns = []
            for return_data in returns_data:
                if return_data['timestamp'] >= cutoff_time:
                    window_returns.append(return_data['return'])
            
            if len(window_returns) < max(1, days * 2):  # At least 2 returns per day
                # Return default window with zero values
                return StrategyPerformanceWindow(
                    strategy_name=strategy_name,
                    window_days=days,
                    start_time=cutoff_time,
                    end_time=datetime.now()
                )
            
            # Calculate basic metrics
            total_return = (1 + np.array(window_returns)).prod() - 1
            annualized_return = total_return * (365 / days)
            volatility = np.std(window_returns) * np.sqrt(365)
            
            # Calculate risk metrics
            max_drawdown = self._calculate_max_drawdown(window_returns)
            var_95 = np.percentile(window_returns, 5)
            
            # Risk-adjusted metrics
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            downside_returns = [r for r in window_returns if r < 0]
            downside_vol = np.std(downside_returns) * np.sqrt(365) if downside_returns else volatility
            sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trading metrics from trade data
            trades_data = [t for t in self.strategy_trades[strategy_name] 
                          if t['timestamp'] >= cutoff_time]
            
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade_return = 0.0
            
            if trades_data:
                winning_trades = [t for t in trades_data if t.get('pnl', 0) > 0]
                win_rate = len(winning_trades) / len(trades_data)
                
                total_profit = sum(t.get('pnl', 0) for t in winning_trades)
                total_loss = abs(sum(t.get('pnl', 0) for t in trades_data if t.get('pnl', 0) < 0))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
                
                avg_trade_return = np.mean([t.get('return', 0) for t in trades_data])
            
            # Performance consistency (inverse of return volatility)
            performance_consistency = max(0, 1 - volatility / 0.5) if volatility < 0.5 else 0
            
            # Trend strength (linear regression slope)
            if len(window_returns) >= 3:
                x = np.arange(len(window_returns))
                trend_slope = np.polyfit(x, window_returns, 1)[0]
                trend_strength = np.tanh(trend_slope * 100)  # Normalize between -1 and 1
            else:
                trend_strength = 0.0
            
            window = StrategyPerformanceWindow(
                strategy_name=strategy_name,
                window_days=days,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                max_drawdown=max_drawdown,
                var_95=var_95,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_return=avg_trade_return,
                trades_count=len(trades_data),
                performance_consistency=performance_consistency,
                trend_strength=trend_strength,
                start_time=cutoff_time,
                end_time=datetime.now()
            )
            
            # Log to database
            asyncio.create_task(self.db.log_performance_window(window))
            
            return window
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error calculating performance window: {e}")
            return StrategyPerformanceWindow(strategy_name=strategy_name, window_days=days)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / running_max
            return float(np.max(drawdown))
            
        except Exception:
            return 0.0
    
    def _determine_performance_regime(
        self, 
        window_1d: StrategyPerformanceWindow,
        window_7d: StrategyPerformanceWindow,
        window_30d: StrategyPerformanceWindow
    ) -> Tuple[PerformanceRegime, float]:
        """Determine current performance regime"""
        try:
            # Calculate regime indicators
            short_momentum = window_1d.trend_strength
            medium_momentum = window_7d.trend_strength
            long_momentum = window_30d.trend_strength
            
            volatility_7d = window_7d.volatility
            consistency_7d = window_7d.performance_consistency
            
            # Regime classification logic
            confidence = 0.5  # Default confidence
            
            # Strong upward momentum across all timeframes
            if short_momentum > 0.5 and medium_momentum > 0.3 and long_momentum > 0.2:
                confidence = min(1.0, abs(short_momentum) + abs(medium_momentum) + abs(long_momentum)) / 3
                return PerformanceRegime.STRONG_MOMENTUM, confidence
            
            # Consistent downward momentum
            elif short_momentum < -0.3 and medium_momentum < -0.2:
                confidence = min(1.0, abs(short_momentum) + abs(medium_momentum)) / 2
                return PerformanceRegime.WEAK_MOMENTUM, confidence
            
            # High volatility, inconsistent performance
            elif volatility_7d > 0.3 or consistency_7d < 0.3:
                confidence = min(1.0, volatility_7d / 0.5)
                return PerformanceRegime.VOLATILE, confidence
            
            # Mean reverting behavior (short-term opposite to long-term)
            elif (short_momentum > 0.2 and long_momentum < -0.1) or (short_momentum < -0.2 and long_momentum > 0.1):
                confidence = min(1.0, abs(short_momentum - long_momentum) / 2)
                return PerformanceRegime.MEAN_REVERTING, confidence
            
            # Stable performance
            else:
                confidence = consistency_7d
                return PerformanceRegime.STABLE, confidence
                
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error determining performance regime: {e}")
            return PerformanceRegime.STABLE, 0.5
    
    def _calculate_momentum(self, strategy_name: str, days: int) -> float:
        """Calculate momentum over specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            returns_data = self.strategy_returns[strategy_name]
            
            # Get recent returns
            recent_returns = []
            for return_data in returns_data:
                if return_data['timestamp'] >= cutoff_time:
                    recent_returns.append(return_data['return'])
            
            if len(recent_returns) < 3:
                return 0.0
            
            # Calculate momentum as trend strength
            x = np.arange(len(recent_returns))
            slope = np.polyfit(x, recent_returns, 1)[0]
            
            # Normalize momentum between -1 and 1
            momentum = np.tanh(slope * 100 * days)
            return float(momentum)
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error calculating momentum: {e}")
            return 0.0
    
    def _calculate_alpha_beta(self, strategy_name: str) -> Tuple[float, float, float]:
        """Calculate alpha, beta, and tracking error vs benchmark"""
        try:
            if not self.benchmark_returns or len(self.benchmark_returns) < 30:
                return 0.0, 1.0, 0.0  # Default values
            
            # Get aligned returns
            cutoff_time = datetime.now() - timedelta(days=30)
            strategy_data = self.strategy_returns[strategy_name]
            
            strategy_returns = []
            benchmark_returns = []
            
            # Get strategy returns in time window
            for return_data in strategy_data:
                if return_data['timestamp'] >= cutoff_time:
                    strategy_returns.append(return_data['return'])
            
            # Get benchmark returns in same time window  
            for return_data in self.benchmark_returns:
                if return_data['timestamp'] >= cutoff_time:
                    benchmark_returns.append(return_data['return'])
            
            # Align lengths
            min_length = min(len(strategy_returns), len(benchmark_returns))
            if min_length < 10:
                return 0.0, 1.0, 0.0
            
            strategy_returns = strategy_returns[-min_length:]
            benchmark_returns = benchmark_returns[-min_length:]
            
            # Calculate beta (covariance / variance)
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # Calculate alpha (average excess return)
            strategy_mean = np.mean(strategy_returns)
            benchmark_mean = np.mean(benchmark_returns)
            alpha = strategy_mean - beta * benchmark_mean
            
            # Calculate tracking error (volatility of excess returns)
            excess_returns = np.array(strategy_returns) - beta * np.array(benchmark_returns)
            tracking_error = np.std(excess_returns)
            
            return float(alpha), float(beta), float(tracking_error)
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error calculating alpha/beta: {e}")
            return 0.0, 1.0, 0.0
    
    def _generate_rebalance_signal(
        self,
        window_1d: StrategyPerformanceWindow,
        window_7d: StrategyPerformanceWindow,
        window_30d: StrategyPerformanceWindow,
        regime: PerformanceRegime,
        momentum_1d: float,
        momentum_7d: float,
        momentum_30d: float
    ) -> Tuple[RebalanceSignal, float, float]:
        """Generate rebalancing signal and recommended allocation change"""
        try:
            # Calculate performance scores
            score_1d = window_1d.performance_score()
            score_7d = window_7d.performance_score()
            score_30d = window_30d.performance_score()
            
            # Weight scores by recency (more weight to recent performance)
            weighted_score = (score_1d * 0.5 + score_7d * 0.3 + score_30d * 0.2)
            
            # Regime adjustments
            regime_multiplier = 1.0
            if regime == PerformanceRegime.STRONG_MOMENTUM:
                regime_multiplier = 1.3  # Boost momentum strategies
            elif regime == PerformanceRegime.WEAK_MOMENTUM:
                regime_multiplier = 0.7  # Reduce weak strategies
            elif regime == PerformanceRegime.VOLATILE:
                regime_multiplier = 0.8  # Reduce volatile strategies
            elif regime == PerformanceRegime.MEAN_REVERTING:
                regime_multiplier = 0.9  # Slight reduction for mean reverting
            
            adjusted_score = weighted_score * regime_multiplier
            
            # Generate signal based on adjusted score
            if adjusted_score >= 8:
                signal = RebalanceSignal.STRONG_BUY
            elif adjusted_score >= 4:
                signal = RebalanceSignal.BUY
            elif adjusted_score >= -4:
                signal = RebalanceSignal.HOLD
            elif adjusted_score >= -8:
                signal = RebalanceSignal.SELL
            else:
                signal = RebalanceSignal.STRONG_SELL
            
            # Calculate allocation change
            base_change = self.rebalance_thresholds[signal]
            
            # Adjust change based on momentum and score strength
            momentum_avg = (momentum_1d * 0.5 + momentum_7d * 0.3 + momentum_30d * 0.2)
            momentum_adjustment = momentum_avg * 0.02  # ±2% based on momentum
            
            allocation_change = base_change + momentum_adjustment
            
            # Calculate confidence based on consistency of signals
            confidence_factors = []
            
            # Score consistency
            score_std = np.std([score_1d, score_7d, score_30d])
            score_consistency = max(0, 1 - score_std / 10)
            confidence_factors.append(score_consistency)
            
            # Momentum alignment
            momentum_alignment = 1 - np.std([momentum_1d, momentum_7d, momentum_30d]) / 2
            confidence_factors.append(max(0, momentum_alignment))
            
            # Performance consistency
            perf_consistency = (window_1d.performance_consistency + 
                              window_7d.performance_consistency + 
                              window_30d.performance_consistency) / 3
            confidence_factors.append(perf_consistency)
            
            confidence = np.mean(confidence_factors)
            
            return signal, allocation_change, confidence
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error generating rebalance signal: {e}")
            return RebalanceSignal.HOLD, 0.0, 0.5
    
    def _calculate_strategy_quality_score(
        self,
        window_1d: StrategyPerformanceWindow,
        window_7d: StrategyPerformanceWindow,
        window_30d: StrategyPerformanceWindow,
        alpha: float,
        beta: float,
        tracking_error: float
    ) -> float:
        """Calculate overall strategy quality score (0-100)"""
        try:
            quality_components = []
            
            # Risk-adjusted returns component (0-30 points)
            avg_sharpe = (window_1d.sharpe_ratio + window_7d.sharpe_ratio + window_30d.sharpe_ratio) / 3
            sharpe_score = min(30, max(0, avg_sharpe * 15 + 15))
            quality_components.append(sharpe_score)
            
            # Consistency component (0-20 points)
            avg_consistency = (window_1d.performance_consistency + 
                             window_7d.performance_consistency + 
                             window_30d.performance_consistency) / 3
            consistency_score = avg_consistency * 20
            quality_components.append(consistency_score)
            
            # Win rate component (0-15 points)
            avg_win_rate = (window_1d.win_rate + window_7d.win_rate + window_30d.win_rate) / 3
            win_rate_score = avg_win_rate * 15
            quality_components.append(win_rate_score)
            
            # Alpha component (0-15 points)
            alpha_score = min(15, max(0, alpha * 100 + 7.5))
            quality_components.append(alpha_score)
            
            # Drawdown component (0-10 points, penalty)
            avg_drawdown = (window_1d.max_drawdown + window_7d.max_drawdown + window_30d.max_drawdown) / 3
            drawdown_penalty = min(10, avg_drawdown * 50)
            quality_components.append(-drawdown_penalty)
            
            # Tracking error component (0-10 points, penalty for high tracking error)
            tracking_penalty = min(10, tracking_error * 50)
            quality_components.append(-tracking_penalty)
            
            total_score = sum(quality_components)
            return max(0, min(100, total_score))
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error calculating quality score: {e}")
            return 50.0
    
    async def get_rebalancing_recommendations(
        self, 
        current_allocations: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Get rebalancing recommendations for all strategies"""
        try:
            recommendations = {}
            
            for strategy_name in current_allocations.keys():
                analysis = await self.analyze_strategy_performance(strategy_name)
                
                if analysis:
                    current_allocation = current_allocations[strategy_name]
                    recommended_allocation = current_allocation + analysis.recommended_allocation_change
                    
                    # Ensure allocation stays within reasonable bounds
                    recommended_allocation = max(0.01, min(0.50, recommended_allocation))
                    
                    recommendations[strategy_name] = {
                        'current_allocation': current_allocation,
                        'recommended_allocation': recommended_allocation,
                        'change': recommended_allocation - current_allocation,
                        'signal': analysis.rebalance_signal.value,
                        'confidence': analysis.recommendation_confidence,
                        'regime': analysis.current_regime.value,
                        'quality_score': analysis.strategy_quality_score,
                        'analysis_data': analysis.to_dict()
                    }
            
            logger.info(f"[PERFORMANCE_REBALANCER] Generated recommendations for {len(recommendations)} strategies")
            return recommendations
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error getting recommendations: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance rebalancer summary"""
        try:
            total_strategies = len(self.strategy_returns)
            active_strategies = len([s for s, returns in self.strategy_returns.items() if len(returns) > 0])
            
            # Get latest analysis results
            latest_analysis = {}
            for strategy_name, analyses in self.analysis_history.items():
                if analyses:
                    latest_analysis[strategy_name] = analyses[-1].to_dict()
            
            return {
                'total_strategies': total_strategies,
                'active_strategies': active_strategies,
                'total_data_points': sum(len(returns) for returns in self.strategy_returns.values()),
                'benchmark_data_points': len(self.benchmark_returns),
                'latest_analysis': latest_analysis,
                'last_analysis_time': self.last_analysis_time.isoformat(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[PERFORMANCE_REBALANCER] Error generating summary: {e}")
            return {'error': str(e)}