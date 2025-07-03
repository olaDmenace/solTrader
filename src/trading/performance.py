import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import json

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    token_address: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    exit_reason: str

@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    roi: float = 0.0


class PerformanceMonitor:
    def __init__(self, settings):
        self.settings = settings
        self.trades: List[Trade] = []
        self.performance_metrics = PerformanceMetrics()
        self.current_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.start_time = datetime.now()

    async def record_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self._update_metrics()

    def _update_metrics(self) -> None:
        try:
            if not self.trades:
                return

            # Basic metrics
            pnls = [t.pnl for t in self.trades]
            self.current_equity = sum(pnls)
            self.peak_equity = max(self.peak_equity, self.current_equity)

            self.performance_metrics = {
                'total_trades': len(self.trades),
                'win_rate': self._calculate_win_rate(),
                'avg_profit': np.mean([t.pnl for t in self.trades if t.pnl > 0]) if any(t.pnl > 0 for t in self.trades) else 0,
                'avg_loss': abs(np.mean([t.pnl for t in self.trades if t.pnl < 0])) if any(t.pnl < 0 for t in self.trades) else 0,
                'profit_factor': self._calculate_profit_factor(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'max_drawdown': self._calculate_max_drawdown(),
                'roi': (self.current_equity / self.settings.INITIAL_CAPITAL - 1) * 100
            }

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def _calculate_win_rate(self) -> float:
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        return (winning_trades / len(self.trades)) * 100 if self.trades else 0

    def _calculate_profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else 0

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if not self.trades:
            return 0
        
        daily_returns = self._get_daily_returns()
        if not daily_returns:
            return 0
            
        returns_array = np.array(daily_returns)
        excess_returns = returns_array - (risk_free_rate / 365)
        return np.sqrt(365) * (np.mean(excess_returns) / np.std(excess_returns)) if np.std(excess_returns) != 0 else 0

    def _calculate_max_drawdown(self) -> float:
        peak = float('-inf')
        max_dd = 0
        
        for trade in self.trades:
            equity = sum(t.pnl for t in self.trades if t.exit_time <= trade.exit_time)
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            
        return max_dd * 100

    def _get_daily_returns(self) -> List[float]:
        daily_pnl = {}
        
        for trade in self.trades:
            date = trade.exit_time.date()
            daily_pnl[date] = daily_pnl.get(date, 0) + trade.pnl

        starting_capital = self.settings.INITIAL_CAPITAL
        returns = []
        
        for date, pnl in sorted(daily_pnl.items()):
            returns.append(pnl / starting_capital)
            starting_capital += pnl
            
        return returns

    def get_performance_summary(self) -> Dict[str, any]:
        return {
            'metrics': self.performance_metrics,
            'equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'trading_days': (datetime.now() - self.start_time).days,
            'recent_trades': [self._format_trade(t) for t in self.trades[-5:]]
        }

    def _format_trade(self, trade: Trade) -> Dict[str, any]:
        return {
            'token': trade.token_address,
            'pnl': trade.pnl,
            'roi': (trade.exit_price / trade.entry_price - 1) * 100,
            'holding_time': (trade.exit_time - trade.entry_time).total_seconds() / 3600,
            'exit_reason': trade.exit_reason
        }

    def save_performance_data(self, filepath: str) -> None:
        try:
            data = {
                'summary': self.get_performance_summary(),
                'trades': [self._format_trade(t) for t in self.trades],
                'timestamp': datetime.now().isoformat()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")