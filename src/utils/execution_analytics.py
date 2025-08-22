#!/usr/bin/env python3
"""
Trade Execution Analytics
Tracks execution timing, slippage, and performance metrics for trades
"""
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class ExecutionMetrics:
    """Trade execution performance metrics"""
    token_address: str
    trade_id: str
    execution_start_time: datetime
    execution_end_time: datetime
    expected_price: float
    actual_price: float
    slippage_percentage: float
    execution_time_ms: float
    gas_fee: float
    success: bool
    error_message: Optional[str] = None
    network_latency_ms: Optional[float] = None
    confirmation_time_ms: Optional[float] = None

@dataclass
class SlippageAnalysis:
    """Slippage analysis results"""
    average_slippage: float
    median_slippage: float
    max_slippage: float
    min_slippage: float
    slippage_std_dev: float
    high_slippage_trades_count: int
    total_trades_analyzed: int

@dataclass
class ExecutionPerformance:
    """Overall execution performance summary"""
    average_execution_time_ms: float
    median_execution_time_ms: float
    success_rate: float
    total_trades: int
    failed_trades: int
    average_gas_fee: float
    slippage_analysis: SlippageAnalysis

class ExecutionAnalytics:
    """Trade execution analytics and monitoring system"""
    
    def __init__(self):
        self.analytics_dir = Path("analytics")
        self.analytics_dir.mkdir(exist_ok=True)
        
        self.execution_file = self.analytics_dir / "execution_metrics.json"
        self.slippage_file = self.analytics_dir / "slippage_analysis.json"
        
        # Execution tracking
        self.execution_metrics: List[ExecutionMetrics] = []
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data
        self._load_execution_data()
        
        # Alert thresholds
        self.high_slippage_threshold = 2.0  # 2%
        self.slow_execution_threshold = 5000  # 5 seconds
        
    def _load_execution_data(self) -> None:
        """Load existing execution metrics"""
        try:
            if self.execution_file.exists():
                with open(self.execution_file, 'r') as f:
                    data = json.load(f)
                    
                for item in data.get('executions', []):
                    metrics = ExecutionMetrics(
                        token_address=item['token_address'],
                        trade_id=item['trade_id'],
                        execution_start_time=datetime.fromisoformat(item['execution_start_time']),
                        execution_end_time=datetime.fromisoformat(item['execution_end_time']),
                        expected_price=item['expected_price'],
                        actual_price=item['actual_price'],
                        slippage_percentage=item['slippage_percentage'],
                        execution_time_ms=item['execution_time_ms'],
                        gas_fee=item['gas_fee'],
                        success=item['success'],
                        error_message=item.get('error_message'),
                        network_latency_ms=item.get('network_latency_ms'),
                        confirmation_time_ms=item.get('confirmation_time_ms')
                    )
                    self.execution_metrics.append(metrics)
                    
        except Exception as e:
            logger.warning(f"Failed to load execution data: {e}")
    
    def _save_execution_data(self) -> None:
        """Save execution metrics to file"""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'total_executions': len(self.execution_metrics),
                'executions': []
            }
            
            for metrics in self.execution_metrics:
                metrics_dict = asdict(metrics)
                # Convert datetime objects to ISO strings
                metrics_dict['execution_start_time'] = metrics.execution_start_time.isoformat()
                metrics_dict['execution_end_time'] = metrics.execution_end_time.isoformat()
                data['executions'].append(metrics_dict)
            
            with open(self.execution_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save execution data: {e}")
    
    def start_execution_tracking(self, trade_id: str, token_address: str, 
                               expected_price: float) -> str:
        """Start tracking trade execution"""
        execution_key = f"{trade_id}_{int(time.time())}"
        
        self.active_executions[execution_key] = {
            'trade_id': trade_id,
            'token_address': token_address,
            'expected_price': expected_price,
            'start_time': time.time(),
            'start_datetime': datetime.now()
        }
        
        logger.info(f"Started execution tracking for {trade_id} on {token_address[:8]}...")
        return execution_key
    
    def complete_execution_tracking(self, execution_key: str, actual_price: float,
                                  gas_fee: float, success: bool, 
                                  error_message: Optional[str] = None,
                                  network_latency_ms: Optional[float] = None,
                                  confirmation_time_ms: Optional[float] = None) -> None:
        """Complete execution tracking and record metrics"""
        if execution_key not in self.active_executions:
            logger.warning(f"Execution key {execution_key} not found in active tracking")
            return
        
        execution_data = self.active_executions.pop(execution_key)
        end_time = time.time()
        end_datetime = datetime.now()
        
        # Calculate metrics
        execution_time_ms = (end_time - execution_data['start_time']) * 1000
        expected_price = execution_data['expected_price']
        slippage_percentage = ((actual_price - expected_price) / expected_price) * 100 if expected_price > 0 else 0
        
        # Create metrics record
        metrics = ExecutionMetrics(
            token_address=execution_data['token_address'],
            trade_id=execution_data['trade_id'],
            execution_start_time=execution_data['start_datetime'],
            execution_end_time=end_datetime,
            expected_price=expected_price,
            actual_price=actual_price,
            slippage_percentage=slippage_percentage,
            execution_time_ms=execution_time_ms,
            gas_fee=gas_fee,
            success=success,
            error_message=error_message,
            network_latency_ms=network_latency_ms,
            confirmation_time_ms=confirmation_time_ms
        )
        
        self.execution_metrics.append(metrics)
        self._save_execution_data()
        
        # Log execution results
        if success:
            logger.info(f"Execution completed - Time: {execution_time_ms:.1f}ms, "
                       f"Slippage: {slippage_percentage:.2f}%, Gas: {gas_fee:.6f} SOL")
        else:
            logger.warning(f"Execution failed - Time: {execution_time_ms:.1f}ms, "
                          f"Error: {error_message}")
        
        # Check for alerts
        self._check_execution_alerts(metrics)
    
    def _check_execution_alerts(self, metrics: ExecutionMetrics) -> None:
        """Check execution metrics for alert conditions"""
        alerts = []
        
        # High slippage alert
        if abs(metrics.slippage_percentage) > self.high_slippage_threshold:
            alerts.append(f"High slippage: {metrics.slippage_percentage:.2f}%")
        
        # Slow execution alert
        if metrics.execution_time_ms > self.slow_execution_threshold:
            alerts.append(f"Slow execution: {metrics.execution_time_ms:.1f}ms")
        
        # Failed execution alert
        if not metrics.success:
            alerts.append(f"Execution failed: {metrics.error_message}")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"[EXECUTION ALERT] {alert} for {metrics.token_address[:8]}...")
    
    def get_slippage_analysis(self, days: int = 30) -> SlippageAnalysis:
        """Analyze slippage over specified period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_executions = [
            m for m in self.execution_metrics 
            if m.execution_end_time > cutoff_date and m.success
        ]
        
        if not recent_executions:
            return SlippageAnalysis(0, 0, 0, 0, 0, 0, 0)
        
        slippages = [abs(m.slippage_percentage) for m in recent_executions]
        
        return SlippageAnalysis(
            average_slippage=statistics.mean(slippages),
            median_slippage=statistics.median(slippages),
            max_slippage=max(slippages),
            min_slippage=min(slippages),
            slippage_std_dev=statistics.stdev(slippages) if len(slippages) > 1 else 0,
            high_slippage_trades_count=len([s for s in slippages if s > self.high_slippage_threshold]),
            total_trades_analyzed=len(recent_executions)
        )
    
    def get_execution_performance(self, days: int = 30) -> ExecutionPerformance:
        """Get comprehensive execution performance analysis"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_executions = [
            m for m in self.execution_metrics 
            if m.execution_end_time > cutoff_date
        ]
        
        if not recent_executions:
            return ExecutionPerformance(0, 0, 0, 0, 0, 0, SlippageAnalysis(0, 0, 0, 0, 0, 0, 0))
        
        successful_executions = [m for m in recent_executions if m.success]
        execution_times = [m.execution_time_ms for m in recent_executions]
        gas_fees = [m.gas_fee for m in successful_executions]
        
        return ExecutionPerformance(
            average_execution_time_ms=statistics.mean(execution_times),
            median_execution_time_ms=statistics.median(execution_times),
            success_rate=(len(successful_executions) / len(recent_executions)) * 100,
            total_trades=len(recent_executions),
            failed_trades=len(recent_executions) - len(successful_executions),
            average_gas_fee=statistics.mean(gas_fees) if gas_fees else 0,
            slippage_analysis=self.get_slippage_analysis(days)
        )
    
    def get_recent_execution_summary(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get summary of recent executions"""
        recent = sorted(self.execution_metrics, key=lambda x: x.execution_end_time, reverse=True)[:count]
        
        summary = []
        for metrics in recent:
            summary.append({
                'timestamp': metrics.execution_end_time.isoformat(),
                'token': metrics.token_address[:8] + "...",
                'execution_time_ms': metrics.execution_time_ms,
                'slippage_pct': metrics.slippage_percentage,
                'gas_fee': metrics.gas_fee,
                'success': metrics.success,
                'error': metrics.error_message if not metrics.success else None
            })
        
        return summary
    
    def generate_execution_report(self) -> str:
        """Generate execution analytics report"""
        performance = self.get_execution_performance(30)
        recent_summary = self.get_recent_execution_summary(5)
        
        report = []
        report.append("TRADE EXECUTION ANALYTICS REPORT")
        report.append("=" * 50)
        report.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance summary
        report.append("EXECUTION PERFORMANCE (30 days)")
        report.append("-" * 30)
        report.append(f"Total Trades: {performance.total_trades}")
        report.append(f"Success Rate: {performance.success_rate:.1f}%")
        report.append(f"Failed Trades: {performance.failed_trades}")
        report.append(f"Average Execution Time: {performance.average_execution_time_ms:.1f}ms")
        report.append(f"Median Execution Time: {performance.median_execution_time_ms:.1f}ms")
        report.append(f"Average Gas Fee: {performance.average_gas_fee:.6f} SOL")
        report.append("")
        
        # Slippage analysis
        slippage = performance.slippage_analysis
        report.append("SLIPPAGE ANALYSIS")
        report.append("-" * 20)
        report.append(f"Trades Analyzed: {slippage.total_trades_analyzed}")
        report.append(f"Average Slippage: {slippage.average_slippage:.3f}%")
        report.append(f"Median Slippage: {slippage.median_slippage:.3f}%")
        report.append(f"Max Slippage: {slippage.max_slippage:.3f}%")
        report.append(f"High Slippage Trades: {slippage.high_slippage_trades_count}")
        report.append(f"Slippage Std Dev: {slippage.slippage_std_dev:.3f}%")
        report.append("")
        
        # Recent executions
        report.append("RECENT EXECUTIONS (Last 5)")
        report.append("-" * 30)
        for i, execution in enumerate(recent_summary, 1):
            status = "SUCCESS" if execution['success'] else f"FAILED ({execution['error']})"
            report.append(f"{i}. {execution['token']} - {execution['execution_time_ms']:.1f}ms - "
                         f"{execution['slippage_pct']:.2f}% slip - {status}")
        
        return "\n".join(report)

# Global execution analytics instance
execution_analytics = ExecutionAnalytics()

def start_execution_tracking(trade_id: str, token_address: str, expected_price: float) -> str:
    """Start tracking trade execution timing and performance"""
    return execution_analytics.start_execution_tracking(trade_id, token_address, expected_price)

def complete_execution_tracking(execution_key: str, actual_price: float, gas_fee: float, 
                               success: bool, error_message: Optional[str] = None,
                               network_latency_ms: Optional[float] = None,
                               confirmation_time_ms: Optional[float] = None) -> None:
    """Complete execution tracking and record metrics"""
    execution_analytics.complete_execution_tracking(
        execution_key, actual_price, gas_fee, success, error_message,
        network_latency_ms, confirmation_time_ms
    )

def get_execution_report() -> str:
    """Get comprehensive execution analytics report"""
    return execution_analytics.generate_execution_report()