#!/usr/bin/env python3
"""
Performance Reporting System for SolTrader
Generates human-readable performance reports and summaries
"""
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from .performance_analytics import performance_analytics, DailyPerformance, WeeklyPerformance

logger = logging.getLogger(__name__)

class PerformanceReporter:
    """Generates performance reports in various formats"""
    
    def __init__(self, output_directory: str = "reports"):
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_daily_report(self, date_str: str = None, save_to_file: bool = True) -> str:
        """Generate a daily performance report"""
        if date_str is None:
            date_str = date.today().isoformat()
        
        daily_perf = performance_analytics.get_daily_summary(date_str)
        
        if daily_perf is None:
            return f"No trading data available for {date_str}"
        
        # Generate report content
        report_lines = [
            f"SOLTRADER DAILY PERFORMANCE REPORT",
            f"=" * 50,
            f"Date: {date_str}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"TRADING SUMMARY",
            f"-" * 20,
            f"Total Trades: {daily_perf.total_trades}",
            f"Profitable Trades: {daily_perf.profitable_trades}",
            f"Losing Trades: {daily_perf.losing_trades}",
            f"Win Rate: {daily_perf.win_rate:.1f}%",
            f"",
            f"PROFIT & LOSS",
            f"-" * 20,
            f"Total P&L (USD): ${daily_perf.total_profit_loss_usd:.2f}",
            f"Total P&L (SOL): {daily_perf.total_profit_loss_sol:.6f} SOL",
            f"Best Trade: +${daily_perf.best_trade_profit:.2f}",
            f"Worst Trade: -${daily_perf.worst_trade_loss:.2f}",
            f"Average Win: ${daily_perf.average_win:.2f}",
            f"Average Loss: ${daily_perf.average_loss:.2f}",
            f"",
            f"TRADING METRICS",
            f"-" * 20,
            f"Average Trade Duration: {daily_perf.average_trade_duration:.1f} minutes",
            f"Total Volume: ${daily_perf.total_volume_usd:.2f}",
            f"Unique Tokens: {daily_perf.unique_tokens_traded}",
            f"Active Hours: {daily_perf.trading_hours_active:.1f}",
            f"",
            f"ADVANCED METRICS",
            f"-" * 20,
            f"Sharpe Ratio: {daily_perf.sharpe_ratio:.2f}",
            f"Max Drawdown: {daily_perf.max_drawdown:.2f}%",
            f"Profit Factor: {daily_perf.profit_factor:.2f}",
            f""
        ]
        
        # Performance assessment
        if daily_perf.total_profit_loss_usd > 0:
            assessment = "PROFITABLE DAY"
            if daily_perf.win_rate > 70:
                assessment += " - EXCELLENT PERFORMANCE"
            elif daily_perf.win_rate > 50:
                assessment += " - GOOD PERFORMANCE"
            else:
                assessment += " - LUCKY DAY (LOW WIN RATE)"
        else:
            assessment = "LOSING DAY"
            if daily_perf.win_rate < 30:
                assessment += " - POOR PERFORMANCE"
            else:
                assessment += " - UNLUCKY DAY (GOOD WIN RATE)"
        
        report_lines.extend([
            f"ASSESSMENT",
            f"-" * 20,
            f"{assessment}",
            f""
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            report_file = self.output_dir / f"daily_report_{date_str}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Daily report saved to {report_file}")
        
        return report_content
    
    def generate_weekly_report(self, week_start: str = None, save_to_file: bool = True) -> str:
        """Generate a weekly performance report"""
        if week_start is None:
            today = date.today()
            monday = today - timedelta(days=today.weekday())
            week_start = monday.isoformat()
        
        weekly_perf = performance_analytics.get_weekly_summary(week_start)
        
        if weekly_perf is None:
            return f"No trading data available for week starting {week_start}"
        
        # Generate report content
        report_lines = [
            f"SOLTRADER WEEKLY PERFORMANCE REPORT",
            f"=" * 55,
            f"Week: {week_start} to {weekly_perf.week_end}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"WEEKLY SUMMARY",
            f"-" * 20,
            f"Active Days: {weekly_perf.days_active}/7",
            f"Total Trades: {weekly_perf.total_trades}",
            f"Weekly Win Rate: {weekly_perf.win_rate:.1f}% (profitable days)",
            f"",
            f"WEEKLY P&L",
            f"-" * 20,
            f"Total Profit/Loss (USD): ${weekly_perf.total_profit_loss_usd:.2f}",
            f"Total Profit/Loss (SOL): {weekly_perf.total_profit_loss_sol:.6f} SOL",
            f"Best Day: +${weekly_perf.best_day_profit:.2f}",
            f"Worst Day: ${weekly_perf.worst_day_loss:.2f}",
            f"Average Daily P&L: ${weekly_perf.total_profit_loss_usd / weekly_perf.days_active:.2f}",
            f"",
            f"PERFORMANCE METRICS",
            f"-" * 20,
            f"Consistency Score: {weekly_perf.consistency_score:.2f}",
            f"Weekly Sharpe Ratio: {weekly_perf.weekly_sharpe:.2f}",
            f"",
            f"DAILY BREAKDOWN",
            f"-" * 20
        ]
        
        # Add daily breakdown
        for daily in weekly_perf.daily_performances:
            day_name = datetime.fromisoformat(daily.date).strftime('%A')
            status = "PROFIT" if daily.total_profit_loss_usd > 0 else "LOSS"
            report_lines.append(
                f"{day_name} ({daily.date}): ${daily.total_profit_loss_usd:+.2f} "
                f"({daily.total_trades} trades, {daily.win_rate:.0f}% win rate) - {status}"
            )
        
        report_lines.extend([
            f"",
            f"WEEKLY ASSESSMENT",
            f"-" * 20
        ])
        
        # Weekly assessment
        if weekly_perf.total_profit_loss_usd > 0:
            assessment = "PROFITABLE WEEK"
            if weekly_perf.consistency_score > 50:
                assessment += " - CONSISTENT PERFORMANCE"
            else:
                assessment += " - VOLATILE PERFORMANCE"
        else:
            assessment = "LOSING WEEK"
            if weekly_perf.days_active < 4:
                assessment += " - LIMITED ACTIVITY"
        
        profitable_days = len([d for d in weekly_perf.daily_performances if d.total_profit_loss_usd > 0])
        if profitable_days == weekly_perf.days_active:
            assessment += " - PERFECT WEEK (ALL DAYS PROFITABLE)"
        elif profitable_days >= weekly_perf.days_active * 0.8:
            assessment += " - STRONG WEEK"
        elif profitable_days >= weekly_perf.days_active * 0.6:
            assessment += " - GOOD WEEK"
        
        report_lines.append(f"{assessment}")
        report_lines.append(f"")
        
        report_content = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            report_file = self.output_dir / f"weekly_report_{week_start}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Weekly report saved to {report_file}")
        
        return report_content
    
    def generate_summary_report(self, days: int = 30, save_to_file: bool = True) -> str:
        """Generate a summary report for the last N days"""
        recent_perf = performance_analytics.get_recent_performance(days)
        
        report_lines = [
            f"SOLTRADER PERFORMANCE SUMMARY ({days} DAYS)",
            f"=" * 60,
            f"Period: {recent_perf['start_date']} to {recent_perf['end_date']}",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"PERIOD OVERVIEW",
            f"-" * 20,
            f"Active Trading Days: {recent_perf['active_days']}/{days}",
            f"Total Trades: {recent_perf['total_trades']}",
            f"Total Profit/Loss (USD): ${recent_perf['total_profit_usd']:.2f}",
            f"Total Profit/Loss (SOL): {recent_perf['total_profit_sol']:.6f} SOL",
            f"Average Daily Profit: ${recent_perf['average_daily_profit']:.2f}",
            f"",
            f"TOP PERFORMING DAYS",
            f"-" * 20
        ]
        
        # Sort days by profit and show top 5
        daily_summaries = list(recent_perf['daily_summaries'].values())
        if daily_summaries:
            top_days = sorted(daily_summaries, key=lambda x: x.total_profit_loss_usd, reverse=True)[:5]
            
            for i, day in enumerate(top_days, 1):
                report_lines.append(
                    f"{i}. {day.date}: ${day.total_profit_loss_usd:+.2f} "
                    f"({day.total_trades} trades, {day.win_rate:.0f}% win)"
                )
        
        report_lines.extend([
            f"",
            f"WORST PERFORMING DAYS", 
            f"-" * 20
        ])
        
        if daily_summaries:
            worst_days = sorted(daily_summaries, key=lambda x: x.total_profit_loss_usd)[:5]
            
            for i, day in enumerate(worst_days, 1):
                report_lines.append(
                    f"{i}. {day.date}: ${day.total_profit_loss_usd:+.2f} "
                    f"({day.total_trades} trades, {day.win_rate:.0f}% win)"
                )
        
        # Overall statistics
        if daily_summaries:
            profitable_days = len([d for d in daily_summaries if d.total_profit_loss_usd > 0])
            avg_trades_per_day = sum(d.total_trades for d in daily_summaries) / len(daily_summaries)
            avg_win_rate = sum(d.win_rate for d in daily_summaries) / len(daily_summaries)
            
            report_lines.extend([
                f"",
                f"PERIOD STATISTICS",
                f"-" * 20,
                f"Profitable Days: {profitable_days}/{len(daily_summaries)} ({profitable_days/len(daily_summaries)*100:.1f}%)",
                f"Average Trades per Day: {avg_trades_per_day:.1f}",
                f"Average Win Rate: {avg_win_rate:.1f}%",
                f""
            ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            report_file = self.output_dir / f"summary_report_{days}days_{date.today().isoformat()}.txt"
            with open(report_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Summary report saved to {report_file}")
        
        return report_content
    
    def generate_json_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate a JSON report suitable for API consumption"""
        recent_perf = performance_analytics.get_recent_performance(days)
        analytics_stats = performance_analytics.get_analytics_summary()
        
        # Current day and week summaries
        today_summary = performance_analytics.get_daily_summary()
        current_week = performance_analytics.get_weekly_summary()
        
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'period_days': days,
                'report_type': 'json_summary'
            },
            'analytics_stats': analytics_stats,
            'recent_performance': recent_perf,
            'today_summary': today_summary.__dict__ if today_summary else None,
            'current_week_summary': {
                'week_start': current_week.week_start if current_week else None,
                'week_end': current_week.week_end if current_week else None,
                'days_active': current_week.days_active if current_week else 0,
                'total_trades': current_week.total_trades if current_week else 0,
                'total_profit_usd': current_week.total_profit_loss_usd if current_week else 0,
                'win_rate': current_week.win_rate if current_week else 0
            },
            'performance_trends': self._calculate_trends(recent_perf['daily_summaries'])
        }
    
    def _calculate_trends(self, daily_summaries: Dict[str, DailyPerformance]) -> Dict[str, Any]:
        """Calculate performance trends from daily summaries"""
        if len(daily_summaries) < 2:
            return {'trend': 'insufficient_data'}
        
        summaries = list(daily_summaries.values())
        summaries.sort(key=lambda x: x.date)
        
        # Calculate trends
        recent_profits = [s.total_profit_loss_usd for s in summaries[-7:]]  # Last 7 days
        earlier_profits = [s.total_profit_loss_usd for s in summaries[:-7]] if len(summaries) > 7 else []
        
        recent_avg = sum(recent_profits) / len(recent_profits) if recent_profits else 0
        earlier_avg = sum(earlier_profits) / len(earlier_profits) if earlier_profits else recent_avg
        
        if recent_avg > earlier_avg * 1.1:
            trend = 'improving'
        elif recent_avg < earlier_avg * 0.9:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_average_daily': recent_avg,
            'earlier_average_daily': earlier_avg,
            'change_percentage': ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg != 0 else 0
        }

# Global reporter instance
performance_reporter = PerformanceReporter()

def generate_daily_report(date_str: str = None) -> str:
    """Generate daily performance report"""
    return performance_reporter.generate_daily_report(date_str)

def generate_weekly_report(week_start: str = None) -> str:
    """Generate weekly performance report"""
    return performance_reporter.generate_weekly_report(week_start)

def generate_summary_report(days: int = 30) -> str:
    """Generate summary performance report"""
    return performance_reporter.generate_summary_report(days)

def generate_json_report(days: int = 7) -> Dict[str, Any]:
    """Generate JSON performance report"""
    return performance_reporter.generate_json_report(days)

def save_all_reports():
    """Generate and save all types of reports"""
    try:
        daily_report = generate_daily_report()
        weekly_report = generate_weekly_report()
        summary_report = generate_summary_report()
        
        logger.info("All performance reports generated successfully")
        return {
            'daily': len(daily_report.split('\n')),
            'weekly': len(weekly_report.split('\n')),
            'summary': len(summary_report.split('\n'))
        }
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        return None