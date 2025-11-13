#!/usr/bin/env python3
"""
Monitoring Dashboard Integration
===============================

Real-time monitoring dashboard for SolTrader production deployment:
- Grafana dashboard configurations
- Custom web dashboard for trading metrics
- Alert visualization and management
- Performance analytics and reporting

Features:
- Real-time trading performance metrics
- System health monitoring
- Alert management interface
- Historical data visualization
- Export capabilities for reports
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)

@dataclass
class DashboardPanel:
    """Dashboard panel configuration"""
    id: str
    title: str
    type: str  # graph, singlestat, table, etc.
    targets: List[Dict[str, Any]]
    grid_pos: Dict[str, int]  # x, y, w, h
    options: Dict[str, Any] = None

@dataclass
class GrafanaDashboard:
    """Grafana dashboard configuration"""
    uid: str
    title: str
    description: str
    tags: List[str]
    panels: List[DashboardPanel]
    time_range: Dict[str, str] = None
    refresh_interval: str = "30s"

class DashboardGenerator:
    """Generate monitoring dashboards for different purposes"""
    
    def __init__(self):
        self.dashboards = {}
        logger.info("DashboardGenerator initialized")
    
    def create_trading_dashboard(self) -> GrafanaDashboard:
        """Create comprehensive trading dashboard"""
        
        panels = [
            # Portfolio Overview
            DashboardPanel(
                id="portfolio_value",
                title="Portfolio Value (SOL)",
                type="singlestat",
                targets=[{
                    "expr": "soltrader_portfolio_value_sol",
                    "legendFormat": "Portfolio Value"
                }],
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                options={"unit": "SOL", "decimals": 2}
            ),
            
            # Total PnL
            DashboardPanel(
                id="total_pnl",
                title="Total PnL (SOL)",
                type="singlestat",
                targets=[{
                    "expr": "sum(soltrader_pnl_total)",
                    "legendFormat": "Total PnL"
                }],
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
                options={"unit": "SOL", "decimals": 2, "colorMode": "value"}
            ),
            
            # Win Rate
            DashboardPanel(
                id="win_rate",
                title="Win Rate %",
                type="singlestat",
                targets=[{
                    "expr": "avg(soltrader_win_rate) * 100",
                    "legendFormat": "Win Rate"
                }],
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
                options={"unit": "percent", "decimals": 1, "thresholds": [40, 60]}
            ),
            
            # Daily Trades
            DashboardPanel(
                id="daily_trades",
                title="Daily Trades",
                type="singlestat",
                targets=[{
                    "expr": "increase(soltrader_trades_total[24h])",
                    "legendFormat": "Trades"
                }],
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
                options={"unit": "short", "decimals": 0}
            ),
            
            # Trade Execution Times
            DashboardPanel(
                id="execution_times",
                title="Trade Execution Times",
                type="graph",
                targets=[{
                    "expr": "histogram_quantile(0.50, soltrader_trade_execution_duration_seconds_bucket)",
                    "legendFormat": "50th percentile"
                }, {
                    "expr": "histogram_quantile(0.95, soltrader_trade_execution_duration_seconds_bucket)",
                    "legendFormat": "95th percentile"
                }],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 6},
                options={"unit": "seconds", "yAxes": [{"logBase": 1}]}
            ),
            
            # Strategy Performance
            DashboardPanel(
                id="strategy_pnl",
                title="Strategy PnL by Type",
                type="graph",
                targets=[{
                    "expr": "soltrader_pnl_total",
                    "legendFormat": "{{strategy}}"
                }],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 6},
                options={"unit": "SOL", "legend": {"show": True, "values": True}}
            ),
            
            # API Performance
            DashboardPanel(
                id="api_performance",
                title="API Request Duration",
                type="graph",
                targets=[{
                    "expr": "histogram_quantile(0.95, soltrader_api_request_duration_seconds_bucket)",
                    "legendFormat": "{{api}} - 95th percentile"
                }],
                grid_pos={"x": 0, "y": 10, "w": 12, "h": 6},
                options={"unit": "seconds", "yAxes": [{"logBase": 1}]}
            ),
            
            # System Resources
            DashboardPanel(
                id="memory_usage",
                title="Memory Usage",
                type="graph",
                targets=[{
                    "expr": "soltrader_memory_usage_bytes / 1024 / 1024",
                    "legendFormat": "{{component}}"
                }],
                grid_pos={"x": 12, "y": 10, "w": 12, "h": 6},
                options={"unit": "MB", "yAxes": [{"min": 0}]}
            )
        ]
        
        dashboard = GrafanaDashboard(
            uid="soltrader-trading",
            title="SolTrader - Trading Performance",
            description="Comprehensive trading performance monitoring",
            tags=["soltrader", "trading", "performance"],
            panels=panels,
            time_range={"from": "now-6h", "to": "now"},
            refresh_interval="30s"
        )
        
        self.dashboards["trading"] = dashboard
        return dashboard
    
    def create_system_dashboard(self) -> GrafanaDashboard:
        """Create system health dashboard"""
        
        panels = [
            # System Status
            DashboardPanel(
                id="system_status",
                title="System Status",
                type="singlestat",
                targets=[{
                    "expr": "up{job=\"soltrader\"}",
                    "legendFormat": "Status"
                }],
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                options={"valueMaps": [{"value": 1, "text": "UP"}, {"value": 0, "text": "DOWN"}]}
            ),
            
            # Memory Usage
            DashboardPanel(
                id="memory_percent",
                title="Memory Usage %",
                type="singlestat",
                targets=[{
                    "expr": "soltrader_memory_usage_bytes / (2 * 1024^3) * 100",
                    "legendFormat": "Memory %"
                }],
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
                options={"unit": "percent", "thresholds": [70, 85], "colorMode": "background"}
            ),
            
            # CPU Usage
            DashboardPanel(
                id="cpu_usage",
                title="CPU Usage %",
                type="singlestat",
                targets=[{
                    "expr": "soltrader_cpu_usage_percent",
                    "legendFormat": "CPU %"
                }],
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
                options={"unit": "percent", "thresholds": [70, 85], "colorMode": "background"}
            ),
            
            # Active Alerts
            DashboardPanel(
                id="active_alerts",
                title="Active Alerts",
                type="singlestat",
                targets=[{
                    "expr": "count(ALERTS{alertstate=\"firing\"})",
                    "legendFormat": "Alerts"
                }],
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
                options={"unit": "short", "colorMode": "background", "thresholds": [0, 1]}
            ),
            
            # Resource Trends
            DashboardPanel(
                id="resource_trends",
                title="Resource Usage Over Time",
                type="graph",
                targets=[{
                    "expr": "soltrader_memory_usage_bytes / 1024^3",
                    "legendFormat": "Memory (GB)"
                }, {
                    "expr": "soltrader_cpu_usage_percent / 100",
                    "legendFormat": "CPU (fraction)"
                }],
                grid_pos={"x": 0, "y": 4, "w": 24, "h": 8}
            )
        ]
        
        dashboard = GrafanaDashboard(
            uid="soltrader-system",
            title="SolTrader - System Health",
            description="System health and resource monitoring",
            tags=["soltrader", "system", "health"],
            panels=panels,
            time_range={"from": "now-1h", "to": "now"},
            refresh_interval="15s"
        )
        
        self.dashboards["system"] = dashboard
        return dashboard
    
    def create_alerts_dashboard(self) -> GrafanaDashboard:
        """Create alerts management dashboard"""
        
        panels = [
            # Alert Summary
            DashboardPanel(
                id="alert_summary",
                title="Alert Summary",
                type="table",
                targets=[{
                    "expr": "count by (alertname, severity) (ALERTS)",
                    "format": "table"
                }],
                grid_pos={"x": 0, "y": 0, "w": 24, "h": 8}
            ),
            
            # Alert History
            DashboardPanel(
                id="alert_history",
                title="Alert History (24h)",
                type="graph",
                targets=[{
                    "expr": "increase(soltrader_alerts_total[1h])",
                    "legendFormat": "{{rule}} - {{severity}}"
                }],
                grid_pos={"x": 0, "y": 8, "w": 24, "h": 8}
            )
        ]
        
        dashboard = GrafanaDashboard(
            uid="soltrader-alerts",
            title="SolTrader - Alert Management",
            description="Alert monitoring and management",
            tags=["soltrader", "alerts", "monitoring"],
            panels=panels,
            time_range={"from": "now-24h", "to": "now"},
            refresh_interval="1m"
        )
        
        self.dashboards["alerts"] = dashboard
        return dashboard
    
    def export_dashboard_json(self, dashboard_name: str) -> Dict[str, Any]:
        """Export dashboard as Grafana JSON"""
        
        if dashboard_name not in self.dashboards:
            raise ValueError(f"Dashboard '{dashboard_name}' not found")
        
        dashboard = self.dashboards[dashboard_name]
        
        # Convert to Grafana JSON format
        grafana_json = {
            "dashboard": {
                "uid": dashboard.uid,
                "title": dashboard.title,
                "description": dashboard.description,
                "tags": dashboard.tags,
                "time": dashboard.time_range or {"from": "now-1h", "to": "now"},
                "refresh": dashboard.refresh_interval,
                "panels": []
            },
            "overwrite": True
        }
        
        # Convert panels
        for i, panel in enumerate(dashboard.panels):
            grafana_panel = {
                "id": i + 1,
                "title": panel.title,
                "type": panel.type,
                "gridPos": panel.grid_pos,
                "targets": panel.targets,
                "options": panel.options or {}
            }
            grafana_json["dashboard"]["panels"].append(grafana_panel)
        
        return grafana_json
    
    def generate_all_dashboards(self) -> Dict[str, Dict[str, Any]]:
        """Generate all dashboards and return as JSON"""
        
        self.create_trading_dashboard()
        self.create_system_dashboard()
        self.create_alerts_dashboard()
        
        return {
            name: self.export_dashboard_json(name)
            for name in self.dashboards.keys()
        }

class MonitoringReports:
    """Generate monitoring reports and analytics"""
    
    def __init__(self):
        self.report_history = []
        logger.info("MonitoringReports initialized")
    
    async def generate_daily_report(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate daily monitoring report"""
        
        report = {
            "report_type": "daily",
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": (datetime.now() - timedelta(days=1)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "summary": {
                "system_health": "healthy",  # Mock
                "total_trades": 45,  # Mock
                "total_pnl_sol": 12.5,  # Mock
                "win_rate": 0.67,  # Mock
                "uptime_percent": 99.8,  # Mock
                "alerts_triggered": 2  # Mock
            },
            "performance_metrics": {
                "avg_execution_time_ms": 1800,
                "api_success_rate": 0.995,
                "memory_usage_avg_mb": 1200,
                "cpu_usage_avg_percent": 45
            },
            "alerts": {
                "critical": 0,
                "warnings": 2,
                "resolved": 2,
                "avg_resolution_time_minutes": 15
            },
            "recommendations": [
                "System performance is within normal ranges",
                "Consider monitoring API response times during peak hours",
                "Memory usage trending upward - monitor closely"
            ]
        }
        
        self.report_history.append(report)
        return report
    
    async def generate_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Generate performance summary report"""
        
        return {
            "report_type": "performance_summary",
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "trading_performance": {
                "total_trades": days * 35,  # Mock
                "total_pnl_sol": days * 8.5,  # Mock
                "avg_win_rate": 0.65,
                "best_strategy": "momentum",
                "worst_strategy": "mean_reversion",
                "avg_execution_time_ms": 1950
            },
            "system_performance": {
                "avg_uptime_percent": 99.7,
                "peak_memory_mb": 1800,
                "avg_memory_mb": 1250,
                "peak_cpu_percent": 78,
                "avg_cpu_percent": 42
            },
            "api_performance": {
                "jupiter_success_rate": 0.998,
                "birdeye_success_rate": 0.995,
                "avg_response_time_ms": 450,
                "total_api_calls": days * 2500
            },
            "trends": {
                "memory_trend": "stable",
                "performance_trend": "improving",
                "alert_trend": "decreasing"
            }
        }
    
    def export_report_json(self, report: Dict[str, Any], filename: str = None) -> str:
        """Export report as JSON file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"monitoring_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Report exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            raise

class IntegratedMonitoringDashboard:
    """Integrated monitoring dashboard combining all components"""
    
    def __init__(self):
        self.dashboard_generator = DashboardGenerator()
        self.reports = MonitoringReports()
        self.is_running = False
        
        logger.info("IntegratedMonitoringDashboard initialized")
    
    async def initialize(self):
        """Initialize the integrated dashboard"""
        
        try:
            # Generate all dashboards
            dashboards = self.dashboard_generator.generate_all_dashboards()
            
            # Save dashboard configurations
            for name, config in dashboards.items():
                filename = f"grafana_dashboard_{name}.json"
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                logger.info(f"Saved dashboard config: {filename}")
            
            self.is_running = True
            logger.info("Integrated monitoring dashboard initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {e}")
            raise
    
    async def generate_comprehensive_status(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring status"""
        
        # Generate daily report
        daily_report = await self.reports.generate_daily_report({})
        
        # Generate performance summary
        performance_summary = await self.reports.generate_performance_summary()
        
        return {
            "dashboard_status": {
                "is_running": self.is_running,
                "dashboards_available": list(self.dashboard_generator.dashboards.keys()),
                "total_panels": sum(len(d.panels) for d in self.dashboard_generator.dashboards.values())
            },
            "current_status": {
                "timestamp": datetime.now().isoformat(),
                "system_health": "healthy",
                "monitoring_active": True,
                "alerts_active": 0
            },
            "daily_report": daily_report,
            "performance_summary": performance_summary
        }
    
    def get_dashboard_urls(self, base_url: str = "http://localhost:3000") -> Dict[str, str]:
        """Get dashboard URLs for Grafana"""
        
        return {
            name: f"{base_url}/d/{dashboard.uid}/{dashboard.title.lower().replace(' ', '-')}"
            for name, dashboard in self.dashboard_generator.dashboards.items()
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_monitoring_dashboard():
        """Test the integrated monitoring dashboard"""
        
        print("Testing Integrated Monitoring Dashboard")
        print("=" * 50)
        
        dashboard = IntegratedMonitoringDashboard()
        
        try:
            # Initialize dashboard
            await dashboard.initialize()
            print("Dashboard initialized successfully")
            
            # Generate comprehensive status
            print("\nGenerating comprehensive status...")
            status = await dashboard.generate_comprehensive_status()
            
            print(f"Dashboard Status:")
            print(f"  Running: {status['dashboard_status']['is_running']}")
            print(f"  Available Dashboards: {status['dashboard_status']['dashboards_available']}")
            print(f"  Total Panels: {status['dashboard_status']['total_panels']}")
            
            print(f"\nDaily Report Summary:")
            daily = status['daily_report']['summary']
            print(f"  Total Trades: {daily['total_trades']}")
            print(f"  Total PnL: {daily['total_pnl_sol']} SOL")
            print(f"  Win Rate: {daily['win_rate']*100:.1f}%")
            print(f"  Uptime: {daily['uptime_percent']}%")
            
            print(f"\nPerformance Summary (7 days):")
            perf = status['performance_summary']
            print(f"  Trading Performance: {perf['trading_performance']['total_pnl_sol']} SOL PnL")
            print(f"  System Uptime: {perf['system_performance']['avg_uptime_percent']}%")
            print(f"  API Success Rate: {perf['api_performance']['jupiter_success_rate']*100:.1f}%")
            
            # Get dashboard URLs
            urls = dashboard.get_dashboard_urls()
            print(f"\nDashboard URLs:")
            for name, url in urls.items():
                print(f"  {name.title()}: {url}")
            
            # Export daily report
            report_file = dashboard.reports.export_report_json(
                status['daily_report'], 
                "test_daily_report.json"
            )
            print(f"\nDaily report exported to: {report_file}")
            
            print("\nIntegrated monitoring dashboard test completed successfully!")
            
        except Exception as e:
            print(f"Dashboard test failed: {e}")
            raise
    
    asyncio.run(test_monitoring_dashboard())