import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
from collections import defaultdict

from ..analytics.performance_analytics import PerformanceAnalytics
from ..notifications.email_system import EmailNotificationSystem
from ..api.solana_tracker import SolanaTrackerClient
from ..config.settings import Settings
from ..cache import get_token_cache, get_token_display_name, TokenMetadata
from ..risk import get_risk_manager

logger = logging.getLogger(__name__)

class EnhancedDashboard:
    def __init__(self, settings: Settings, analytics: PerformanceAnalytics, 
                 email_system: EmailNotificationSystem, solana_tracker: SolanaTrackerClient):
        self.settings = settings
        self.analytics = analytics
        self.email_system = email_system
        self.solana_tracker = solana_tracker
        
        # Token metadata cache integration
        self.token_cache = get_token_cache()
        
        # Risk management integration
        self.risk_manager = get_risk_manager()
        
        # Dashboard state
        self.is_running = False
        self.last_update = time.time()
        self.update_interval = 5  # 5 seconds
        
        # Dashboard data cache
        self.dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'real_time_metrics': {},
            'daily_breakdown': {},
            'historical_analysis': {},
            'token_discovery': {},
            'enhanced_portfolio': {},
            'token_metadata_stats': {},
            'risk_analysis': {},
            'system_health': {},
            'api_status': {},
            'recent_alerts': []
        }
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.alert_history = []
        
        # Cache refresh tracking
        self.last_cache_refresh = time.time()
        self.cache_refresh_interval = 300  # 5 minutes
        
        # Update task
        self.update_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced Dashboard initialized")

    async def start(self):
        """Start the dashboard update loop"""
        if self.is_running:
            logger.warning("Dashboard already running")
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("Enhanced Dashboard started")

    async def stop(self):
        """Stop the dashboard"""
        if not self.is_running:
            return
        
        logger.info("Stopping Enhanced Dashboard...")
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Enhanced Dashboard stopped")

    async def _update_loop(self):
        """Main dashboard update loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Update all dashboard sections
                await self._update_real_time_metrics()
                await self._update_daily_breakdown()
                await self._update_historical_analysis()
                await self._update_token_discovery()
                await self._update_enhanced_portfolio()
                await self._update_token_metadata_stats()
                await self._update_risk_analysis()
                await self._update_system_health()
                await self._update_api_status()
                
                # Periodic token cache refresh (every 5 minutes)
                if time.time() - self.last_cache_refresh > self.cache_refresh_interval:
                    asyncio.create_task(self.refresh_token_cache_background())
                    self.last_cache_refresh = time.time()
                
                # Update timestamp
                self.dashboard_data['timestamp'] = datetime.now().isoformat()
                self.last_update = time.time()
                
                # Calculate update duration and sleep
                update_duration = time.time() - start_time
                sleep_time = max(0.1, self.update_interval - update_duration)
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(5)

    async def _update_real_time_metrics(self):
        """Update real-time performance metrics"""
        try:
            metrics = self.analytics.get_real_time_metrics()
            
            # Add additional real-time calculations
            metrics.update({
                'performance_trend': self._calculate_performance_trend(),
                'position_health': self._assess_position_health(),
                'trading_velocity': self._calculate_trading_velocity(),
                'risk_level': self._calculate_current_risk_level()
            })
            
            self.dashboard_data['real_time_metrics'] = metrics
            
            # Store historical data
            self.performance_history['portfolio_value'].append({
                'timestamp': time.time(),
                'value': metrics['current_portfolio_value']
            })
            
            # Keep only last 1000 data points
            if len(self.performance_history['portfolio_value']) > 1000:
                self.performance_history['portfolio_value'] = self.performance_history['portfolio_value'][-1000:]
                
        except Exception as e:
            logger.error(f"Error updating real-time metrics: {e}")

    async def _update_daily_breakdown(self):
        """Update daily performance breakdown"""
        try:
            daily_data = self.analytics.get_daily_breakdown()
            
            # Enhanced daily metrics
            enhanced_daily = {
                **daily_data,
                'performance_grade': self._calculate_performance_grade(daily_data),
                'efficiency_score': self._calculate_efficiency_score(daily_data),
                'opportunity_score': self._calculate_opportunity_score(daily_data),
                'risk_adjusted_return': self._calculate_risk_adjusted_return(daily_data)
            }
            
            self.dashboard_data['daily_breakdown'] = enhanced_daily
            
        except Exception as e:
            logger.error(f"Error updating daily breakdown: {e}")

    async def _update_historical_analysis(self):
        """Update historical performance analysis"""
        try:
            # 7-day performance chart data
            seven_day_data = self._get_seven_day_performance()
            
            # Win/loss ratio trends
            win_loss_trends = self._calculate_win_loss_trends()
            
            # Hourly performance heatmap
            hourly_heatmap = self._generate_hourly_heatmap()
            
            # Token category performance
            category_performance = self._analyze_category_performance()
            
            # Risk metrics evolution
            risk_evolution = self._track_risk_evolution()
            
            self.dashboard_data['historical_analysis'] = {
                'seven_day_chart': seven_day_data,
                'win_loss_trends': win_loss_trends,
                'hourly_heatmap': hourly_heatmap,
                'category_performance': category_performance,
                'risk_evolution': risk_evolution,
                'performance_milestones': self._identify_performance_milestones()
            }
            
        except Exception as e:
            logger.error(f"Error updating historical analysis: {e}")

    async def _update_token_discovery(self):
        """Update token discovery intelligence with cached metadata"""
        try:
            discovery_data = self.analytics.get_token_discovery_intelligence()
            
            # Enhance token data with cached metadata
            enhanced_tokens = await self._enhance_tokens_with_metadata(discovery_data)
            
            # Enhanced discovery metrics
            enhanced_discovery = {
                **discovery_data,
                'enhanced_tokens': enhanced_tokens,
                'discovery_efficiency': self._calculate_discovery_efficiency(),
                'source_ranking': self._rank_discovery_sources(),
                'timing_analysis': self._analyze_discovery_timing(),
                'quality_trends': self._analyze_quality_trends(),
                'competitive_analysis': self._perform_competitive_analysis(),
                'cache_stats': self.token_cache.get_cache_stats()
            }
            
            self.dashboard_data['token_discovery'] = enhanced_discovery
            
        except Exception as e:
            logger.error(f"Error updating token discovery: {e}")
    
    async def _update_enhanced_portfolio(self):
        """Update enhanced portfolio view with token metadata"""
        try:
            portfolio_data = await self.get_enhanced_portfolio_view()
            self.dashboard_data['enhanced_portfolio'] = portfolio_data
            
        except Exception as e:
            logger.error(f"Error updating enhanced portfolio: {e}")
    
    async def _update_token_metadata_stats(self):
        """Update token metadata cache statistics"""
        try:
            metadata_stats = self.get_token_metadata_stats()
            self.dashboard_data['token_metadata_stats'] = metadata_stats
            
        except Exception as e:
            logger.error(f"Error updating token metadata stats: {e}")

    async def _update_risk_analysis(self):
        """Update risk analysis and monitoring with real risk manager data"""
        try:
            # Get comprehensive risk summary from risk manager
            risk_summary = self.risk_manager.get_risk_summary()
            
            # Get active risk alerts
            active_alerts = self.risk_manager.get_active_alerts()
            
            # Enhanced risk analysis with token metadata
            enhanced_risk_analysis = await self._enhance_risk_analysis_with_metadata(risk_summary)
            
            self.dashboard_data['risk_analysis'] = {
                'summary': risk_summary,
                'active_alerts': active_alerts,
                'enhanced_analysis': enhanced_risk_analysis,
                'emergency_status': {
                    'emergency_stop_active': risk_summary.get('emergency_stop_active', False),
                    'alert_count': len(active_alerts),
                    'critical_alerts': len([a for a in active_alerts if a.get('level') == 'critical'])
                },
                'position_analysis': {
                    'total_positions': risk_summary.get('active_positions', 0),
                    'position_limits': {
                        'max_positions': risk_summary.get('limits', {}).get('max_positions', 0),
                        'max_position_size': risk_summary.get('limits', {}).get('max_position_size', 0)
                    }
                },
                'performance_metrics': {
                    'daily_pnl': risk_summary.get('daily_pnl', 0),
                    'daily_trade_count': risk_summary.get('daily_trade_count', 0),
                    'current_drawdown': risk_summary.get('portfolio_metrics', {}).get('current_drawdown', 0),
                    'max_drawdown': risk_summary.get('portfolio_metrics', {}).get('max_drawdown', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating risk analysis: {e}")

    async def _update_system_health(self):
        """Update system health monitoring"""
        try:
            # System performance metrics
            uptime = time.time() - self.analytics.real_time_metrics['system_uptime']
            
            # Component health check
            component_health = {
                'analytics': self._check_analytics_health(),
                'email_system': self._check_email_health(),
                'api_client': self._check_api_health(),
                'scanner': self._check_scanner_health()
            }
            
            # Performance metrics
            performance_metrics = {
                'update_frequency': 1 / self.update_interval,
                'last_update_duration': time.time() - self.last_update,
                'memory_usage': self._estimate_memory_usage(),
                'data_freshness': self._assess_data_freshness()
            }
            
            self.dashboard_data['system_health'] = {
                'uptime_hours': uptime / 3600,
                'component_health': component_health,
                'performance_metrics': performance_metrics,
                'system_alerts': self._generate_system_alerts(),
                'maintenance_recommendations': self._generate_maintenance_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Error updating system health: {e}")

    async def _update_api_status(self):
        """Update API usage and status"""
        try:
            # Solana Tracker API status
            api_stats = self.solana_tracker.get_usage_stats()
            
            # API health metrics
            api_health = {
                'solana_tracker': {
                    'status': 'operational',  # Would check with actual ping
                    'usage': api_stats,
                    'performance': self._calculate_api_performance(),
                    'rate_limit_status': self._assess_rate_limit_status(api_stats)
                }
            }
            
            self.dashboard_data['api_status'] = api_health
            
        except Exception as e:
            logger.error(f"Error updating API status: {e}")

    def _calculate_performance_trend(self) -> str:
        """Calculate current performance trend"""
        if len(self.performance_history['portfolio_value']) < 10:
            return "insufficient_data"
        
        recent_values = [point['value'] for point in self.performance_history['portfolio_value'][-10:]]
        
        # Simple trend calculation
        if recent_values[-1] > recent_values[0] * 1.02:
            return "strongly_positive"
        elif recent_values[-1] > recent_values[0] * 1.005:
            return "positive"
        elif recent_values[-1] < recent_values[0] * 0.98:
            return "strongly_negative"
        elif recent_values[-1] < recent_values[0] * 0.995:
            return "negative"
        else:
            return "stable"

    def _assess_position_health(self) -> Dict[str, Any]:
        """Assess health of current positions"""
        open_positions = len(self.analytics.open_positions)
        max_positions = self.settings.MAX_SIMULTANEOUS_POSITIONS
        
        utilization = (open_positions / max_positions) * 100 if max_positions > 0 else 0
        
        health_score = 100 - max(0, utilization - 80) * 2  # Penalty for high utilization
        
        return {
            'utilization_percentage': utilization,
            'health_score': health_score,
            'status': 'healthy' if health_score > 80 else 'warning' if health_score > 60 else 'critical'
        }

    def _calculate_trading_velocity(self) -> Dict[str, Any]:
        """Calculate current trading velocity"""
        # Count trades in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_trades = [
            trade for trade in self.analytics.recent_trades
            if trade.timestamp >= one_hour_ago
        ]
        
        trades_per_hour = len(recent_trades)
        max_hourly = self.settings.MAX_DAILY_TRADES / 24
        
        velocity_percentage = (trades_per_hour / max_hourly) * 100 if max_hourly > 0 else 0
        
        return {
            'trades_per_hour': trades_per_hour,
            'velocity_percentage': velocity_percentage,
            'status': 'high' if velocity_percentage > 80 else 'normal' if velocity_percentage > 30 else 'low'
        }

    def _calculate_current_risk_level(self) -> str:
        """Calculate current overall risk level"""
        risk_score = self.analytics.risk_metrics.get('risk_score', 0)
        
        if risk_score > 80:
            return "very_high"
        elif risk_score > 60:
            return "high"
        elif risk_score > 40:
            return "moderate"
        elif risk_score > 20:
            return "low"
        else:
            return "very_low"

    def _calculate_performance_grade(self, daily_data: Dict) -> str:
        """Calculate performance grade for the day"""
        score = 0
        
        # Win rate scoring (0-30 points)
        win_rate = daily_data.get('win_rate', 0)
        score += min(30, win_rate * 0.5)
        
        # Approval rate scoring (0-25 points)
        approval_rate = daily_data.get('approval_rate', 0)
        score += min(25, approval_rate * 1.25)
        
        # PnL scoring (0-25 points)
        pnl = daily_data.get('total_pnl', 0)
        if pnl > 0:
            score += min(25, pnl * 5)
        
        # Efficiency scoring (0-20 points)
        if daily_data.get('trades_executed', 0) > 0:
            efficiency = daily_data.get('total_pnl', 0) / daily_data.get('trades_executed', 1)
            score += min(20, max(0, efficiency * 10 + 10))
        
        # Convert to letter grade
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 45:
            return "D"
        else:
            return "F"

    def _calculate_efficiency_score(self, daily_data: Dict) -> float:
        """Calculate trading efficiency score"""
        trades = daily_data.get('trades_executed', 0)
        if trades == 0:
            return 0.0
        
        pnl_per_trade = daily_data.get('total_pnl', 0) / trades
        gas_per_trade = daily_data.get('gas_fees', 0) / trades
        
        # Efficiency = (PnL - Gas) / Trade
        net_per_trade = pnl_per_trade - gas_per_trade
        
        # Normalize to 0-100 scale
        return max(0, min(100, (net_per_trade + 1) * 50))

    def _calculate_opportunity_score(self, daily_data: Dict) -> float:
        """Calculate how well we're capturing opportunities"""
        approved = daily_data.get('tokens_approved', 0)
        traded = daily_data.get('trades_executed', 0)
        
        if approved == 0:
            return 0.0
        
        capture_rate = (traded / approved) * 100
        return min(100, capture_rate)

    def _calculate_risk_adjusted_return(self, daily_data: Dict) -> float:
        """Calculate risk-adjusted return"""
        pnl = daily_data.get('total_pnl', 0)
        trades = daily_data.get('trades_executed', 0)
        
        if trades == 0:
            return 0.0
        
        # Simple risk adjustment based on number of trades
        risk_factor = min(1.0, trades / 10)  # More trades = higher risk
        
        return pnl * (1 - risk_factor * 0.2)  # Reduce return by up to 20% for risk

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get condensed dashboard summary"""
        return {
            'status': 'operational' if self.is_running else 'stopped',
            'last_update': self.dashboard_data['timestamp'],
            'portfolio_value': self.dashboard_data['real_time_metrics'].get('current_portfolio_value', 0),
            'todays_pnl': self.dashboard_data['real_time_metrics'].get('todays_pnl', 0),
            'active_positions': self.dashboard_data['real_time_metrics'].get('active_positions', 0),
            'approval_rate': self.dashboard_data['daily_breakdown'].get('approval_rate', 0),
            'win_rate': self.dashboard_data['daily_breakdown'].get('win_rate', 0),
            'api_requests_remaining': self.dashboard_data['real_time_metrics'].get('api_requests_remaining', 0),
            'system_health': 'healthy',  # Simplified
            'performance_grade': self.dashboard_data['daily_breakdown'].get('performance_grade', 'N/A')
        }

    # Placeholder methods for complex analytics (would be fully implemented)
    def _get_seven_day_performance(self) -> List[Dict]:
        return []
    
    def _calculate_win_loss_trends(self) -> Dict:
        return {}
    
    def _generate_hourly_heatmap(self) -> Dict:
        return {}
    
    def _analyze_category_performance(self) -> Dict:
        return {}
    
    def _track_risk_evolution(self) -> List[Dict]:
        return []
    
    def _identify_performance_milestones(self) -> List[Dict]:
        return []
    
    def _calculate_discovery_efficiency(self) -> float:
        return 0.0
    
    def _rank_discovery_sources(self) -> List[Dict]:
        return []
    
    def _analyze_discovery_timing(self) -> Dict:
        return {}
    
    def _analyze_quality_trends(self) -> Dict:
        return {}
    
    def _perform_competitive_analysis(self) -> Dict:
        return {}
    
    def _analyze_risk_trends(self) -> Dict:
        return {}
    
    def _analyze_position_concentration(self) -> Dict:
        return {}
    
    def _generate_risk_alerts(self) -> List[Dict]:
        return []
    
    def _breakdown_risk_score(self) -> Dict:
        return {}
    
    def _generate_risk_recommendations(self) -> List[str]:
        return []
    
    def _check_analytics_health(self) -> str:
        return 'healthy'
    
    def _check_email_health(self) -> str:
        return 'healthy' if self.email_system.enabled else 'disabled'
    
    def _check_api_health(self) -> str:
        return 'healthy'
    
    def _check_scanner_health(self) -> str:
        return 'healthy'
    
    def _estimate_memory_usage(self) -> str:
        return 'normal'
    
    def _assess_data_freshness(self) -> str:
        return 'fresh'
    
    def _generate_system_alerts(self) -> List[Dict]:
        return []
    
    def _generate_maintenance_recommendations(self) -> List[str]:
        return []
    
    def _calculate_api_performance(self) -> Dict:
        return {'latency': 'low', 'success_rate': 99.5}
    
    def _assess_rate_limit_status(self, api_stats: Dict) -> str:
        usage_pct = api_stats.get('usage_percentage', 0)
        if usage_pct > 90:
            return 'critical'
        elif usage_pct > 75:
            return 'warning'
        else:
            return 'normal'
    
    async def _enhance_tokens_with_metadata(self, discovery_data: Dict) -> Dict[str, Any]:
        """Enhance token discovery data with cached metadata"""
        try:
            enhanced_tokens = {}
            token_addresses = []
            
            # Extract token addresses from discovery data
            if 'recent_discoveries' in discovery_data:
                for token_info in discovery_data['recent_discoveries']:
                    if isinstance(token_info, dict) and 'address' in token_info:
                        token_addresses.append(token_info['address'])
                    elif isinstance(token_info, str):
                        token_addresses.append(token_info)
            
            # Get batch metadata for all tokens
            if token_addresses:
                logger.info(f"[DASHBOARD] Enhancing {len(token_addresses)} tokens with cached metadata")
                metadata_batch = await self.token_cache.get_batch_metadata(token_addresses)
                
                # Process each token with its metadata
                for address, metadata in metadata_batch.items():
                    if metadata:
                        enhanced_tokens[address] = {
                            'address': address,
                            'display_name': metadata.display_name,
                            'full_display_name': metadata.full_display_name,
                            'symbol': metadata.symbol,
                            'name': metadata.name,
                            'verified': metadata.verified,
                            'price_usd': metadata.price_usd,
                            'market_cap_usd': metadata.market_cap_usd,
                            'warnings': metadata.warnings,
                            'logo_uri': metadata.logo_uri,
                            'source': metadata.source,
                            'last_updated': metadata.cached_at.isoformat() if metadata.cached_at else None
                        }
                    else:
                        # Fallback for tokens without metadata
                        enhanced_tokens[address] = {
                            'address': address,
                            'display_name': f"{address[:8]}...",
                            'full_display_name': f"Unknown ({address[:8]}...)",
                            'symbol': 'UNKNOWN',
                            'name': 'Unknown Token',
                            'verified': False,
                            'warnings': ['metadata_unavailable'],
                            'source': 'fallback'
                        }
            
            return {
                'tokens': enhanced_tokens,
                'total_enhanced': len(enhanced_tokens),
                'metadata_coverage': len([t for t in enhanced_tokens.values() if t['source'] != 'fallback']) / len(enhanced_tokens) * 100 if enhanced_tokens else 0
            }
            
        except Exception as e:
            logger.error(f"[DASHBOARD] Error enhancing tokens with metadata: {e}")
            return {'tokens': {}, 'total_enhanced': 0, 'metadata_coverage': 0}
    
    async def get_enhanced_portfolio_view(self) -> Dict[str, Any]:
        """Get portfolio view with enhanced token information"""
        try:
            # Get current positions from analytics
            positions = getattr(self.analytics, 'current_positions', {})
            enhanced_positions = {}
            
            if positions:
                # Get metadata for all position tokens
                position_addresses = list(positions.keys())
                metadata_batch = await self.token_cache.get_batch_metadata(position_addresses)
                
                for address, position_data in positions.items():
                    metadata = metadata_batch.get(address)
                    
                    enhanced_positions[address] = {
                        **position_data,
                        'token_info': {
                            'display_name': metadata.display_name if metadata else f"{address[:8]}...",
                            'full_display_name': metadata.full_display_name if metadata else f"Unknown ({address[:8]}...)",
                            'symbol': metadata.symbol if metadata else 'UNKNOWN',
                            'name': metadata.name if metadata else 'Unknown Token',
                            'verified': metadata.verified if metadata else False,
                            'price_usd': metadata.price_usd if metadata else None,
                            'logo_uri': metadata.logo_uri if metadata else None,
                            'warnings': metadata.warnings if metadata else [],
                        }
                    }
            
            return {
                'enhanced_positions': enhanced_positions,
                'total_positions': len(enhanced_positions),
                'verified_positions': len([p for p in enhanced_positions.values() if p['token_info']['verified']]),
                'total_value_usd': sum(
                    p.get('value_usd', 0) for p in enhanced_positions.values() 
                    if p.get('value_usd') is not None
                )
            }
            
        except Exception as e:
            logger.error(f"[DASHBOARD] Error creating enhanced portfolio view: {e}")
            return {'enhanced_positions': {}, 'total_positions': 0, 'verified_positions': 0, 'total_value_usd': 0}
    
    async def refresh_token_cache_background(self):
        """Background task to refresh popular token metadata"""
        try:
            await self.token_cache.refresh_popular_tokens()
            await self.token_cache.clear_expired_cache()
            logger.info("[DASHBOARD] Background token cache refresh completed")
        except Exception as e:
            logger.error(f"[DASHBOARD] Background token cache refresh failed: {e}")
    
    def get_token_metadata_stats(self) -> Dict[str, Any]:
        """Get token metadata cache statistics for dashboard"""
        try:
            cache_stats = self.token_cache.get_cache_stats()
            return {
                'cache_performance': {
                    'memory_entries': cache_stats.get('memory_cache_size', 0),
                    'disk_entries': cache_stats.get('disk_cache_files', 0),
                    'last_refresh': cache_stats.get('last_refresh'),
                    'cache_directory': cache_stats.get('cache_directory')
                },
                'metadata_sources': {
                    'jupiter_primary': True,
                    'birdeye_enhancement': bool(self.token_cache.birdeye_api_key),
                    'solana_rpc_fallback': True
                }
            }
        except Exception as e:
            logger.error(f"[DASHBOARD] Error getting token metadata stats: {e}")
            return {'cache_performance': {}, 'metadata_sources': {}}
    
    async def _enhance_risk_analysis_with_metadata(self, risk_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance risk analysis with token metadata"""
        try:
            enhanced_analysis = {
                'position_details': {},
                'token_risk_assessment': {},
                'portfolio_composition': {}
            }
            
            # Get current positions from risk manager
            positions = getattr(self.risk_manager, 'current_positions', {})
            
            if positions:
                # Get metadata for all position tokens
                position_addresses = list(positions.keys())
                metadata_batch = await self.token_cache.get_batch_metadata(position_addresses)
                
                for address, position_risk in positions.items():
                    metadata = metadata_batch.get(address)
                    
                    # Enhanced position details
                    enhanced_analysis['position_details'][address] = {
                        'position_size': float(position_risk.position_size),
                        'token_info': {
                            'display_name': metadata.display_name if metadata else f"{address[:8]}...",
                            'symbol': metadata.symbol if metadata else 'UNKNOWN',
                            'verified': metadata.verified if metadata else False,
                            'warnings': metadata.warnings if metadata else [],
                            'price_usd': metadata.price_usd if metadata else None
                        },
                        'risk_assessment': {
                            'concentration_risk': position_risk.concentration_risk.value if hasattr(position_risk, 'concentration_risk') else 'unknown',
                            'volatility_score': float(position_risk.volatility_score) if hasattr(position_risk, 'volatility_score') and position_risk.volatility_score else None
                        }
                    }
                
                # Token risk assessment summary
                verified_count = sum(1 for pos in enhanced_analysis['position_details'].values() if pos['token_info']['verified'])
                warning_count = sum(1 for pos in enhanced_analysis['position_details'].values() if pos['token_info']['warnings'])
                
                enhanced_analysis['token_risk_assessment'] = {
                    'total_positions': len(positions),
                    'verified_positions': verified_count,
                    'unverified_positions': len(positions) - verified_count,
                    'positions_with_warnings': warning_count,
                    'verification_rate': verified_count / len(positions) * 100 if positions else 0
                }
                
                # Portfolio composition analysis
                total_value = sum(
                    pos.get('position_size', 0) for pos in enhanced_analysis['position_details'].values()
                )
                
                enhanced_analysis['portfolio_composition'] = {
                    'total_value_estimate': total_value,
                    'largest_position': max(
                        (pos.get('position_size', 0) for pos in enhanced_analysis['position_details'].values()),
                        default=0
                    ),
                    'concentration_score': (
                        max((pos.get('position_size', 0) for pos in enhanced_analysis['position_details'].values()), default=0)
                        / total_value * 100 if total_value > 0 else 0
                    )
                }
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"[DASHBOARD] Error enhancing risk analysis with metadata: {e}")
            return {'position_details': {}, 'token_risk_assessment': {}, 'portfolio_composition': {}}