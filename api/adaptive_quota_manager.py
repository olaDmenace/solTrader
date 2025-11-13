import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class QuotaStrategy(Enum):
    CONSERVATIVE = "conservative"    # 70% of quota max
    BALANCED = "balanced"           # 85% of quota max  
    AGGRESSIVE = "aggressive"       # 95% of quota max
    ADAPTIVE = "adaptive"           # Dynamic based on performance

@dataclass
class QuotaAllocation:
    provider: str
    total_quota: int
    allocated_quota: int
    used_quota: int
    reserved_quota: int
    time_window: str
    allocation_strategy: QuotaStrategy
    
    @property
    def remaining_quota(self) -> int:
        return self.allocated_quota - self.used_quota
    
    @property
    def utilization_rate(self) -> float:
        return self.used_quota / self.allocated_quota if self.allocated_quota > 0 else 0.0
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on tokens discovered per quota used"""
        # This will be updated by the quota manager
        return getattr(self, '_efficiency_score', 0.0)

class AdaptiveQuotaManager:
    """
    Intelligent quota management system that optimizes API usage for maximum token discovery.
    
    Features:
    - Dynamic quota allocation based on API performance
    - Time-based quota distribution (peak hours vs off-peak)
    - Emergency quota conservation
    - Performance-based reallocation
    - Predictive quota management
    """
    
    def __init__(self):
        # Quota allocations for different providers
        self.allocations = {
            'solana_tracker': QuotaAllocation(
                provider='solana_tracker',
                total_quota=333,  # Conservative daily limit
                allocated_quota=283,  # 85% allocation (15% reserved)
                used_quota=0,
                reserved_quota=50,  # Emergency reserve
                time_window='daily',
                allocation_strategy=QuotaStrategy.BALANCED
            ),
            'geckoterminal': QuotaAllocation(
                provider='geckoterminal', 
                total_quota=36000,  # Conservative limit for "unlimited" API
                allocated_quota=30600,  # 85% allocation
                used_quota=0,
                reserved_quota=5400,  # Large reserve for fallback
                time_window='daily',
                allocation_strategy=QuotaStrategy.BALANCED
            )
        }
        
        # Performance tracking for quota optimization
        self.performance_history = {
            'solana_tracker': [],
            'geckoterminal': []
        }
        
        # Time-based distribution (24-hour cycle)
        self.hourly_distribution = self._calculate_optimal_hourly_distribution()
        
        # Adaptive parameters
        self.learning_enabled = True
        self.emergency_mode = False
        self.last_reallocation = datetime.now()
        self.reallocation_interval = timedelta(hours=2)  # Reallocate every 2 hours
        
        # Performance thresholds
        self.min_efficiency_threshold = 5.0  # Tokens per quota
        self.excellent_efficiency_threshold = 25.0
        self.emergency_quota_threshold = 0.1  # 10% remaining triggers emergency mode
        
        logger.info("Adaptive Quota Manager initialized")

    def _calculate_optimal_hourly_distribution(self) -> Dict[int, float]:
        """Calculate optimal quota distribution across 24 hours based on market activity"""
        # Peak market hours: 6 AM - 10 PM UTC (crypto most active)
        # Off-peak: 10 PM - 6 AM UTC
        
        distribution = {}
        
        for hour in range(24):
            if 6 <= hour <= 22:  # Peak hours
                # Higher allocation during peak trading hours
                if 9 <= hour <= 16:  # US market hours
                    distribution[hour] = 1.3  # 30% more quota
                else:
                    distribution[hour] = 1.1  # 10% more quota
            else:  # Off-peak hours
                distribution[hour] = 0.7  # 30% less quota
        
        # Normalize to ensure total = 24 (average of 1.0)
        total = sum(distribution.values())
        for hour in distribution:
            distribution[hour] = distribution[hour] / total * 24
            
        return distribution

    def get_current_hour_multiplier(self) -> float:
        """Get the quota multiplier for the current hour"""
        current_hour = datetime.now().hour
        return self.hourly_distribution.get(current_hour, 1.0)

    def calculate_optimal_quota_for_timeframe(self, provider: str, hours_remaining: float) -> int:
        """Calculate optimal quota allocation for remaining time in day"""
        allocation = self.allocations.get(provider)
        if not allocation:
            return 0
        
        # Calculate time-based distribution
        current_hour = datetime.now().hour
        hours_left_today = 24 - current_hour + (datetime.now().minute / 60.0)
        
        if hours_left_today <= 0:
            return allocation.reserved_quota  # Use only reserve quota
        
        # Calculate remaining hourly multipliers for the day
        remaining_multipliers = []
        for i in range(int(hours_left_today) + 1):
            hour = (current_hour + i) % 24
            remaining_multipliers.append(self.hourly_distribution.get(hour, 1.0))
        
        # Calculate weighted distribution
        total_remaining_weight = sum(remaining_multipliers)
        current_weight = remaining_multipliers[0] if remaining_multipliers else 1.0
        
        # Optimal quota for current timeframe
        base_quota_per_hour = allocation.remaining_quota / hours_left_today
        optimal_quota = int(base_quota_per_hour * current_weight)
        
        return min(optimal_quota, allocation.remaining_quota)

    def request_quota(self, provider: str, requested_amount: int) -> Tuple[bool, int, str]:
        """
        Request quota allocation with intelligent approval/denial
        
        Returns: (approved, allocated_amount, reason)
        """
        allocation = self.allocations.get(provider)
        if not allocation:
            return False, 0, f"Provider {provider} not found"
        
        # Check if in emergency mode
        if self.emergency_mode and provider == 'solana_tracker':
            if allocation.remaining_quota < allocation.reserved_quota * 0.5:
                return False, 0, "Emergency mode: conserving critical quota"
        
        # Calculate time-based quota availability
        hours_remaining = 24 - datetime.now().hour
        optimal_quota = self.calculate_optimal_quota_for_timeframe(provider, hours_remaining)
        
        # Check current efficiency
        recent_efficiency = self._get_recent_efficiency(provider)
        
        # Intelligent quota allocation logic
        if requested_amount <= optimal_quota:
            # Standard approval
            allocation.used_quota += requested_amount
            return True, requested_amount, "Standard allocation approved"
        
        elif requested_amount <= allocation.remaining_quota:
            # Check if efficiency justifies higher allocation
            if recent_efficiency >= self.excellent_efficiency_threshold:
                allocation.used_quota += requested_amount
                return True, requested_amount, "High efficiency bonus allocation"
            elif recent_efficiency >= self.min_efficiency_threshold:
                # Partial approval
                approved_amount = min(requested_amount, int(optimal_quota * 1.2))
                allocation.used_quota += approved_amount
                return True, approved_amount, "Partial allocation based on efficiency"
            else:
                # Deny due to low efficiency
                return False, 0, f"Denied: efficiency too low ({recent_efficiency:.1f} < {self.min_efficiency_threshold})"
        
        else:
            return False, 0, f"Insufficient quota: requested {requested_amount}, available {allocation.remaining_quota}"

    def record_performance(self, provider: str, quota_used: int, tokens_discovered: int, 
                         response_time: float, success: bool):
        """Record performance metrics for quota optimization"""
        performance_entry = {
            'timestamp': datetime.now(),
            'quota_used': quota_used,
            'tokens_discovered': tokens_discovered,
            'response_time': response_time,
            'success': success,
            'efficiency': tokens_discovered / quota_used if quota_used > 0 else 0.0,
            'hour_of_day': datetime.now().hour
        }
        
        self.performance_history[provider].append(performance_entry)
        
        # Keep only last 100 entries per provider
        if len(self.performance_history[provider]) > 100:
            self.performance_history[provider] = self.performance_history[provider][-100:]
        
        # Update allocation efficiency score
        allocation = self.allocations.get(provider)
        if allocation:
            allocation._efficiency_score = self._get_recent_efficiency(provider)
        
        # Check if reallocation needed
        if self.learning_enabled and datetime.now() - self.last_reallocation > self.reallocation_interval:
            self._reallocate_quotas()

    def _get_recent_efficiency(self, provider: str) -> float:
        """Get recent efficiency score for a provider"""
        history = self.performance_history.get(provider, [])
        if not history:
            return 0.0
        
        # Calculate efficiency from last 10 entries
        recent_entries = history[-10:]
        total_quota = sum(entry['quota_used'] for entry in recent_entries)
        total_tokens = sum(entry['tokens_discovered'] for entry in recent_entries)
        
        return total_tokens / total_quota if total_quota > 0 else 0.0

    def _reallocate_quotas(self):
        """Intelligently reallocate quotas based on performance"""
        logger.info("Starting adaptive quota reallocation...")
        
        # Calculate efficiency scores
        efficiencies = {}
        for provider in self.allocations:
            efficiencies[provider] = self._get_recent_efficiency(provider)
        
        # Reallocate based on relative performance
        total_efficiency = sum(efficiencies.values())
        if total_efficiency > 0:
            for provider, allocation in self.allocations.items():
                efficiency_ratio = efficiencies[provider] / total_efficiency
                
                # Adjust allocation based on performance (within limits)
                if efficiency_ratio > 0.6:  # High performing API
                    new_allocation = min(
                        int(allocation.total_quota * 0.95),  # Max 95%
                        int(allocation.total_quota * (0.7 + efficiency_ratio * 0.3))
                    )
                elif efficiency_ratio < 0.3:  # Low performing API
                    new_allocation = max(
                        int(allocation.total_quota * 0.5),   # Min 50%
                        int(allocation.total_quota * (0.5 + efficiency_ratio * 0.4))
                    )
                else:  # Balanced performance
                    new_allocation = int(allocation.total_quota * 0.85)  # Standard 85%
                
                allocation.allocated_quota = new_allocation
                allocation.reserved_quota = allocation.total_quota - allocation.allocated_quota
                
                logger.info(f"Reallocated {provider}: {allocation.allocated_quota}/{allocation.total_quota} "
                           f"(efficiency: {efficiencies[provider]:.1f})")
        
        self.last_reallocation = datetime.now()

    def enter_emergency_mode(self, reason: str):
        """Enter emergency quota conservation mode"""
        if not self.emergency_mode:
            self.emergency_mode = True
            logger.warning(f"Entering emergency quota mode: {reason}")
            
            # Reduce all allocations to conservative levels
            for provider, allocation in self.allocations.items():
                if provider == 'solana_tracker':
                    # Ultra-conservative for limited API
                    allocation.allocated_quota = int(allocation.total_quota * 0.6)
                    allocation.reserved_quota = allocation.total_quota - allocation.allocated_quota
                    
            logger.info("Emergency quota conservation activated")

    def exit_emergency_mode(self):
        """Exit emergency quota conservation mode"""
        if self.emergency_mode:
            self.emergency_mode = False
            logger.info("Exiting emergency quota mode - returning to balanced allocation")
            
            # Restore balanced allocations
            for provider, allocation in self.allocations.items():
                allocation.allocated_quota = int(allocation.total_quota * 0.85)
                allocation.reserved_quota = allocation.total_quota - allocation.allocated_quota

    def check_quota_health(self) -> Dict[str, Any]:
        """Check overall quota health and trigger alerts if needed"""
        health_report = {
            'overall_status': 'healthy',
            'emergency_mode': self.emergency_mode,
            'alerts': [],
            'recommendations': []
        }
        
        for provider, allocation in self.allocations.items():
            utilization = allocation.utilization_rate
            efficiency = allocation.efficiency_score
            
            # Check for quota exhaustion risk
            if utilization > 0.9:
                health_report['alerts'].append(f"{provider}: High quota utilization ({utilization:.1%})")
                if utilization > 0.95:
                    health_report['overall_status'] = 'critical'
            
            # Check for low efficiency
            if efficiency < self.min_efficiency_threshold and allocation.used_quota > 10:
                health_report['alerts'].append(f"{provider}: Low efficiency ({efficiency:.1f} tokens/quota)")
                health_report['recommendations'].append(f"Consider reducing {provider} usage")
            
            # Check for emergency mode triggers
            if provider == 'solana_tracker' and allocation.remaining_quota < allocation.reserved_quota:
                if not self.emergency_mode:
                    self.enter_emergency_mode(f"{provider} quota critically low")
                    health_report['alerts'].append(f"Emergency mode activated for {provider}")
        
        return health_report

    def get_quota_status(self) -> Dict[str, Any]:
        """Get comprehensive quota status for all providers"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'emergency_mode': self.emergency_mode,
            'hour_multiplier': self.get_current_hour_multiplier(),
            'providers': {}
        }
        
        for provider, allocation in self.allocations.items():
            provider_status = {
                'total_quota': allocation.total_quota,
                'allocated_quota': allocation.allocated_quota,
                'used_quota': allocation.used_quota,
                'remaining_quota': allocation.remaining_quota,
                'reserved_quota': allocation.reserved_quota,
                'utilization_rate': f"{allocation.utilization_rate:.1%}",
                'efficiency_score': allocation.efficiency_score,
                'allocation_strategy': allocation.allocation_strategy.value,
                'optimal_hour_quota': self.calculate_optimal_quota_for_timeframe(provider, 1.0)
            }
            
            status['providers'][provider] = provider_status
        
        return status

    def reset_daily_quotas(self):
        """Reset quota counters for new day"""
        for provider, allocation in self.allocations.items():
            allocation.used_quota = 0
            
        # Reset emergency mode if it was quota-related
        if self.emergency_mode:
            self.exit_emergency_mode()
            
        logger.info("Daily quotas reset - new trading day started")

    def optimize_for_target(self, target_tokens_per_day: int = 2500):
        """Optimize quota allocation to achieve target token discovery"""
        logger.info(f"Optimizing quota allocation for target: {target_tokens_per_day} tokens/day")
        
        # Calculate required efficiency for each provider
        daily_scans = 96  # Based on 15-minute intervals
        tokens_needed_per_scan = target_tokens_per_day / daily_scans  # ~26 tokens/scan
        
        for provider, allocation in self.allocations.items():
            current_efficiency = allocation.efficiency_score
            
            if current_efficiency > 0:
                # Calculate optimal quota allocation
                quota_needed = int(tokens_needed_per_scan / current_efficiency) if current_efficiency > 0 else allocation.total_quota
                optimal_daily_quota = quota_needed * daily_scans
                
                # Adjust allocation within limits
                if provider == 'solana_tracker':
                    # Ensure we don't exceed daily limits
                    optimal_allocation = min(optimal_daily_quota, allocation.total_quota * 0.9)
                else:
                    # GeckoTerminal has higher limits
                    optimal_allocation = min(optimal_daily_quota, allocation.total_quota * 0.8)
                
                allocation.allocated_quota = int(optimal_allocation)
                allocation.reserved_quota = allocation.total_quota - allocation.allocated_quota
                
                logger.info(f"Optimized {provider}: {allocation.allocated_quota} quota "
                           f"(efficiency: {current_efficiency:.1f}, needed: {optimal_daily_quota})")

    def get_performance_insights(self) -> Dict[str, Any]:
        """Generate performance insights for quota optimization"""
        insights = {
            'efficiency_analysis': {},
            'time_based_patterns': {},
            'optimization_opportunities': []
        }
        
        for provider, history in self.performance_history.items():
            if not history:
                continue
                
            # Efficiency analysis
            efficiencies = [entry['efficiency'] for entry in history if entry['success']]
            if efficiencies:
                insights['efficiency_analysis'][provider] = {
                    'avg_efficiency': sum(efficiencies) / len(efficiencies),
                    'max_efficiency': max(efficiencies),
                    'min_efficiency': min(efficiencies),
                    'consistency': 1.0 - (max(efficiencies) - min(efficiencies)) / max(efficiencies, 1)
                }
            
            # Time-based patterns
            hourly_performance = {}
            for entry in history:
                hour = entry['hour_of_day']
                if hour not in hourly_performance:
                    hourly_performance[hour] = {'total_tokens': 0, 'total_quota': 0}
                hourly_performance[hour]['total_tokens'] += entry['tokens_discovered']
                hourly_performance[hour]['total_quota'] += entry['quota_used']
            
            # Calculate hourly efficiency
            hourly_efficiency = {}
            for hour, data in hourly_performance.items():
                efficiency = data['total_tokens'] / data['total_quota'] if data['total_quota'] > 0 else 0
                hourly_efficiency[hour] = efficiency
            
            insights['time_based_patterns'][provider] = hourly_efficiency
        
        # Optimization opportunities
        for provider, allocation in self.allocations.items():
            if allocation.utilization_rate < 0.5 and allocation.efficiency_score > 15:
                insights['optimization_opportunities'].append(
                    f"Increase {provider} usage - low utilization ({allocation.utilization_rate:.1%}) "
                    f"with high efficiency ({allocation.efficiency_score:.1f})"
                )
            elif allocation.utilization_rate > 0.8 and allocation.efficiency_score < 8:
                insights['optimization_opportunities'].append(
                    f"Reduce {provider} usage - high utilization ({allocation.utilization_rate:.1%}) "
                    f"with low efficiency ({allocation.efficiency_score:.1f})"
                )
        
        return insights