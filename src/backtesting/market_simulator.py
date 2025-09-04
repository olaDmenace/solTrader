#!/usr/bin/env python3
"""
Realistic Market Structure Simulation
Advanced order book simulation with realistic latency, slippage, and market impact modeling
for accurate backtesting of trading strategies.

Features:
- Order book depth simulation
- Bid/ask spread modeling
- Market impact calculation
- Execution latency simulation
- Dynamic slippage based on order size
- Cross-DEX synchronization delays
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random
import math

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderBookLevel:
    """Represents a single level in the order book"""
    price: float
    size: float
    orders: int = 1

@dataclass
class OrderBook:
    """Complete order book state"""
    bids: List[OrderBookLevel] = field(default_factory=list)  # Sorted descending by price
    asks: List[OrderBookLevel] = field(default_factory=list)  # Sorted ascending by price
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0
    
    @property
    def spread_percentage(self) -> float:
        if self.best_bid and self.spread > 0:
            return (self.spread / self.best_bid) * 100
        return 0.0
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    def get_bid_depth(self, levels: int = 10) -> float:
        """Get total bid depth for specified levels"""
        return sum(level.size for level in self.bids[:levels])
    
    def get_ask_depth(self, levels: int = 10) -> float:
        """Get total ask depth for specified levels"""
        return sum(level.size for level in self.asks[:levels])

@dataclass
class MarketImpact:
    """Market impact calculation result"""
    original_price: float
    final_price: float
    impact_percentage: float
    liquidity_consumed: float
    remaining_size: float
    executed_size: float
    average_price: float
    slippage_usd: float

@dataclass
class ExecutionResult:
    """Result of order execution simulation"""
    executed_price: float
    executed_size: float
    remaining_size: float
    market_impact: MarketImpact
    execution_time_ms: float
    gas_cost: float
    total_fees: float
    success: bool
    error_message: str = ""

class OrderBookGenerator:
    """Generates realistic order books based on historical patterns"""
    
    def __init__(self, base_price: float = 100.0, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility
        
        # Market microstructure parameters
        self.typical_spread_bps = 5  # 5 basis points typical spread
        self.depth_decay_factor = 0.9  # How depth decays away from best price
        self.levels_count = 20  # Number of levels per side
        
    def generate_orderbook(self, timestamp: datetime, price_adjustment: float = 0.0) -> OrderBook:
        """Generate realistic order book for given timestamp"""
        try:
            # Calculate current mid price with adjustment
            adjusted_price = self.base_price * (1 + price_adjustment)
            
            # Calculate spread (wider during volatile periods)
            current_volatility = abs(price_adjustment) * 100  # Convert to percentage
            spread_multiplier = 1 + (current_volatility / 100)  # Widen spread with volatility
            spread_bps = self.typical_spread_bps * spread_multiplier
            spread = adjusted_price * (spread_bps / 10000)
            
            best_bid = adjusted_price - (spread / 2)
            best_ask = adjusted_price + (spread / 2)
            
            # Generate bid levels
            bids = []
            for i in range(self.levels_count):
                level_price = best_bid - (i * spread * 0.1)  # Levels spread out
                
                # Size decreases with distance from best price
                base_size = random.uniform(50, 200)  # Base size range
                level_size = base_size * (self.depth_decay_factor ** i)
                
                # Add some randomness
                size_noise = random.uniform(0.8, 1.2)
                final_size = level_size * size_noise
                
                bids.append(OrderBookLevel(
                    price=round(level_price, 6),
                    size=round(final_size, 2),
                    orders=random.randint(1, 5)
                ))
            
            # Generate ask levels
            asks = []
            for i in range(self.levels_count):
                level_price = best_ask + (i * spread * 0.1)
                
                base_size = random.uniform(50, 200)
                level_size = base_size * (self.depth_decay_factor ** i)
                size_noise = random.uniform(0.8, 1.2)
                final_size = level_size * size_noise
                
                asks.append(OrderBookLevel(
                    price=round(level_price, 6),
                    size=round(final_size, 2),
                    orders=random.randint(1, 5)
                ))
            
            return OrderBook(bids=bids, asks=asks, timestamp=timestamp)
            
        except Exception as e:
            logger.error(f"[ORDERBOOK_GEN] Error generating order book: {e}")
            return OrderBook()

class MarketImpactCalculator:
    """Calculates market impact based on order size and order book depth"""
    
    def __init__(self):
        self.linear_impact_coefficient = 0.1  # Linear impact per $1000
        self.sqrt_impact_coefficient = 0.5    # Square root impact
        self.permanent_impact_ratio = 0.3     # Portion that's permanent
        
    def calculate_impact(
        self,
        order_book: OrderBook,
        side: OrderSide,
        size: float,
        aggressive: bool = True
    ) -> MarketImpact:
        """
        Calculate market impact for given order
        
        Args:
            order_book: Current order book state
            side: BUY or SELL
            size: Order size in base currency
            aggressive: True for market orders, False for limit orders
        """
        try:
            if side == OrderSide.BUY:
                levels = order_book.asks
                best_price = order_book.best_ask
            else:
                levels = order_book.bids
                best_price = order_book.best_bid
            
            if not levels or not best_price:
                return MarketImpact(
                    original_price=0,
                    final_price=0,
                    impact_percentage=0,
                    liquidity_consumed=0,
                    remaining_size=size,
                    executed_size=0,
                    average_price=0,
                    slippage_usd=0
                )
            
            # Simulate order execution through order book levels
            remaining_size = size
            total_cost = 0.0
            liquidity_consumed = 0.0
            executed_size = 0.0
            
            for level in levels:
                if remaining_size <= 0:
                    break
                
                # Amount to execute at this level
                level_execution = min(remaining_size, level.size)
                
                # Calculate cost at this level
                level_cost = level_execution * level.price
                total_cost += level_cost
                
                # Update counters
                executed_size += level_execution
                remaining_size -= level_execution
                liquidity_consumed += level_execution
                
                # Break if order is filled or we've gone through significant depth
                if remaining_size <= 0 or executed_size >= size * 0.95:
                    break
            
            # Calculate average execution price
            average_price = total_cost / max(executed_size, 0.001)
            
            # Calculate impact metrics
            impact_percentage = ((average_price - best_price) / best_price) * 100
            if side == OrderSide.SELL:
                impact_percentage = -impact_percentage  # Negative for sells
            
            slippage_usd = abs(average_price - best_price) * executed_size
            
            return MarketImpact(
                original_price=best_price,
                final_price=average_price,
                impact_percentage=impact_percentage,
                liquidity_consumed=liquidity_consumed,
                remaining_size=remaining_size,
                executed_size=executed_size,
                average_price=average_price,
                slippage_usd=slippage_usd
            )
            
        except Exception as e:
            logger.error(f"[IMPACT_CALC] Error calculating market impact: {e}")
            return MarketImpact(
                original_price=0, final_price=0, impact_percentage=0,
                liquidity_consumed=0, remaining_size=size, executed_size=0,
                average_price=0, slippage_usd=0
            )

class LatencySimulator:
    """Simulates realistic execution latencies"""
    
    def __init__(self):
        # Network latency parameters (milliseconds)
        self.base_latency = 50      # Base network latency
        self.jitter_range = 20      # Random jitter
        self.congestion_factor = 1.0 # Network congestion multiplier
        
        # Exchange processing latency
        self.exchange_processing = 30  # Exchange processing time
        self.matching_engine_delay = 10  # Matching engine delay
        
        # DEX-specific latencies
        self.dex_latencies = {
            'raydium': {'base': 40, 'jitter': 15},
            'orca': {'base': 45, 'jitter': 18},
            'meteora': {'base': 55, 'jitter': 25},
            'phoenix': {'base': 35, 'jitter': 12}
        }
    
    def simulate_execution_latency(
        self,
        dex_name: str = 'raydium',
        order_size: float = 1000,
        market_volatility: float = 0.02
    ) -> float:
        """
        Simulate execution latency in milliseconds
        
        Factors affecting latency:
        - Base network latency
        - DEX-specific processing time
        - Order size (larger orders take longer)
        - Market volatility (busy markets = higher latency)
        """
        try:
            dex_config = self.dex_latencies.get(dex_name, self.dex_latencies['raydium'])
            
            # Base latency for this DEX
            base = dex_config['base']
            jitter = random.uniform(-dex_config['jitter'], dex_config['jitter'])
            
            # Size-based delay (larger orders take longer)
            size_delay = min(50, math.log10(max(order_size / 1000, 1)) * 10)
            
            # Volatility-based delay (busy markets are slower)
            volatility_delay = market_volatility * 1000  # Convert to ms
            
            # Network congestion simulation
            congestion = random.uniform(0.8, 1.5)  # 80% to 150% of normal
            
            total_latency = (base + jitter + size_delay + volatility_delay) * congestion
            
            # Ensure minimum latency
            return max(10, total_latency)
            
        except Exception as e:
            logger.error(f"[LATENCY_SIM] Error simulating latency: {e}")
            return 50.0  # Default fallback

class RealisticMarketSimulator:
    """
    Complete market structure simulator combining order books, impact calculation,
    and latency modeling for realistic backtesting.
    """
    
    def __init__(self, settings: Any):
        self.settings = settings
        
        # Initialize components
        self.orderbook_generator = OrderBookGenerator()
        self.impact_calculator = MarketImpactCalculator()
        self.latency_simulator = LatencySimulator()
        
        # Market state
        self.current_orderbooks: Dict[str, OrderBook] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Simulation parameters
        self.update_frequency = 1.0  # Order book update frequency (seconds)
        self.price_volatility = 0.02  # Daily price volatility
        
        logger.info("[MARKET_SIM] Realistic market simulator initialized")
    
    def initialize_market(self, tokens: List[str], base_prices: Dict[str, float]):
        """Initialize market state for given tokens"""
        try:
            for token in tokens:
                base_price = base_prices.get(token, 100.0)
                
                # Initialize order book generator for this token
                generator = OrderBookGenerator(base_price=base_price)
                
                # Generate initial order book
                initial_orderbook = generator.generate_orderbook(datetime.now())
                self.current_orderbooks[token] = initial_orderbook
                
                # Initialize price history
                self.price_history[token] = [(datetime.now(), base_price)]
                
            logger.info(f"[MARKET_SIM] Market initialized for {len(tokens)} tokens")
            
        except Exception as e:
            logger.error(f"[MARKET_SIM] Error initializing market: {e}")
    
    async def simulate_order_execution(
        self,
        token: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        dex_name: str = 'raydium'
    ) -> ExecutionResult:
        """
        Simulate realistic order execution with all market structure effects
        """
        try:
            start_time = datetime.now()
            
            # Get current order book
            order_book = self.current_orderbooks.get(token)
            if not order_book:
                return ExecutionResult(
                    executed_price=0,
                    executed_size=0,
                    remaining_size=size,
                    market_impact=MarketImpact(0, 0, 0, 0, size, 0, 0, 0),
                    execution_time_ms=0,
                    gas_cost=0,
                    total_fees=0,
                    success=False,
                    error_message=f"No order book available for token {token}"
                )
            
            # Simulate execution latency
            market_volatility = self._calculate_current_volatility(token)
            execution_latency = self.latency_simulator.simulate_execution_latency(
                dex_name=dex_name,
                order_size=size,
                market_volatility=market_volatility
            )
            
            # Simulate latency delay
            await asyncio.sleep(execution_latency / 1000)  # Convert to seconds
            
            # Update order book during execution delay (market moves)
            await self._update_orderbook_during_execution(token, execution_latency)
            updated_orderbook = self.current_orderbooks[token]
            
            # Calculate market impact
            market_impact = self.impact_calculator.calculate_impact(
                order_book=updated_orderbook,
                side=side,
                size=size,
                aggressive=(order_type == OrderType.MARKET)
            )
            
            # Determine execution success and price
            if order_type == OrderType.MARKET:
                # Market orders execute at market impact price
                executed_price = market_impact.average_price
                executed_size = market_impact.executed_size
                success = executed_size > 0
            else:
                # Limit orders only execute if price is favorable
                best_price = updated_orderbook.best_ask if side == OrderSide.BUY else updated_orderbook.best_bid
                
                if limit_price and best_price:
                    if (side == OrderSide.BUY and limit_price >= best_price) or \
                       (side == OrderSide.SELL and limit_price <= best_price):
                        executed_price = limit_price
                        executed_size = size  # Simplified - assume full fill at limit
                        success = True
                    else:
                        executed_price = 0
                        executed_size = 0
                        success = False
                else:
                    executed_price = market_impact.average_price
                    executed_size = market_impact.executed_size
                    success = executed_size > 0
            
            # Calculate fees and costs
            gas_cost = self._calculate_gas_cost(dex_name, size)
            dex_fees = self._calculate_dex_fees(dex_name, executed_size * executed_price)
            total_fees = gas_cost + dex_fees
            
            # Calculate total execution time
            total_execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ExecutionResult(
                executed_price=executed_price,
                executed_size=executed_size,
                remaining_size=size - executed_size,
                market_impact=market_impact,
                execution_time_ms=total_execution_time,
                gas_cost=gas_cost,
                total_fees=total_fees,
                success=success,
                error_message="" if success else "Execution failed"
            )
            
        except Exception as e:
            logger.error(f"[MARKET_SIM] Error simulating execution: {e}")
            return ExecutionResult(
                executed_price=0,
                executed_size=0,
                remaining_size=size,
                market_impact=MarketImpact(0, 0, 0, 0, size, 0, 0, 0),
                execution_time_ms=0,
                gas_cost=0,
                total_fees=0,
                success=False,
                error_message=str(e)
            )
    
    async def _update_orderbook_during_execution(self, token: str, latency_ms: float):
        """Update order book to simulate market movement during execution"""
        try:
            current_orderbook = self.current_orderbooks.get(token)
            if not current_orderbook:
                return
            
            # Calculate price movement during latency period
            latency_seconds = latency_ms / 1000
            volatility_per_second = self.price_volatility / (24 * 3600)  # Convert daily to per-second
            
            # Price change during execution
            price_change = np.random.normal(0, volatility_per_second * math.sqrt(latency_seconds))
            
            # Generate new order book with price adjustment
            new_orderbook = self.orderbook_generator.generate_orderbook(
                datetime.now(),
                price_adjustment=price_change
            )
            
            self.current_orderbooks[token] = new_orderbook
            
        except Exception as e:
            logger.error(f"[MARKET_SIM] Error updating orderbook: {e}")
    
    def _calculate_current_volatility(self, token: str) -> float:
        """Calculate current market volatility for the token"""
        try:
            history = self.price_history.get(token, [])
            if len(history) < 2:
                return self.price_volatility
            
            # Calculate recent price volatility
            recent_prices = [price for _, price in history[-20:]]  # Last 20 observations
            returns = np.diff(recent_prices) / recent_prices[:-1]
            
            return float(np.std(returns)) if len(returns) > 1 else self.price_volatility
            
        except Exception:
            return self.price_volatility
    
    def _calculate_gas_cost(self, dex_name: str, size: float) -> float:
        """Calculate gas costs based on DEX and trade size"""
        try:
            # Base gas costs by DEX (in SOL)
            base_costs = {
                'raydium': 0.01,
                'orca': 0.012,
                'meteora': 0.015,
                'phoenix': 0.008
            }
            
            base_cost = base_costs.get(dex_name, 0.01)
            
            # Size adjustment (larger trades might need more gas)
            size_multiplier = 1 + (size / 10000) * 0.1  # Small increase for large trades
            
            return base_cost * size_multiplier
            
        except Exception:
            return 0.01
    
    def _calculate_dex_fees(self, dex_name: str, trade_value: float) -> float:
        """Calculate DEX trading fees"""
        try:
            # Fee rates by DEX (percentage)
            fee_rates = {
                'raydium': 0.0025,    # 0.25%
                'orca': 0.0030,       # 0.30%
                'meteora': 0.0020,    # 0.20%
                'phoenix': 0.0035     # 0.35%
            }
            
            fee_rate = fee_rates.get(dex_name, 0.0025)
            return trade_value * fee_rate
            
        except Exception:
            return 0.0
    
    def get_current_market_data(self, token: str) -> Dict[str, Any]:
        """Get current market data for a token"""
        try:
            orderbook = self.current_orderbooks.get(token)
            if not orderbook:
                return {}
            
            return {
                'best_bid': orderbook.best_bid,
                'best_ask': orderbook.best_ask,
                'mid_price': orderbook.mid_price,
                'spread': orderbook.spread,
                'spread_percentage': orderbook.spread_percentage,
                'bid_depth_5': orderbook.get_bid_depth(5),
                'ask_depth_5': orderbook.get_ask_depth(5),
                'total_levels': len(orderbook.bids) + len(orderbook.asks),
                'timestamp': orderbook.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"[MARKET_SIM] Error getting market data: {e}")
            return {}
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get overall simulation statistics"""
        try:
            return {
                'active_tokens': len(self.current_orderbooks),
                'total_price_updates': sum(len(history) for history in self.price_history.values()),
                'avg_spread_percentage': np.mean([
                    ob.spread_percentage for ob in self.current_orderbooks.values()
                    if ob.spread_percentage > 0
                ]) if self.current_orderbooks else 0,
                'simulation_parameters': {
                    'update_frequency': self.update_frequency,
                    'price_volatility': self.price_volatility
                }
            }
            
        except Exception as e:
            logger.error(f"[MARKET_SIM] Error getting stats: {e}")
            return {}