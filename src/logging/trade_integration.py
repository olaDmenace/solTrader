#!/usr/bin/env python3
"""
Trade Logger Integration Utilities
Provides easy integration of the centralized trade logger with existing trading strategies.
"""

import logging
from typing import Dict, Any, Optional
from .trade_logger import CentralizedTradeLogger, TradeType, TradeStatus

logger = logging.getLogger(__name__)

class TradeLoggerMixin:
    """
    Mixin class to add trade logging capabilities to existing strategy classes
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trade_logger: Optional[CentralizedTradeLogger] = None
    
    def set_trade_logger(self, trade_logger: CentralizedTradeLogger):
        """Set the trade logger instance"""
        self.trade_logger = trade_logger
    
    def log_trade_request(
        self,
        strategy_name: str,
        token_address: str,
        token_symbol: str,
        is_buy: bool,
        theoretical_price: float,
        requested_price: float,
        requested_size: float,
        dex_name: str = "",
        signal_strength: float = 0.0,
        risk_score: float = 0.0,
        confidence_level: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a trade request with standardized parameters
        
        Returns:
            trade_id: Unique identifier for tracking this trade
        """
        if not self.trade_logger:
            logger.warning(f"[{strategy_name}] Trade logger not initialized - skipping log")
            return ""
        
        try:
            # Determine trade type
            trade_type = TradeType.BUY if is_buy else TradeType.SELL
            
            # Add strategy-specific metadata
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                'strategy_name': strategy_name,
                'logging_version': '1.0'
            })
            
            trade_id = self.trade_logger.log_trade_request(
                strategy=strategy_name,
                token_address=token_address,
                token_symbol=token_symbol,
                trade_type=trade_type,
                theoretical_price=theoretical_price,
                requested_price=requested_price,
                requested_size=requested_size,
                dex_name=dex_name,
                signal_strength=signal_strength,
                risk_score=risk_score,
                confidence_level=confidence_level,
                metadata=enhanced_metadata
            )
            
            return trade_id
            
        except Exception as e:
            logger.error(f"[{strategy_name}] Error logging trade request: {e}")
            return ""
    
    def log_trade_execution(
        self,
        trade_id: str,
        executed_price: float,
        executed_size: float,
        execution_time_ms: float,
        gas_fee: float = 0.0,
        dex_fees: float = 0.0,
        success: bool = True,
        notes: str = ""
    ):
        """
        Log trade execution results
        """
        if not self.trade_logger or not trade_id:
            return
        
        try:
            status = TradeStatus.EXECUTED if success else TradeStatus.FAILED
            
            self.trade_logger.log_trade_execution(
                trade_id=trade_id,
                executed_price=executed_price,
                executed_size=executed_size,
                execution_time_ms=execution_time_ms,
                gas_fee=gas_fee,
                dex_fees=dex_fees,
                status=status,
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"Error logging trade execution for {trade_id}: {e}")

def get_trade_type_for_strategy(strategy_name: str, is_buy: bool) -> TradeType:
    """
    Get appropriate trade type based on strategy and direction
    """
    strategy_lower = strategy_name.lower()
    
    if 'arbitrage' in strategy_lower:
        return TradeType.ARBITRAGE_BUY if is_buy else TradeType.ARBITRAGE_SELL
    elif 'grid' in strategy_lower:
        return TradeType.GRID_BUY if is_buy else TradeType.GRID_SELL
    else:
        return TradeType.BUY if is_buy else TradeType.SELL

def create_trade_metadata(
    signal_data: Optional[Dict[str, Any]] = None,
    market_data: Optional[Dict[str, Any]] = None,
    risk_data: Optional[Dict[str, Any]] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized trade metadata from various data sources
    """
    metadata = {}
    
    if signal_data:
        metadata['signal'] = {
            'strength': signal_data.get('strength', 0),
            'confidence': signal_data.get('confidence', 0),
            'type': signal_data.get('type', 'unknown'),
            'components': signal_data.get('components', [])
        }
    
    if market_data:
        metadata['market'] = {
            'price': market_data.get('price', 0),
            'volume_24h': market_data.get('volume_24h', 0),
            'liquidity': market_data.get('liquidity', 0),
            'volatility': market_data.get('volatility', 0)
        }
    
    if risk_data:
        metadata['risk'] = {
            'score': risk_data.get('risk_score', 0),
            'factors': risk_data.get('risk_factors', []),
            'position_size': risk_data.get('position_size', 0)
        }
    
    if additional_data:
        metadata['additional'] = additional_data
    
    return metadata