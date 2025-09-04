# src/trading/types.py

from typing import TypeVar, Protocol, Dict, Any, Optional, List, runtime_checkable
from datetime import datetime
from dataclasses import dataclass

T = TypeVar('T')
SettingsType = TypeVar('SettingsType')
JupiterClient = TypeVar('JupiterClient')
Wallet = TypeVar('Wallet')
Scanner = TypeVar('Scanner')

@dataclass
class MarketData:
    price: float
    volume_24h: float
    liquidity: float
    bid_depth: float
    ask_depth: float
    last_update: datetime

@dataclass
class TradeResult:
    success: bool
    transaction_hash: Optional[str]
    price: float
    size: float
    timestamp: datetime
    error: Optional[str] = None

@runtime_checkable
class StrategyProtocol(Protocol):
    is_trading: bool
    
    async def start_trading(self) -> bool: ...
    async def stop_trading(self) -> bool: ...
    async def get_metrics(self) -> Dict[str, Dict[str, Any]]: ...