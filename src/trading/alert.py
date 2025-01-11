import logging
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    level: str  
    type: str   
    message: str
    timestamp: datetime
    data: Dict[str, Any]

class AlertSystem:
    def __init__(self, settings):
        self.settings = settings
        self.alerts: List[Alert] = []
        self.handlers: Dict[str, List[Callable]] = {
            'info': [],
            'warning': [],
            'error': [],
            'critical': []
        }
        self._monitor_task: Optional[asyncio.Task] = None

    async def emit_alert(self, level: str, type: str, message: str, data: Dict[str, Any]) -> None:
        alert = Alert(
            level=level,
            type=type,
            message=message,
            timestamp=datetime.now(),
            data=data
        )
        self.alerts.append(alert)
        await self._process_alert(alert)

    async def _process_alert(self, alert: Alert) -> None:
        try:
            handlers = self.handlers.get(alert.level, [])
            for handler in handlers:
                await handler(alert)
            
            if alert.level in ['error', 'critical']:
                logger.error(f"{alert.type}: {alert.message}")
            else:
                logger.info(f"{alert.type}: {alert.message}")

        except Exception as e:
            logger.error(f"Error processing alert: {str(e)}")

    def add_handler(self, level: str, handler: Callable) -> None:
        if level in self.handlers:
            self.handlers[level].append(handler)

    async def price_alert(self, token_address: str, price: float, threshold: float) -> None:
        await self.emit_alert(
            'info',
            'price',
            f"Price threshold reached for {token_address}",
            {'price': price, 'threshold': threshold}
        )

    async def position_alert(self, position_id: str, status: str, pnl: float) -> None:
        await self.emit_alert(
            'info',
            'position',
            f"Position {status} for {position_id}",
            {'position_id': position_id, 'status': status, 'pnl': pnl}
        )

    async def risk_alert(self, message: str, risk_level: str, metrics: Dict[str, Any]) -> None:
        await self.emit_alert(
            'warning' if risk_level == 'high' else 'info',
            'risk',
            message,
            {'risk_level': risk_level, 'metrics': metrics}
        )

    async def performance_alert(self, metric: str, value: float, threshold: float) -> None:
        await self.emit_alert(
            'info',
            'performance',
            f"Performance metric {metric} reached {value}",
            {'metric': metric, 'value': value, 'threshold': threshold}
        )

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        recent = self.alerts[-limit:] if self.alerts else []
        return [
            {
                'level': alert.level,
                'type': alert.type,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            for alert in recent
        ]

    def save_alerts(self, filepath: str) -> None:
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_recent_alerts(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alerts: {str(e)}")