import logging
from typing import Dict, Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from src.api.jupiter import JupiterClient
from src.api.alchemy import AlchemyClient

logger = logging.getLogger(__name__)

class RecoveryAction(Enum):
    RETRY = "retry"
    ROLLBACK = "rollback"
    RESET = "reset"
    HALT = "halt"

@dataclass
class ComponentState:
    name: str
    healthy: bool
    last_error: Optional[str] = None
    last_recovery: Optional[datetime] = None
    recovery_attempts: int = 0

class ErrorRecoveryManager:
    def __init__(self, strategy: Any):
        self.strategy = strategy
        self.component_states: Dict[str, ComponentState] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        self.max_retries = 3
        self.cooldown_period = 300  # 5 minutes
        self._monitor_task: Optional[asyncio.Task] = None

    async def register_recovery_handlers(self) -> None:
        """Register default recovery handlers"""
        self.register_component('wallet', self._wallet_recovery_handler)
        self.register_component('jupiter', self._api_recovery_handler)
        self.register_component('alchemy', self._api_recovery_handler)
        self.register_component('position_manager', self._position_recovery_handler)

    async def _wallet_recovery_handler(self, component: str, params: Dict[str, Any]) -> bool:
        """Recovery handler for wallet component"""
        action = params.get('action')
        if action == 'retry':
            try:
                await self.strategy.wallet.reconnect()
                return True
            except Exception:
                return False
        elif action == 'reset':
            try:
                await self.strategy.wallet.disconnect()
                await asyncio.sleep(1)
                await self.strategy.wallet.connect()
                return True
            except Exception:
                return False
        return False

    async def _api_recovery_handler(self, component: str, params: Dict[str, Any]) -> bool:
        """Recovery handler for API clients"""
        action = params.get('action')
        if action == 'retry':
            try:
                if component == 'jupiter':
                    return await self.strategy.jupiter.test_connection()
                elif component == 'alchemy':
                    return await self.strategy.alchemy.test_connection()
            except Exception:
                return False
        elif action == 'reset':
            try:
                if component == 'jupiter':
                    await self.strategy.jupiter.close()
                    self.strategy.jupiter = JupiterClient()
                elif component == 'alchemy':
                    await self.strategy.alchemy.close()
                    self.strategy.alchemy = AlchemyClient(self.strategy.settings.ALCHEMY_RPC_URL)
                return True
            except Exception:
                return False
        return False

    async def _position_recovery_handler(self, component: str, params: Dict[str, Any]) -> bool:
        """Recovery handler for position manager"""
        action = params.get('action')
        if action == 'retry':
            try:
                positions = self.strategy.position_manager.get_open_positions()
                for addr, pos in positions.items():
                    price = await self.strategy._get_current_price(addr)
                    if price:
                        pos.update_price(price)
                return True
            except Exception:
                return False
        elif action == 'reset':
            try:
                if self.strategy.settings.CLOSE_POSITIONS_ON_STOP:
                    positions = list(self.strategy.position_manager.positions.keys())
                    for addr in positions:
                        await self.strategy.close_position(addr, "Position manager reset")
                return True
            except Exception:
                return False
        return False

    def register_component(self, name: str, 
                         recovery_handler: Callable[[str, Any], Awaitable[bool]]) -> None:
        """Register a component for error recovery monitoring"""
        self.component_states[name] = ComponentState(name=name, healthy=True)
        self.recovery_handlers[name] = recovery_handler

    async def handle_error(self, component: str, error: Exception) -> bool:
        """Handle component error and attempt recovery"""
        try:
            state = self.component_states.get(component)
            if not state:
                logger.error(f"Unregistered component error: {component}")
                return False

            state.healthy = False
            state.last_error = str(error)
            state.recovery_attempts += 1

            if state.recovery_attempts > self.max_retries:
                await self._handle_critical_failure(component)
                return False

            action = self._determine_recovery_action(state, error)
            return await self._execute_recovery(component, action)

        except Exception as e:
            logger.error(f"Error in recovery handler: {str(e)}")
            return False

    # ... [rest of the class implementation remains the same]

    def _determine_recovery_action(self, state: ComponentState, 
                                 error: Exception) -> RecoveryAction:
        """Determine appropriate recovery action"""
        if state.recovery_attempts >= self.max_retries:
            return RecoveryAction.HALT

        if isinstance(error, (ConnectionError, TimeoutError)):
            return RecoveryAction.RETRY

        if "state corruption" in str(error).lower():
            return RecoveryAction.RESET

        return RecoveryAction.ROLLBACK

    async def _execute_recovery(self, component: str, 
                              action: RecoveryAction) -> bool:
        """Execute recovery action"""
        try:
            handler = self.recovery_handlers.get(component)
            if not handler:
                return False

            state = self.component_states[component]

            if action == RecoveryAction.RETRY:
                success = await self._retry_operation(component, handler)
            elif action == RecoveryAction.ROLLBACK:
                success = await self._rollback_state(component, handler)
            elif action == RecoveryAction.RESET:
                success = await self._reset_component(component, handler)
            else:  # HALT
                await self._handle_critical_failure(component)
                return False

            if success:
                state.healthy = True
                state.last_recovery = datetime.now()
                state.recovery_attempts = 0
                return True

            return False

        except Exception as e:
            logger.error(f"Recovery execution error: {str(e)}")
            return False

    async def _retry_operation(self, component: str, 
                             handler: Callable) -> bool:
        """Retry failed operation with exponential backoff"""
        state = self.component_states[component]
        delay = 2 ** state.recovery_attempts

        await asyncio.sleep(delay)
        return await handler(component, {"action": "retry"})

    async def _rollback_state(self, component: str, 
                            handler: Callable) -> bool:
        """Rollback component to last known good state"""
        return await handler(component, {"action": "rollback"})

    async def _reset_component(self, component: str, 
                             handler: Callable) -> bool:
        """Reset component to initial state"""
        return await handler(component, {"action": "reset"})

    async def _handle_critical_failure(self, component: str) -> None:
        """Handle critical component failure"""
        logger.critical(f"Critical failure in component: {component}")
        
        # Stop trading if critical component fails
        if self.strategy.is_trading:
            await self.strategy.stop_trading()

        # Notify about critical failure
        if hasattr(self.strategy, 'alert_system'):
            await self.strategy.alert_system.emit_alert(
                level="critical",
                type="component_failure",
                message=f"Critical failure in {component}",
                data={
                    "component": component,
                    "error": self.component_states[component].last_error,
                    "recovery_attempts": self.component_states[component].recovery_attempts
                }
            )

    async def start_monitoring(self) -> None:
        """Start component health monitoring"""
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop component health monitoring"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self) -> None:
        """Monitor component health status"""
        while True:
            try:
                await self._check_component_health()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                await asyncio.sleep(5)

    async def _check_component_health(self) -> None:
        """Check health of all components"""
        for component, state in self.component_states.items():
            if not state.healthy:
                # Skip if in cooldown
                if state.last_recovery and \
                   (datetime.now() - state.last_recovery).total_seconds() < self.cooldown_period:
                    continue
                
                handler = self.recovery_handlers.get(component)
                if handler:
                    try:
                        is_healthy = await handler(component, {"action": "health_check"})
                        state.healthy = is_healthy
                    except Exception as e:
                        logger.error(f"Health check error for {component}: {str(e)}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            "components": {
                name: {
                    "healthy": state.healthy,
                    "last_error": state.last_error,
                    "last_recovery": state.last_recovery.isoformat() if state.last_recovery else None,
                    "recovery_attempts": state.recovery_attempts
                }
                for name, state in self.component_states.items()
            },
            "overall_health": all(state.healthy for state in self.component_states.values())
        }