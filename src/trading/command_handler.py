# src/trading/command_handler.py

import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime
import asyncio
import time
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class CommandHistory:
    command: str
    args: Dict[str, Any]
    status: str
    error: Optional[str]
    timestamp: datetime

class RateLimiter:
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[float] = []

    def is_allowed(self) -> bool:
        now = time.time()
        self.calls = [t for t in self.calls if now - t <= self.time_window]
        if len(self.calls) >= self.max_calls:
            return False
        self.calls.append(now)
        return True

class CommandValidator:
    @staticmethod
    def validate_command(command: str, args: Dict[str, Any]) -> bool:
        required_args = {
            'start_trading': ['max_positions', 'max_trade_size'],
            'place_trade': ['token_address', 'amount', 'side'],
            'cancel_trade': ['order_id'],
        }
        return all(arg in args for arg in required_args.get(command, []))

class CommandHandler:
    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        self.history: List[CommandHistory] = []
        self.rate_limiter = RateLimiter(max_calls, time_window)
        self.validator = CommandValidator()

    def rate_limit_decorator(self, func: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
        @wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
            if not self.rate_limiter.is_allowed():
                raise Exception("Rate limit exceeded")
            return await func(self, *args, **kwargs)
        return wrapper

    @property
    def rate_limit(self) -> Callable[..., Any]:
        return self.rate_limit_decorator

    async def execute_command(self, command: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.validator.validate_command(command, args):
                raise ValueError(f"Invalid command arguments for {command}")

            result = await self._process_command(command, args)
            if result is None:
                result = {"status": "error", "message": "Command processing failed"}
                
            self._record_history(command, args, "success")
            return result

        except Exception as e:
            error_msg = str(e)
            self._record_history(command, args, "failed", error_msg)
            return {"status": "error", "message": error_msg}

    def _record_history(self, command: str, args: Dict[str, Any], 
                       status: str, error: Optional[str] = None) -> None:
        self.history.append(CommandHistory(
            command=command,
            args=args,
            status=status,
            error=error,
            timestamp=datetime.now()
        ))

    async def _process_command(self, command: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder implementation
        return {"status": "success", "command": command, "args": args}

    def get_command_history(self) -> List[CommandHistory]:
        return self.history[-100:]  # Return last 100 commands