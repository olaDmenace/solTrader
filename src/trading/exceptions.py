# class TradingStrategyError(Exception):
#     pass

# class InsufficientBalanceError(TradingStrategyError):
#     pass

# class ValidationError(TradingStrategyError):
#     pass

"""
exceptions.py - Trading system exceptions
"""

class TradingStrategyError(Exception):
    """Base class for trading strategy exceptions"""
    pass

class InsufficientBalanceError(TradingStrategyError):
    """Raised when account balance is insufficient"""
    pass

class ValidationError(TradingStrategyError):
    """Raised when validation fails"""
    pass

class MarketConditionError(TradingStrategyError):
    """Raised when market conditions are unfavorable"""
    pass

class ExecutionError(TradingStrategyError):
    """Raised when trade execution fails"""
    pass