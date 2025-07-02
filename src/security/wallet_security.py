"""
Wallet Security Module
Enhanced security measures for wallet operations and transaction validation
"""

import logging
import os
import re
import base58
from datetime import datetime, timedelta
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class WalletSecurity:
    """Enhanced wallet security utilities"""
    
    @staticmethod
    def validate_private_key(private_key: str) -> bool:
        """Validate private key format and entropy"""
        try:
            if not private_key or not isinstance(private_key, str):
                return False
                
            # Remove whitespace
            private_key = private_key.strip()
            
            # Check if base58 encoded
            try:
                decoded = base58.b58decode(private_key)
                if len(decoded) != 32:
                    return False
                    
                # Check entropy
                return WalletSecurity._has_sufficient_entropy(decoded)
                
            except Exception:
                return False
                
        except Exception:
            return False
    
    @staticmethod
    def _has_sufficient_entropy(key_bytes: bytes) -> bool:
        """Check if key has sufficient entropy"""
        try:
            # Count unique bytes
            unique_bytes = len(set(key_bytes))
            if unique_bytes < 32:  # Should have reasonable variety
                return False
                
            # Check for repeated patterns
            for i in range(0, len(key_bytes) - 8, 4):
                chunk = key_bytes[i:i+4]
                if key_bytes.count(chunk) > 2:  # Same 4-byte pattern repeated
                    return False
                    
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from log data"""
        sensitive_keys = [
            'private_key', 'privateKey', 'secret', 'password', 'key',
            'signature', 'seed', 'mnemonic', 'passphrase'
        ]
        
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = WalletSecurity.sanitize_log_data(value)
            elif isinstance(value, str) and len(value) > 20:
                # Potential sensitive string - mask middle
                sanitized[key] = f"{value[:4]}...{value[-4:]}"
            else:
                sanitized[key] = value
                
        return sanitized
    
    @staticmethod
    def validate_wallet_address(address: str) -> bool:
        """Validate Solana wallet address format"""
        try:
            if not address or not isinstance(address, str):
                return False
                
            # Basic length check
            if len(address) < 32 or len(address) > 44:
                return False
                
            # Validate base58
            decoded = base58.b58decode(address)
            return len(decoded) == 32
            
        except Exception:
            return False
    
    @staticmethod
    def validate_transaction_params(params: Dict[str, Any]) -> bool:
        """Validate transaction parameters for security"""
        try:
            # Check required fields
            required_fields = ['amount', 'token_address']
            for field in required_fields:
                if field not in params:
                    logger.error(f"Missing required field: {field}")
                    return False
                    
            # Validate amount
            amount = params.get('amount')
            if not isinstance(amount, (int, float)) or amount <= 0:
                logger.error("Invalid amount specified")
                return False
                
            # Check for reasonable amount limits
            max_amount = float(os.getenv('MAX_TRANSACTION_AMOUNT', '1000'))
            if amount > max_amount:
                logger.error(f"Amount exceeds maximum: {amount} > {max_amount}")
                return False
                
            # Validate token address
            token_address = params.get('token_address')
            if not WalletSecurity.validate_wallet_address(token_address):
                logger.error("Invalid token address")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating transaction params: {e}")
            return False


class RateLimiter:
    """Rate limiting for API calls and transactions"""
    
    def __init__(self):
        self.calls: Dict[str, List[datetime]] = {}
        self.limits = {
            'transaction': {'count': 10, 'window': 60},  # 10 per minute
            'api_call': {'count': 100, 'window': 60},   # 100 per minute
            'price_check': {'count': 30, 'window': 60}  # 30 per minute
        }
    
    def is_allowed(self, operation_type: str) -> bool:
        """Check if operation is allowed under rate limits"""
        try:
            if operation_type not in self.limits:
                return True  # No limit configured
                
            now = datetime.now()
            limit_config = self.limits[operation_type]
            window_seconds = limit_config['window']
            max_calls = limit_config['count']
            
            # Initialize if not exists
            if operation_type not in self.calls:
                self.calls[operation_type] = []
                
            # Clean old calls
            cutoff = now - timedelta(seconds=window_seconds)
            self.calls[operation_type] = [
                call_time for call_time in self.calls[operation_type]
                if call_time > cutoff
            ]
            
            # Check limit
            if len(self.calls[operation_type]) >= max_calls:
                logger.warning(f"Rate limit exceeded for {operation_type}")
                return False
                
            # Record this call
            self.calls[operation_type].append(now)
            return True
            
        except Exception as e:
            logger.error(f"Error in rate limiter: {e}")
            return True  # Allow on error
    
    def reset_limits(self) -> None:
        """Reset all rate limits"""
        self.calls.clear()
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current rate limit status"""
        status = {}
        now = datetime.now()
        
        for operation_type, limit_config in self.limits.items():
            if operation_type not in self.calls:
                current_count = 0
            else:
                # Count recent calls
                cutoff = now - timedelta(seconds=limit_config['window'])
                current_count = len([
                    call_time for call_time in self.calls[operation_type]
                    if call_time > cutoff
                ])
                
            status[operation_type] = {
                'current_count': current_count,
                'max_count': limit_config['count'],
                'window_seconds': limit_config['window'],
                'remaining': limit_config['count'] - current_count
            }
            
        return status


class InputValidator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_token_address(address: str) -> bool:
        """Validate token mint address"""
        return WalletSecurity.validate_wallet_address(address)
    
    @staticmethod
    def validate_amount(amount: Any, min_amount: float = 0.0001, max_amount: float = 1000000) -> bool:
        """Validate trading amount"""
        try:
            if not isinstance(amount, (int, float)):
                return False
                
            amount = float(amount)
            return min_amount <= amount <= max_amount
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_slippage(slippage: Any) -> bool:
        """Validate slippage percentage"""
        try:
            if not isinstance(slippage, (int, float)):
                return False
                
            slippage = float(slippage)
            return 0.001 <= slippage <= 0.5  # 0.1% to 50%
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_priority_fee(fee: Any) -> bool:
        """Validate priority fee in microlamports"""
        try:
            if fee is None:
                return True  # Optional parameter
                
            if not isinstance(fee, (int, float)):
                return False
                
            fee = int(fee)
            return 0 <= fee <= 1000000  # 0 to 1M microlamports
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 100) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            return ""
            
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[^a-zA-Z0-9\-_.,\s]', '', input_str)
        
        # Limit length
        return sanitized[:max_length]
    
    @staticmethod
    def validate_network_config(config: Dict[str, Any]) -> bool:
        """Validate network configuration"""
        try:
            required_fields = ['rpc_url']
            for field in required_fields:
                if field not in config:
                    return False
                    
            # Validate RPC URL
            rpc_url = config.get('rpc_url')
            if not isinstance(rpc_url, str) or not rpc_url.startswith(('http://', 'https://')):
                return False
                
            return True
            
        except Exception:
            return False


# Global instances
_rate_limiter = RateLimiter()

def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    return _rate_limiter