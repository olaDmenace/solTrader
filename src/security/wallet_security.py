"""
wallet_security.py - Security measures for wallet and private key handling
"""
import logging
import os
import re
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import base58

logger = logging.getLogger(__name__)


class WalletSecurity:
    """Security utilities for wallet operations"""
    
    @staticmethod
    def validate_private_key(private_key: str) -> bool:
        """
        Validate private key format and security
        
        Args:
            private_key: Base58 encoded private key string
            
        Returns:
            bool: True if valid and secure, False otherwise
        """
        try:
            if not private_key or not isinstance(private_key, str):
                logger.error("Private key must be a non-empty string")
                return False
                
            # Check for common weak patterns
            if WalletSecurity._is_weak_key(private_key):
                logger.error("Private key appears to be weak or test key")
                return False
                
            # Validate base58 format
            try:
                decoded = base58.b58decode(private_key)
                if len(decoded) != 64:
                    logger.error("Private key must decode to exactly 64 bytes")
                    return False
            except Exception:
                logger.error("Invalid base58 encoding in private key")
                return False
                
            # Check entropy (basic)
            if not WalletSecurity._has_sufficient_entropy(decoded):
                logger.error("Private key has insufficient entropy")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating private key: {e}")
            return False
    
    @staticmethod
    def _is_weak_key(private_key: str) -> bool:
        """Check for common weak key patterns"""
        weak_patterns = [
            r"^1{44,}$",  # All ones
            r"^A{44,}$",  # All A's (common test pattern)
            r"^1234",     # Starts with 1234
            r"test",      # Contains "test"
            r"demo",      # Contains "demo"
            r"example",   # Contains "example"
        ]
        
        for pattern in weak_patterns:
            if re.search(pattern, private_key, re.IGNORECASE):
                return True
                
        return False
    
    @staticmethod
    def _has_sufficient_entropy(key_bytes: bytes) -> bool:\n        \"\"\"Check if key has sufficient entropy\"\"\"\n        try:\n            # Count unique bytes\n            unique_bytes = len(set(key_bytes))\n            if unique_bytes < 32:  # Should have reasonable variety\n                return False\n                \n            # Check for repeated patterns\n            for i in range(0, len(key_bytes) - 8, 4):\n                chunk = key_bytes[i:i+4]\n                if key_bytes.count(chunk) > 2:  # Same 4-byte pattern repeated\n                    return False\n                    \n            return True\n            \n        except Exception:\n            return False\n    \n    @staticmethod\n    def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Remove sensitive information from log data\"\"\"\n        sensitive_keys = [\n            'private_key', 'privateKey', 'secret', 'password', 'key',\n            'signature', 'seed', 'mnemonic', 'passphrase'\n        ]\n        \n        sanitized = {}\n        for key, value in data.items():\n            if any(sensitive in key.lower() for sensitive in sensitive_keys):\n                sanitized[key] = \"[REDACTED]\"\n            elif isinstance(value, dict):\n                sanitized[key] = WalletSecurity.sanitize_log_data(value)\n            elif isinstance(value, str) and len(value) > 20:\n                # Potential sensitive string - mask middle\n                sanitized[key] = f\"{value[:4]}...{value[-4:]}\"\n            else:\n                sanitized[key] = value\n                \n        return sanitized\n    \n    @staticmethod\n    def validate_wallet_address(address: str) -> bool:\n        \"\"\"Validate Solana wallet address format\"\"\"\n        try:\n            if not address or not isinstance(address, str):\n                return False\n                \n            # Basic length check\n            if len(address) < 32 or len(address) > 44:\n                return False\n                \n            # Validate base58\n            decoded = base58.b58decode(address)\n            return len(decoded) == 32\n            \n        except Exception:\n            return False\n    \n    @staticmethod\n    def validate_transaction_params(params: Dict[str, Any]) -> bool:\n        \"\"\"Validate transaction parameters for security\"\"\"\n        try:\n            # Check required fields\n            required_fields = ['amount', 'token_address']\n            for field in required_fields:\n                if field not in params:\n                    logger.error(f\"Missing required field: {field}\")\n                    return False\n                    \n            # Validate amount\n            amount = params.get('amount')\n            if not isinstance(amount, (int, float)) or amount <= 0:\n                logger.error(\"Invalid amount specified\")\n                return False\n                \n            # Check for reasonable amount limits\n            max_amount = float(os.getenv('MAX_TRANSACTION_AMOUNT', '1000'))\n            if amount > max_amount:\n                logger.error(f\"Amount exceeds maximum: {amount} > {max_amount}\")\n                return False\n                \n            # Validate token address\n            token_address = params.get('token_address')\n            if not WalletSecurity.validate_wallet_address(token_address):\n                logger.error(\"Invalid token address\")\n                return False\n                \n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error validating transaction params: {e}\")\n            return False\n\n\nclass RateLimiter:\n    \"\"\"Rate limiting for API calls and transactions\"\"\"\n    \n    def __init__(self):\n        self.calls: Dict[str, List[datetime]] = {}\n        self.limits = {\n            'transaction': {'count': 10, 'window': 60},  # 10 per minute\n            'api_call': {'count': 100, 'window': 60},   # 100 per minute\n            'price_check': {'count': 30, 'window': 60}  # 30 per minute\n        }\n    \n    def is_allowed(self, operation_type: str) -> bool:\n        \"\"\"Check if operation is allowed under rate limits\"\"\"\n        try:\n            if operation_type not in self.limits:\n                return True  # No limit configured\n                \n            now = datetime.now()\n            limit_config = self.limits[operation_type]\n            window_seconds = limit_config['window']\n            max_calls = limit_config['count']\n            \n            # Initialize if not exists\n            if operation_type not in self.calls:\n                self.calls[operation_type] = []\n                \n            # Clean old calls\n            cutoff = now - timedelta(seconds=window_seconds)\n            self.calls[operation_type] = [\n                call_time for call_time in self.calls[operation_type]\n                if call_time > cutoff\n            ]\n            \n            # Check limit\n            if len(self.calls[operation_type]) >= max_calls:\n                logger.warning(f\"Rate limit exceeded for {operation_type}\")\n                return False\n                \n            # Record this call\n            self.calls[operation_type].append(now)\n            return True\n            \n        except Exception as e:\n            logger.error(f\"Error in rate limiter: {e}\")\n            return True  # Allow on error\n    \n    def reset_limits(self) -> None:\n        \"\"\"Reset all rate limits\"\"\"\n        self.calls.clear()\n    \n    def get_status(self) -> Dict[str, Dict[str, Any]]:\n        \"\"\"Get current rate limit status\"\"\"\n        status = {}\n        now = datetime.now()\n        \n        for operation_type, limit_config in self.limits.items():\n            if operation_type not in self.calls:\n                current_count = 0\n            else:\n                # Count recent calls\n                cutoff = now - timedelta(seconds=limit_config['window'])\n                current_count = len([\n                    call_time for call_time in self.calls[operation_type]\n                    if call_time > cutoff\n                ])\n                \n            status[operation_type] = {\n                'current_count': current_count,\n                'max_count': limit_config['count'],\n                'window_seconds': limit_config['window'],\n                'remaining': limit_config['count'] - current_count\n            }\n            \n        return status\n\n\nclass InputValidator:\n    \"\"\"Input validation utilities\"\"\"\n    \n    @staticmethod\n    def validate_token_address(address: str) -> bool:\n        \"\"\"Validate token mint address\"\"\"\n        return WalletSecurity.validate_wallet_address(address)\n    \n    @staticmethod\n    def validate_amount(amount: Any, min_amount: float = 0.0001, max_amount: float = 1000000) -> bool:\n        \"\"\"Validate trading amount\"\"\"\n        try:\n            if not isinstance(amount, (int, float)):\n                return False\n                \n            amount = float(amount)\n            return min_amount <= amount <= max_amount\n            \n        except (ValueError, TypeError):\n            return False\n    \n    @staticmethod\n    def validate_slippage(slippage: Any) -> bool:\n        \"\"\"Validate slippage percentage\"\"\"\n        try:\n            if not isinstance(slippage, (int, float)):\n                return False\n                \n            slippage = float(slippage)\n            return 0.001 <= slippage <= 0.5  # 0.1% to 50%\n            \n        except (ValueError, TypeError):\n            return False\n    \n    @staticmethod\n    def validate_priority_fee(fee: Any) -> bool:\n        \"\"\"Validate priority fee in microlamports\"\"\"\n        try:\n            if fee is None:\n                return True  # Optional parameter\n                \n            if not isinstance(fee, (int, float)):\n                return False\n                \n            fee = int(fee)\n            return 0 <= fee <= 1000000  # 0 to 1M microlamports\n            \n        except (ValueError, TypeError):\n            return False\n    \n    @staticmethod\n    def sanitize_string(input_str: str, max_length: int = 100) -> str:\n        \"\"\"Sanitize string input\"\"\"\n        if not isinstance(input_str, str):\n            return \"\"\n            \n        # Remove potentially dangerous characters\n        sanitized = re.sub(r'[^a-zA-Z0-9\\-_.,\\s]', '', input_str)\n        \n        # Limit length\n        return sanitized[:max_length]\n    \n    @staticmethod\n    def validate_network_config(config: Dict[str, Any]) -> bool:\n        \"\"\"Validate network configuration\"\"\"\n        try:\n            required_fields = ['rpc_url']\n            for field in required_fields:\n                if field not in config:\n                    return False\n                    \n            # Validate RPC URL\n            rpc_url = config.get('rpc_url')\n            if not isinstance(rpc_url, str) or not rpc_url.startswith(('http://', 'https://')):\n                return False\n                \n            return True\n            \n        except Exception:\n            return False\n\n\n# Global instances\n_rate_limiter = RateLimiter()\n\ndef get_rate_limiter() -> RateLimiter:\n    \"\"\"Get global rate limiter instance\"\"\"\n    return _rate_limiter