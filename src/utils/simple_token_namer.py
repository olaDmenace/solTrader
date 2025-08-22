#!/usr/bin/env python3
"""
Simple Token Name Generator
Quick, reliable token naming for dashboard without API dependencies
"""
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional

class SimpleTokenNamer:
    """Simple token name generator without external API dependencies"""
    
    def __init__(self):
        # Load any manual mappings from recent logs
        self.manual_mappings = {
            # From the trading logs - most profitable tokens
            "12Z5CsL5kCPRmnLQcwpY2QrEub3MahHCtRHxoQEdpump": {
                "name": "Hot Pump Token (Multi-trader)", 
                "symbol": "$12Z5", 
                "source": "manual"
            },
            "PjKEvRv7aYxhf6hmezsxEEa3eVS9CwPfffULY4Rpump": {
                "name": "Pump Token PjKE",
                "symbol": "$PJKE",
                "source": "manual" 
            },
            "CA2wm5cbcMjawbf5Lai7cLywpBSWyV94LbgAtAmDpump": {
                "name": "CA2w Pump (+462%)",
                "symbol": "$CA2W",
                "source": "manual"
            },
            "HL7xpwxwXSjvaHmYZUDDeGFb1p7tRZESZvXb4Lb9dQwg": {
                "name": "HL7x Token (+493%)",
                "symbol": "$HL7X",
                "source": "manual"
            },
            # Test tokens
            "TEST_EXECUTION_12345": {
                "name": "Test Token",
                "symbol": "TEST",
                "source": "test"
            },
            "TEST123456789abcdef": {
                "name": "Test Token",
                "symbol": "TEST",
                "source": "test"
            }
        }
        
        # Common token patterns
        self.patterns = [
            (r".*pump$", "Pump Token", "PUMP"),
            (r"^[A-Za-z0-9]{44}$", "Solana Token", "SOL"),
            (r"TEST.*", "Test Token", "TEST"),
        ]
    
    def get_token_info(self, token_address: str) -> Dict[str, str]:
        """Get token info quickly without external API calls"""
        
        # Check manual mappings first
        if token_address in self.manual_mappings:
            return self.manual_mappings[token_address]
        
        # Generate based on patterns
        for pattern, name_base, symbol_base in self.patterns:
            if re.match(pattern, token_address, re.IGNORECASE):
                return self._generate_from_pattern(token_address, name_base, symbol_base)
        
        # Default generation
        return self._generate_default_name(token_address)
    
    def _generate_from_pattern(self, token_address: str, name_base: str, symbol_base: str) -> Dict[str, str]:
        """Generate name based on detected pattern"""
        
        if token_address.endswith("pump"):
            # Pump.fun token - use first 6 chars as identifier
            identifier = token_address[:6].upper()
            return {
                "name": f"{identifier} (Pump.fun)",
                "symbol": f"${identifier}",
                "source": "pump_pattern"
            }
        
        if token_address.startswith("TEST"):
            return {
                "name": "Test Token",
                "symbol": "TEST",
                "source": "test_pattern"
            }
        
        # Generic Solana token
        return self._generate_default_name(token_address)
    
    def _generate_default_name(self, token_address: str) -> Dict[str, str]:
        """Generate default name for unknown tokens"""
        
        # Use first 6 characters as symbol
        symbol = token_address[:6].upper()
        
        # Create deterministic but unique name based on hash
        hash_obj = hashlib.md5(token_address.encode())
        hash_hex = hash_obj.hexdigest()[:4]
        
        return {
            "name": f"Token {symbol}...{hash_hex}",
            "symbol": symbol,
            "source": "generated"
        }
    
    def add_manual_mapping(self, token_address: str, name: str, symbol: str):
        """Add manual mapping for known tokens"""
        self.manual_mappings[token_address] = {
            "name": name,
            "symbol": symbol,
            "source": "manual"
        }
    
    def get_batch_token_info(self, token_addresses: list) -> Dict[str, Dict[str, str]]:
        """Get info for multiple tokens quickly"""
        return {addr: self.get_token_info(addr) for addr in token_addresses}

# Global instance
simple_namer = SimpleTokenNamer()

def get_token_display_info(token_address: str) -> Dict[str, str]:
    """Quick function to get token display info"""
    return simple_namer.get_token_info(token_address)