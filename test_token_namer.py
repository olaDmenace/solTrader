#!/usr/bin/env python3
"""
Test Token Naming System
"""
import sys
from pathlib import Path

# Add project root to path  
sys.path.append(str(Path(__file__).parent))
from src.utils.simple_token_namer import SimpleTokenNamer

def test_token_naming():
    """Test the token naming system"""
    namer = SimpleTokenNamer()
    
    # Test tokens from actual trades
    test_tokens = [
        "12Z5CsL5kCPRmnLQcwpY2QrEub3MahHCtRHxoQEdpump",
        "CA2wm5cbcMjawbf5Lai7cLywpBSWyV94LbgAtAmDpump", 
        "HL7xpwxwXSjvaHmYZUDDeGFb1p7tRZESZvXb4Lb9dQwg",
        "PjKEvRv7aYxhf6hmezsxEEa3eVS9CwPfffULY4Rpump"
    ]
    
    print("Testing Token Naming System")
    print("=" * 50)
    
    for token_addr in test_tokens:
        info = namer.get_token_info(token_addr)
        print(f"Token: {token_addr[:15]}...")
        print(f"  Name: {info['name']}")
        print(f"  Symbol: {info['symbol']}")
        print(f"  Source: {info['source']}")
        print()
    
    # Test batch functionality
    print("Testing Batch Functionality")
    print("=" * 50)
    
    batch_info = namer.get_batch_token_info(test_tokens)
    
    for addr, info in batch_info.items():
        short_addr = addr[:15] + "..."
        print(f"{short_addr:<18} -> {info['symbol']:<6} | {info['name'][:30]}")

if __name__ == "__main__":
    test_token_naming()