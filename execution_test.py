#!/usr/bin/env python3
"""
Test actual trade execution pipeline
"""
import asyncio
import sys
import os
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path.cwd()))
os.environ['PYTHONPATH'] = str(Path.cwd())

from src.config.settings import load_settings
from src.enhanced_token_scanner import EnhancedTokenScanner

async def test_execution_pipeline():
    """Test complete execution pipeline"""
    print("=== EXECUTION PIPELINE TEST ===")
    
    settings = load_settings()
    print(f"Paper Trading: {settings.PAPER_TRADING}")
    print(f"Initial Balance: {settings.INITIAL_PAPER_BALANCE} SOL")
    
    # Test scanner
    print("\n1. Testing Token Scanner...")
    scanner = EnhancedTokenScanner(settings)
    
    try:
        # Get some tokens
        tokens = await scanner.scan_for_new_tokens()
        print(f"   Found {len(tokens)} tokens")
        
        if tokens:
            print(f"   Sample token: {tokens[0]}")
        
        # Test approved tokens
        approved = await scanner.get_approved_tokens()
        print(f"   Approved tokens: {len(approved)}")
        
        if approved:
            print(f"   First approved: {approved[0].get('symbol', 'N/A')}")
            return True
        else:
            print("   ERROR: No approved tokens!")
            return False
            
    except Exception as e:
        print(f"   ERROR: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_execution_pipeline())
    print(f"\nResult: {'PASS' if result else 'FAIL'}")