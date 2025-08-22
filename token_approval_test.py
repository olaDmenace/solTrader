#!/usr/bin/env python3
"""
Test token approval pipeline specifically
"""
import asyncio
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path.cwd()))

from src.config.settings import load_settings
from src.enhanced_token_scanner import EnhancedTokenScanner

async def test_approval_pipeline():
    """Test the approval pipeline specifically"""
    print("=== TOKEN APPROVAL PIPELINE TEST ===")
    
    settings = load_settings()
    scanner = EnhancedTokenScanner(settings)
    
    try:
        # Start scanner to initialize API connections
        await scanner.api_client.start_session()
        
        # Perform full scan
        print("Running full scan...")
        results = await scanner._perform_full_scan()
        
        if results:
            print(f"SUCCESS: {len(results)} tokens approved!")
            for i, result in enumerate(results[:3]):  # Show first 3
                print(f"  {i+1}. {result.token.symbol} - Score: {result.score:.1f} - Reasons: {result.reasons[:2]}")
        else:
            print("FAILED: No tokens approved")
        
        # Try get_approved_tokens method
        print("\nTesting get_approved_tokens...")
        approved = await scanner.get_approved_tokens()
        print(f"Available approved tokens: {len(approved)}")
        
        return len(results) > 0 if results else False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        await scanner.api_client.close()

if __name__ == "__main__":
    result = asyncio.run(test_approval_pipeline())
    print(f"\nApproval Test: {'PASS' if result else 'FAIL'}")