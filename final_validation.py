import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.config.settings import load_settings
from core.token_scanner import EnhancedTokenScanner

async def validation_test():
    print('=== SOLTRADER VALIDATION TEST ===')
    
    # Test 1: Settings
    print('Test 1: Settings Loading...')
    settings = load_settings()
    print(f'  Paper Trading: {settings.PAPER_TRADING}')
    print(f'  Initial Balance: {settings.INITIAL_PAPER_BALANCE} SOL')
    print('  PASSED')
    
    # Test 2: Scanner initialization
    print('Test 2: Scanner Initialization...')
    scanner = EnhancedTokenScanner(settings)
    print('  PASSED')
    
    # Test 3: API connectivity
    print('Test 3: API Connectivity Test...')
    try:
        # Quick scan to test APIs
        tokens = await scanner.scan_for_new_tokens()
        print(f'  Found {len(tokens)} tokens')
        if tokens:
            token = tokens[0]
            print(f'  Sample: {token.get("symbol", "N/A")} - ${token.get("price", 0):.6f}')
        print('  PASSED')
    except Exception as e:
        print(f'  FAILED: {e}')
        return False
    
    print('=== ALL VALIDATION TESTS PASSED ===')
    return True

if __name__ == "__main__":
    result = asyncio.run(validation_test())