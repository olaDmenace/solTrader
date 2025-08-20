#!/usr/bin/env python3
"""
Simple validation test without Unicode characters to avoid encoding issues
"""
import asyncio
import os
import sys
import time
from dotenv import load_dotenv
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(override=True)

def print_status(message):
    """Print status message without Unicode"""
    print(f"[STATUS] {message}")

def print_success(message):
    """Print success message without Unicode"""
    print(f"[SUCCESS] {message}")

def print_error(message):
    """Print error message without Unicode"""
    print(f"[ERROR] {message}")

async def test_basic_imports():
    """Test basic imports"""
    print_status("Testing basic imports...")
    try:
        from src.config.settings import Settings, load_settings
        from src.api.alchemy import AlchemyClient
        from src.api.solana_tracker import SolanaTrackerClient
        from src.api.geckoterminal_client import GeckoTerminalClient
        from src.enhanced_token_scanner import EnhancedTokenScanner
        from src.trading.strategy import TradingStrategy, TradingMode
        print_success("All imports successful")
        return True
    except Exception as e:
        print_error(f"Import failed: {e}")
        return False

async def test_settings_loading():
    """Test settings loading"""
    print_status("Testing settings loading...")
    try:
        from src.config.settings import load_settings
        settings = load_settings()
        print_success(f"Settings loaded - Paper trading: {settings.PAPER_TRADING}")
        return True, settings
    except Exception as e:
        print_error(f"Settings loading failed: {e}")
        return False, None

async def test_api_connections():
    """Test API connections"""
    print_status("Testing API connections...")
    try:
        from src.api.solana_tracker import SolanaTrackerClient
        from src.api.geckoterminal_client import GeckoTerminalClient
        
        # Test Solana Tracker
        solana_client = SolanaTrackerClient()
        print_success("Solana Tracker client initialized")
        
        # Test GeckoTerminal
        gecko_client = GeckoTerminalClient()
        print_success("GeckoTerminal client initialized")
        
        return True
    except Exception as e:
        print_error(f"API connection test failed: {e}")
        return False

async def test_token_scanner():
    """Test token scanner"""
    print_status("Testing Enhanced Token Scanner...")
    try:
        from src.enhanced_token_scanner import EnhancedTokenScanner
        from src.config.settings import load_settings
        
        settings = load_settings()
        scanner = EnhancedTokenScanner(settings)
        print_success("Enhanced Token Scanner initialized")
        return True
    except Exception as e:
        print_error(f"Token scanner test failed: {e}")
        return False

async def test_paper_trading():
    """Test paper trading setup"""
    print_status("Testing paper trading setup...")
    try:
        from src.trading.strategy import TradingStrategy, TradingMode
        from src.config.settings import load_settings
        
        settings = load_settings()
        strategy = TradingStrategy(settings)
        
        # Check paper mode
        if settings.PAPER_TRADING:
            print_success(f"Paper trading enabled - Balance: {settings.INITIAL_PAPER_BALANCE} SOL")
        else:
            print_error("Paper trading not enabled - DANGEROUS!")
            return False
            
        return True
    except Exception as e:
        print_error(f"Paper trading test failed: {e}")
        return False

async def test_environment():
    """Test environment variables"""
    print_status("Testing environment variables...")
    
    required_vars = [
        'SOLANA_TRACKER_KEY',
        'ALCHEMY_RPC_URL',
        'WALLET_ADDRESS'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print_error(f"Missing environment variables: {missing_vars}")
        return False
    
    print_success("All required environment variables are set")
    return True

async def main():
    """Run all validation tests"""
    print("=" * 60)
    print("SOLTRADER VALIDATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment),
        ("Basic Imports", test_basic_imports),
        ("Settings Loading", test_settings_loading),
        ("API Connections", test_api_connections),
        ("Token Scanner", test_token_scanner),
        ("Paper Trading", test_paper_trading),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        try:
            result = await test_func()
            if isinstance(result, tuple):
                result = result[0]  # Extract boolean from tuple
            
            if result:
                passed += 1
                print_success(f"{test_name}: PASSED")
            else:
                failed += 1
                print_error(f"{test_name}: FAILED")
        except Exception as e:
            failed += 1
            print_error(f"{test_name}: FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Tests Passed: {passed}")
    print(f"Tests Failed: {failed}")
    print(f"Success Rate: {(passed/(passed+failed)*100):.1f}%" if (passed+failed) > 0 else "0%")
    
    if failed == 0:
        print_success("ALL TESTS PASSED - System is ready!")
        return 0
    else:
        print_error(f"SOME TESTS FAILED - {failed} issues need attention")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)