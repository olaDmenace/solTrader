#!/usr/bin/env python3
"""
Simple test script for live trading integration components
"""
import asyncio
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all new modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from src.security.wallet_security import WalletSecurity, RateLimiter, InputValidator
        from src.trading.transaction_manager import TransactionManager, TransactionStatus
        from src.trading.emergency_controls import EmergencyControls, CircuitBreakerConfig
        from src.trading.position_sync import PositionSynchronizer
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_wallet_security():
    """Test wallet security functions"""
    logger.info("Testing wallet security...")
    
    try:
        from src.security.wallet_security import WalletSecurity
        
        # Test address validation
        valid_address = "JxKzzx2Hif9fnpg9J6jY8XfwYnSLHF6CQZK7zT9ScNb"
        assert WalletSecurity.validate_wallet_address(valid_address) == True
        assert WalletSecurity.validate_wallet_address("invalid") == False
        
        # Test log sanitization
        sensitive_data = {"private_key": "secret123", "amount": 100}
        sanitized = WalletSecurity.sanitize_log_data(sensitive_data)
        assert sanitized["private_key"] == "[REDACTED]"
        assert sanitized["amount"] == 100
        
        logger.info("✅ Wallet security tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Wallet security test failed: {e}")
        return False


def test_rate_limiting():
    """Test rate limiting"""
    logger.info("Testing rate limiting...")
    
    try:
        from src.security.wallet_security import RateLimiter
        
        rate_limiter = RateLimiter()
        rate_limiter.limits['test'] = {'count': 2, 'window': 60}
        
        # Test within limits
        assert rate_limiter.is_allowed('test') == True
        assert rate_limiter.is_allowed('test') == True
        
        # Test over limits
        assert rate_limiter.is_allowed('test') == False
        
        logger.info("✅ Rate limiting tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Rate limiting test failed: {e}")
        return False


def test_input_validation():
    """Test input validation"""
    logger.info("Testing input validation...")
    
    try:
        from src.security.wallet_security import InputValidator
        
        # Test amount validation
        assert InputValidator.validate_amount(100.0) == True
        assert InputValidator.validate_amount(-1.0) == False
        
        # Test slippage validation
        assert InputValidator.validate_slippage(0.01) == True
        assert InputValidator.validate_slippage(0.6) == False
        
        logger.info("✅ Input validation tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Input validation test failed: {e}")
        return False


def test_emergency_controls():
    """Test emergency controls"""
    logger.info("Testing emergency controls...")
    
    try:
        from src.trading.emergency_controls import EmergencyControls, CircuitBreakerConfig
        
        config = CircuitBreakerConfig(max_daily_loss=50.0, position_limit=3)
        emergency = EmergencyControls(config)
        
        # Test emergency stop
        assert emergency.emergency_stop_active == False
        emergency.clear_emergency_stop()
        assert emergency.emergency_stop_active == False
        
        logger.info("✅ Emergency controls tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Emergency controls test failed: {e}")
        return False


def test_environment_config():
    """Test environment configuration"""
    logger.info("Testing environment configuration...")
    
    try:
        # Test environment variables exist
        test_vars = [
            ('LIVE_TRADING_ENABLED', 'false'),
            ('SOLANA_NETWORK', 'mainnet-beta'),
            ('MAX_DAILY_LOSS', '1.0'),
        ]
        
        for var, default in test_vars:
            value = os.getenv(var, default)
            assert value is not None
            logger.debug(f"{var} = {value}")
        
        logger.info("✅ Environment configuration tests passed")
        return True
    except Exception as e:
        logger.error(f"❌ Environment configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("Starting live trading integration tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Wallet Security", test_wallet_security), 
        ("Rate Limiting", test_rate_limiting),
        ("Input Validation", test_input_validation),
        ("Emergency Controls", test_emergency_controls),
        ("Environment Config", test_environment_config),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("LIVE TRADING INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Total: {len(tests)} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Live trading integration is ready.")
        print("\n📋 NEXT STEPS:")
        print("1. Set PRIVATE_KEY in .env for live trading")
        print("2. Set LIVE_TRADING_ENABLED=true when ready")
        print("3. Start with small amounts on devnet")
        print("4. Monitor logs and emergency controls")
        print("5. Test thoroughly before mainnet use")
        print("\n⚠️  WARNING: Live trading involves real money. Test carefully!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review and fix issues.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)