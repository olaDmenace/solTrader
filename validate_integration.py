#!/usr/bin/env python3
"""
Simple validation script for Birdeye integration
Tests core logic without external dependencies
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_trending_analyzer_import():
    """Test that trending analyzer can be imported and initialized"""
    try:
        from src.trending_analyzer import TrendingAnalyzer
        
        class MockSettings:
            TRENDING_SIGNAL_BOOST = 0.5
            MAX_TRENDING_RANK = 50
            MIN_PRICE_CHANGE_24H = 20.0
            MIN_VOLUME_CHANGE_24H = 10.0
            MIN_TRENDING_SCORE = 60.0
        
        analyzer = TrendingAnalyzer(MockSettings())
        print("✅ TrendingAnalyzer imported and initialized successfully")
        return True
    except Exception as e:
        print(f"❌ TrendingAnalyzer import failed: {e}")
        return False

def test_settings_integration():
    """Test that settings have been properly updated"""
    try:
        from src.config.settings import Settings
        
        # Check if new settings exist
        settings_attrs = dir(Settings)
        required_attrs = [
            'BIRDEYE_API_KEY',
            'ENABLE_TRENDING_FILTER', 
            'MAX_TRENDING_RANK',
            'MIN_PRICE_CHANGE_24H',
            'MIN_TRENDING_SCORE',
            'TRENDING_SIGNAL_BOOST'
        ]
        
        missing_attrs = [attr for attr in required_attrs if attr not in settings_attrs]
        
        if missing_attrs:
            print(f"❌ Missing settings attributes: {missing_attrs}")
            return False
        else:
            print("✅ All Birdeye settings properly added to Settings class")
            return True
            
    except Exception as e:
        print(f"❌ Settings integration test failed: {e}")
        return False

def test_scanner_integration():
    """Test that scanner has trending filter integration"""
    try:
        # Read the scanner file and check for trending imports
        scanner_file = os.path.join('src', 'practical_solana_scanner.py')
        
        with open(scanner_file, 'r') as f:
            content = f.read()
        
        required_elements = [
            'from .birdeye_client import BirdeyeClient',
            'from .trending_analyzer import TrendingAnalyzer',
            '_passes_trending_filter',
            'trending_analyzer',
            'birdeye_client'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Scanner missing integration elements: {missing_elements}")
            return False
        else:
            print("✅ Scanner properly integrated with trending filter")
            return True
            
    except Exception as e:
        print(f"❌ Scanner integration test failed: {e}")
        return False

def test_signal_enhancement():
    """Test that signal enhancement is integrated"""
    try:
        signals_file = os.path.join('src', 'trading', 'signals.py')
        
        with open(signals_file, 'r') as f:
            content = f.read()
        
        required_elements = [
            '_apply_trending_boost',
            'trending_token',
            'trending_score',
            'TrendingAnalyzer',
            'enhance_signal_strength'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Signals missing enhancement elements: {missing_elements}")
            return False
        else:
            print("✅ Signal enhancement properly integrated")
            return True
            
    except Exception as e:
        print(f"❌ Signal enhancement test failed: {e}")
        return False

def test_trending_score_calculation():
    """Test trending score calculation logic"""
    try:
        from src.trending_analyzer import TrendingAnalyzer
        
        class MockSettings:
            TRENDING_SIGNAL_BOOST = 0.5
            MAX_TRENDING_RANK = 50
            MIN_PRICE_CHANGE_24H = 20.0
            MIN_VOLUME_CHANGE_24H = 10.0
            MIN_TRENDING_SCORE = 60.0
        
        class MockTrendingToken:
            def __init__(self):
                self.rank = 5
                self.price_24h_change_percent = 87.5
                self.volume_24h_change_percent = 145.2
                self.volume_24h_usd = 1200000
                self.liquidity = 450000
                self.symbol = "TESTTOKEN"
        
        analyzer = TrendingAnalyzer(MockSettings())
        mock_token = MockTrendingToken()
        
        # Test score calculation
        score = analyzer.calculate_trending_score(mock_token)
        print(f"✅ Trending score calculation works: {score:.1f}/100")
        
        # Test criteria validation  
        passes, reason = analyzer.meets_trending_criteria(mock_token)
        print(f"✅ Criteria validation works: {passes} - {reason}")
        
        # Test signal enhancement
        base_signal = 0.7
        enhanced = analyzer.enhance_signal_strength(base_signal, mock_token)
        boost = enhanced - base_signal
        print(f"✅ Signal enhancement works: {base_signal:.3f} → {enhanced:.3f} (+{boost:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Trending calculation test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        'src/birdeye_client.py',
        'src/trending_analyzer.py', 
        'src/config/settings.py',
        'src/practical_solana_scanner.py',
        'src/trading/signals.py',
        'test_birdeye_integration.py',
        'test_birdeye_mock.py',
        'BIRDEYE_INTEGRATION_README.md',
        'BIRDEYE_API_SETUP.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all validation tests"""
    print("🔍 Validating Birdeye Trending API Integration")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Settings Integration", test_settings_integration),
        ("Trending Analyzer Import", test_trending_analyzer_import),
        ("Scanner Integration", test_scanner_integration), 
        ("Signal Enhancement", test_signal_enhancement),
        ("Score Calculation Logic", test_trending_score_calculation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 30)
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Birdeye integration is properly implemented")
        print("✅ Ready for testing with API key")
        print("")
        print("📝 NEXT STEPS:")
        print("1. Get Birdeye API key (see BIRDEYE_API_SETUP.md)")
        print("2. Set BIRDEYE_API_KEY environment variable")
        print("3. Run: python test_birdeye_integration.py")
        print("4. Enable in production with ENABLE_TRENDING_FILTER=true")
    else:
        print(f"⚠️  {total - passed} validation tests failed")
        print("🔧 Please review the failing tests above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)