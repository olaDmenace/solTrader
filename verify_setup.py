#!/usr/bin/env python3
"""
Setup Verification Script
Checks if all dependencies and configurations are properly set up
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.9+")
        return False

def check_required_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'aiohttp',
        'anchorpy', 
        'base58',
        'dotenv',
        'solana',
        'solders',
        'pytest',
        'numpy',
        'pandas',
        'scipy',
        'asyncio',
        'telegram'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                from dotenv import load_dotenv
            elif package == 'telegram':
                import telegram
            elif package == 'asyncio':
                import asyncio
            else:
                __import__(package)
            print(f"   ✅ {package} - OK")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("   ✅ All packages installed")
        return True

def check_env_file():
    """Check if .env file exists and has required settings"""
    print("\n⚙️ Checking environment configuration...")
    
    env_path = Path('.env')
    if not env_path.exists():
        print("   ❌ .env file not found")
        print("   Create .env file following SETUP_GUIDE.md")
        return False
    
    print("   ✅ .env file exists")
    
    # Check for required variables
    required_vars = ['ALCHEMY_RPC_URL', 'WALLET_ADDRESS']
    
    try:
        with open('.env', 'r') as f:
            content = f.read()
            
        missing_vars = []
        for var in required_vars:
            if f"{var}=" not in content or f"{var}=" in content and content.split(f"{var}=")[1].split('\n')[0].strip() == "":
                missing_vars.append(var)
                print(f"   ❌ {var} - MISSING or EMPTY")
            else:
                print(f"   ✅ {var} - SET")
        
        if missing_vars:
            print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
            return False
        else:
            print("   ✅ Required environment variables set")
            return True
            
    except Exception as e:
        print(f"   ❌ Error reading .env file: {e}")
        return False

def check_project_structure():
    """Check if project files are in place"""
    print("\n📁 Checking project structure...")
    
    required_files = [
        'main.py',
        'src/config/settings.py',
        'src/trading/strategy.py',
        'src/trading/position.py',
        'src/token_scanner.py',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path} - OK")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("   ✅ All required files present")
        return True

def check_import_functionality():
    """Test if main imports work"""
    print("\n🔍 Testing import functionality...")
    
    try:
        # Test basic imports
        from src.config.settings import Settings, load_settings
        print("   ✅ Settings import - OK")
        
        from src.trading.position import Position, ExitReason
        print("   ✅ Position import - OK")
        
        from src.token_scanner import TokenScanner
        print("   ✅ TokenScanner import - OK")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Check that all files are properly structured")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def run_basic_functionality_test():
    """Run basic functionality test"""
    print("\n🧪 Running basic functionality test...")
    
    try:
        # Test position creation
        from src.trading.position import Position
        
        position = Position(
            token_address="TEST123",
            size=1.0,
            entry_price=1.0
        )
        
        # Test price update
        position.update_price(1.1)
        
        # Test PnL calculation
        expected_pnl = 0.1  # (1.1 - 1.0) * 1.0
        actual_pnl = position.unrealized_pnl
        
        if abs(actual_pnl - expected_pnl) < 0.001:
            print("   ✅ Position management - OK")
        else:
            print(f"   ❌ Position management - PnL calculation error")
            return False
            
        # Test momentum calculation
        for price in [1.0, 1.05, 1.1, 1.15, 1.2]:
            position.update_price(price)
            
        momentum = position._calculate_momentum()
        if isinstance(momentum, float):
            print("   ✅ Momentum calculation - OK")
        else:
            print("   ❌ Momentum calculation - Error")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Functionality test failed: {e}")
        return False

def generate_report(results):
    """Generate final setup report"""
    print("\n" + "="*50)
    print("🦍 SOLTRADER APE SETUP REPORT")
    print("="*50)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print("="*50)
    
    if all_passed:
        print("🎉 SETUP COMPLETE! Your APE bot is ready to run!")
        print("\nNext steps:")
        print("1. Run: python main.py (for paper trading)")
        print("2. Monitor performance for 1+ weeks")
        print("3. Switch to live trading when profitable")
        print("\n📖 Read SETUP_GUIDE.md for detailed instructions")
    else:
        print("❌ SETUP INCOMPLETE - Fix the failed checks above")
        print("\n🛠️ Common solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Create .env file with required settings")
        print("- Check SETUP_GUIDE.md for detailed instructions")
    
    return all_passed

def main():
    """Main setup verification"""
    print("🚀 SolTrader APE Bot - Setup Verification")
    print("="*50)
    
    results = {
        "Python Version": check_python_version(),
        "Required Packages": check_required_packages(),
        "Environment Config": check_env_file(),
        "Project Structure": check_project_structure(),
        "Import Functionality": check_import_functionality(),
        "Basic Functionality": run_basic_functionality_test()
    }
    
    success = generate_report(results)
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)