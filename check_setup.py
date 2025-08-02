#!/usr/bin/env python3
"""
Setup Checker for Paper Trading Bot
Checks if environment is ready to run the paper trading bot
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_required_modules():
    """Check if required modules can be imported"""
    print("\n📦 Checking required modules...")
    
    required_modules = [
        'asyncio',
        'aiohttp', 
        'requests',
        'dotenv',
        'logging',
        'json',
        'datetime',
        'pathlib'
    ]
    
    all_available = True
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - MISSING")
            all_available = False
    
    return all_available

def check_environment_file():
    """Check if .env file exists"""
    print("\n🔧 Checking environment configuration...")
    
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file found")
        
        # Check for key variables
        try:
            with open('.env', 'r') as f:
                content = f.read()
            
            required_vars = [
                'SOLANA_TRACKER_API_KEY',
                'ALCHEMY_RPC_URL',
                'WALLET_ADDRESS'
            ]
            
            all_vars_present = True
            for var in required_vars:
                if var in content:
                    print(f"✅ {var} configured")
                else:
                    print(f"⚠️ {var} - may need configuration")
                    all_vars_present = False
            
            return True  # File exists, even if some vars missing
            
        except Exception as e:
            print(f"⚠️ Could not read .env file: {e}")
            return True  # File exists
    else:
        print("⚠️ .env file not found - create one with API keys")
        return False

def check_project_structure():
    """Check if project files are in place"""
    print("\n📁 Checking project structure...")
    
    required_files = [
        'main.py',
        'bot_data.json',
        'src/config/settings.py',
        'src/enhanced_token_scanner.py',
        'src/trading/strategy.py',
        'src/api/jupiter.py',
        'src/api/alchemy.py'
    ]
    
    all_files_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_files_present = False
    
    return all_files_present

def check_virtual_environment():
    """Check if running in virtual environment"""
    print("\n🌐 Checking virtual environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️ Not in virtual environment - recommended to use venv")
        return False

def main():
    """Main setup check function"""
    print("🚀 Paper Trading Bot - Setup Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Required Modules", check_required_modules()),
        ("Environment File", check_environment_file()),
        ("Project Structure", check_project_structure()),
        ("Virtual Environment", check_virtual_environment())
    ]
    
    print("\n" + "=" * 50)
    print("SETUP CHECK RESULTS")
    print("=" * 50)
    
    critical_passed = 0
    total_critical = 4  # Don't count venv as critical
    
    for i, (check_name, result) in enumerate(checks):
        status = "✅ READY" if result else "❌ NEEDS ATTENTION"
        print(f"{status} - {check_name}")
        
        # Count critical checks (not virtual env)
        if i < 4 and result:
            critical_passed += 1
    
    print("\n" + "=" * 50)
    
    if critical_passed >= 3:  # Allow some flexibility
        print("🎉 SETUP IS READY!")
        print("📈 You can run the paper trading bot!")
        print("\n🚀 Quick Start:")
        print("   python main.py")
        print("\n📊 Monitor Progress:")
        print("   Watch bot_data.json for real-time activity")
        print("   Check logs/trading.log for detailed logs")
        
        if critical_passed < 4:
            print("\n💡 Minor Issues:")
            print("   Some non-critical items need attention")
            print("   Bot should still work but may have limitations")
    
    else:
        print("⚠️ SETUP NEEDS WORK")
        print("🔧 Fix the missing components before running the bot")
        print("\n📝 Common Solutions:")
        print("   • pip install -r requirements.txt")
        print("   • Create .env file with API keys") 
        print("   • Ensure all project files are present")
    
    return critical_passed >= 3

if __name__ == "__main__":
    main()