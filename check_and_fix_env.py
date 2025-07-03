#!/usr/bin/env python3
"""
Environment Configuration Checker and Fixer
This script checks your .env configuration and helps fix common issues
"""
import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and is configured"""
    env_path = Path('.env')
    
    print("🔍 Checking .env configuration...")
    
    if not env_path.exists():
        print("❌ .env file not found!")
        create_sample_env()
        return False
    
    # Read current .env
    env_vars = {}
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                env_vars[key] = value
    
    # Check required variables
    required_vars = {
        'ALCHEMY_RPC_URL': 'Your Alchemy Solana RPC URL',
        'WALLET_ADDRESS': 'Your Solana wallet address',
        'PAPER_TRADING': 'Should be "true" for testing',
        'INITIAL_PAPER_BALANCE': 'Starting balance for paper trading'
    }
    
    missing_vars = []
    placeholder_vars = []
    
    for var, description in required_vars.items():
        if var not in env_vars:
            missing_vars.append(f"{var} - {description}")
        elif env_vars[var] in ['', 'YOUR_API_KEY_HERE', 'YOUR_SOLANA_WALLET_ADDRESS_HERE']:
            placeholder_vars.append(f"{var} - {description}")
    
    if missing_vars:
        print("❌ Missing required variables:")
        for var in missing_vars:
            print(f"   • {var}")
    
    if placeholder_vars:
        print("⚠️  Variables need configuration:")
        for var in placeholder_vars:
            print(f"   • {var}")
    
    if not missing_vars and not placeholder_vars:
        print("✅ .env configuration looks good!")
        return True
    
    print("\n🔧 Configuration help:")
    print("1. Get Alchemy API key from: https://alchemy.com")
    print("2. Create Solana app, select Mainnet")
    print("3. Copy the HTTPS RPC URL")
    print("4. Use your existing Solana wallet address")
    
    return False

def create_sample_env():
    """Create a sample .env file"""
    print("📝 Creating sample .env file...")
    
    env_content = """# ===================
# REQUIRED SETTINGS
# ===================

# Alchemy RPC URL (get from https://alchemy.com)
ALCHEMY_RPC_URL=https://solana-mainnet.g.alchemy.com/v2/YOUR_API_KEY_HERE

# Your Solana wallet address  
WALLET_ADDRESS=YOUR_SOLANA_WALLET_ADDRESS_HERE

# Private key (BASE58 encoded) - KEEP SECRET!
PRIVATE_KEY=YOUR_PRIVATE_KEY_BASE58_ENCODED

# ===================
# TRADING SETTINGS
# ===================

# Trading mode
TRADING_MODE=paper
PAPER_TRADING=true
INITIAL_PAPER_BALANCE=2.5
LIVE_TRADING_ENABLED=false

# APE Strategy Parameters
MOMENTUM_EXIT_ENABLED=true
MIN_CONTRACT_SCORE=70
MAX_POSITION_PER_TOKEN=0.5
MAX_SIMULTANEOUS_POSITIONS=3
MAX_HOLD_TIME_MINUTES=180
SCANNER_INTERVAL=5
MIN_LIQUIDITY=500.0

# Risk Management
MAX_DAILY_LOSS=1.0
STOP_LOSS_PERCENTAGE=15

# Network Configuration
SOLANA_NETWORK=devnet
JUPITER_API_URL=https://quote-api.jup.ag

# Optional: Notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Sample .env file created!")
    print("⚠️  Please edit .env with your actual API keys")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🐍 Testing Python imports...")
    
    required_modules = [
        'aiohttp',
        'solana', 
        'base58',
        'dotenv',
        'numpy',
        'pandas'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required modules available!")
    return True

def test_bot_components():
    """Test if bot components can be loaded"""
    print("\n🤖 Testing bot components...")
    
    try:
        sys.path.append('src')
        from src.config.settings import load_settings
        print("   ✅ Settings module")
        
        settings = load_settings()
        print("   ✅ Settings loaded")
        print(f"   📊 Paper trading: {settings.PAPER_TRADING}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Component test failed: {e}")
        return False

def main():
    """Main function"""
    print("🦍 SolTrader Environment Checker")
    print("=" * 40)
    
    # Check working directory
    if not Path('main.py').exists():
        print("❌ Please run this from the solTrader directory")
        print("   cd /home/trader/solTrader")
        sys.exit(1)
    
    # Check virtual environment
    if not Path('venv').exists():
        print("❌ Virtual environment not found!")
        print("   Create with: python3 -m venv venv")
        sys.exit(1)
    
    print("✅ In correct directory")
    
    # Run checks
    env_ok = check_env_file()
    imports_ok = test_imports()
    
    if env_ok and imports_ok:
        components_ok = test_bot_components()
        
        if components_ok:
            print("\n🎉 All checks passed!")
            print("✅ Ready to start the bot")
            print("\n📋 Next steps:")
            print("1. Run the fix script: chmod +x fix_soltrader_deployment.sh && ./fix_soltrader_deployment.sh") 
            print("2. Check logs: tail -f logs/trading.log")
            print("3. Monitor status: sudo systemctl status soltrader-bot")
        else:
            print("\n⚠️  Component test failed")
            print("Check your .env file configuration")
    else:
        print("\n❌ Setup incomplete")
        if not env_ok:
            print("• Configure your .env file")
        if not imports_ok:
            print("• Install missing dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
