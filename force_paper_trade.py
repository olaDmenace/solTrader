#!/usr/bin/env python3
"""
Force Paper Trade Test - Execute ONE paper trade immediately
"""
import asyncio
import sys
import os
sys.path.append('.')

from config.settings import load_settings
from src.trading.strategy import TradingStrategy, TradingMode
from core.wallet_manager import WalletManager
from api.alchemy import AlchemyClient
from api.enhanced_jupiter import enhanced_jupiter_client

async def force_paper_trade():
    """Force execute one paper trade to test the system"""
    print("🚀 FORCING PAPER TRADE EXECUTION...")
    
    try:
        # Load settings
        settings = load_settings()
        print(f"✅ Settings loaded - Paper Trading: {settings.PAPER_TRADING}")
        print(f"✅ Paper Balance: {settings.INITIAL_PAPER_BALANCE} SOL")
        
        # Initialize components
        alchemy = AlchemyClient(settings.ALCHEMY_RPC_URL)
        wallet = WalletManager(alchemy)
        wallet.wallet_address = settings.WALLET_ADDRESS
        
        # Initialize trading strategy
        strategy = TradingStrategy(
            jupiter_client=enhanced_jupiter_client,
            wallet=wallet,
            settings=settings,
            scanner=None,
            mode=TradingMode.PAPER,
            trade_logger=None
        )
        
        print(f"✅ Strategy initialized in PAPER mode")
        print(f"✅ Paper balance: {strategy.state.paper_balance} SOL")
        
        # Create a fake token that should definitely trade
        fake_token = {
            'address': 'CEqFiJoB6HLncfeieBw3PKKm9wyqrpSHV3VH4qZApump',  # Use CHARLIE from logs
            'symbol': 'CHARLIE',
            'name': 'Charlie Token',
            'price': 0.00244120,  # Price from logs
            'price_change_24h': 15.5,  # Good momentum
            'volume_24h': 8298537.62,  # High volume from logs
            'market_cap': 1000000,  # $1M - within our new $1B limit
            'liquidity': 300029.02,  # High liquidity from logs
            'age_minutes': 120,  # 2 hours old
            'momentum_score': 85.0,  # High momentum
            'source': 'force_test',
            'score': 95.0,
            'reasons': ['high_momentum', 'good_liquidity'],
            'bypassed_filters': []
        }
        
        print("🎯 Testing with forced token:")
        print(f"  Token: {fake_token['symbol']}")
        print(f"  Price: ${fake_token['price']:.8f} SOL")
        print(f"  Market Cap: ${fake_token['market_cap']:,}")
        print(f"  Volume: {fake_token['volume_24h']:,.0f} SOL")
        print(f"  Momentum: {fake_token['momentum_score']}")
        
        # Force process this token
        print("🔥 FORCING TRADE EXECUTION...")
        
        # Manually call the analyze method that should trigger a trade
        result = await strategy._analyze_and_trade_token(fake_token)
        
        if result:
            print("🎉 SUCCESS! Paper trade executed!")
            print(f"📊 New paper balance: {strategy.state.paper_balance} SOL")
            print(f"📈 Active positions: {len(strategy.state.paper_positions)}")
            
            # Show position details
            for addr, pos in strategy.state.paper_positions.items():
                print(f"  Position {addr[:8]}...: {pos.size} tokens at ${pos.entry_price:.8f}")
        else:
            print("❌ FAILED to execute trade")
            print("🔍 Check logs for failure reason")
            
        return result
        
    except Exception as e:
        print(f"❌ ERROR in force trade test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(force_paper_trade())
    print(f"\n{'✅ SUCCESS' if result else '❌ FAILED'}: Force paper trade test")