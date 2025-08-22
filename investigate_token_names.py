#!/usr/bin/env python3
"""
Investigate Token Names and Trading Diversity
Check what tokens are actually being traded and get their real names
"""
import asyncio
import logging
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config.settings import load_settings
from src.api.jupiter import JupiterClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def investigate_token_trading():
    """Investigate what tokens are actually being traded"""
    logger.info("🔍 INVESTIGATING TOKEN TRADING DIVERSITY")
    logger.info("=" * 60)
    
    # Read dashboard data
    try:
        with open("dashboard_data.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Could not read dashboard data: {e}")
        return
    
    # Initialize Jupiter client for token info
    jupiter = JupiterClient()
    
    # Extract unique token addresses from trades
    trades = data.get("trades", [])
    unique_tokens = set()
    
    logger.info(f"📊 ANALYZING {len(trades)} TOTAL TRADES")
    
    for trade in trades:
        token_addr = trade.get("token_address", "")
        if token_addr and token_addr not in ["TEST_EXECUTION_12345", "TEST123456789abcdef"]:
            unique_tokens.add(token_addr)
    
    logger.info(f"🎯 FOUND {len(unique_tokens)} UNIQUE TOKEN ADDRESSES (excluding test tokens)")
    logger.info("-" * 60)
    
    # Get token info for each unique token
    token_info = {}
    for i, token_addr in enumerate(unique_tokens, 1):
        logger.info(f"[{i}/{len(unique_tokens)}] Fetching info for {token_addr[:12]}...")
        
        try:
            info = await jupiter.get_token_info(token_addr)
            if info:
                token_info[token_addr] = {
                    "name": info.get("name", "Unknown"),
                    "symbol": info.get("symbol", "???"),
                    "address": token_addr
                }
                logger.info(f"  ✅ {info.get('symbol', '???')} - {info.get('name', 'Unknown')}")
            else:
                # If Jupiter doesn't have it, it's likely a pump.fun token
                token_info[token_addr] = {
                    "name": f"Pump.fun Token {token_addr[:8]}...",
                    "symbol": token_addr[:8],
                    "address": token_addr
                }
                logger.info(f"  ⚠️ Not in Jupiter token list - likely Pump.fun token")
        except Exception as e:
            logger.error(f"  ❌ Error fetching token info: {e}")
    
    # Analyze trade frequency per token
    logger.info("\n📈 TRADE FREQUENCY ANALYSIS")
    logger.info("-" * 60)
    
    token_trade_count = {}
    for trade in trades:
        token_addr = trade.get("token_address", "")
        if token_addr in unique_tokens:
            token_trade_count[token_addr] = token_trade_count.get(token_addr, 0) + 1
    
    # Sort by trade frequency
    sorted_tokens = sorted(token_trade_count.items(), key=lambda x: x[1], reverse=True)
    
    for token_addr, count in sorted_tokens:
        info = token_info.get(token_addr, {})
        name = info.get("name", "Unknown")
        symbol = info.get("symbol", "???")
        percentage = (count / len([t for t in trades if t.get("token_address") in unique_tokens])) * 100
        
        logger.info(f"  {symbol:>8} ({name[:20]:<20}) - {count:2} trades ({percentage:5.1f}%)")
    
    # Recent activity analysis
    logger.info("\n📊 RECENT TRADING ACTIVITY")
    logger.info("-" * 60)
    
    recent_trades = [t for t in trades if t.get("status") == "closed"][-10:]
    
    for trade in recent_trades:
        token_addr = trade.get("token_address", "")
        info = token_info.get(token_addr, {})
        symbol = info.get("symbol", "???")
        pnl = trade.get("pnl_percentage", 0)
        timestamp = trade.get("timestamp", "")[:19]
        
        logger.info(f"  {symbol:>8} - {pnl:>6.1f}% profit - {timestamp}")
    
    # Cleanup
    await jupiter.close()
    
    # Summary
    logger.info(f"\n🎯 SUMMARY:")
    logger.info(f"  • Total Trades: {len(trades)}")
    logger.info(f"  • Unique Tokens: {len(unique_tokens)}")
    logger.info(f"  • Most Traded: {sorted_tokens[0][0][:8]}... ({sorted_tokens[0][1]} trades)")
    logger.info(f"  • Token Diversity: {'HIGH' if len(unique_tokens) > 5 else 'MODERATE' if len(unique_tokens) > 2 else 'LOW'}")
    
    return token_info

if __name__ == "__main__":
    asyncio.run(investigate_token_trading())