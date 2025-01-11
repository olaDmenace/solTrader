import asyncio
import logging
from dotenv import load_dotenv
import os
from src.api.alchemy import AlchemyClient
from src.api.jupiter import JupiterClient
from src.phantom_wallet import PhantomWallet
from src.token_scanner import TokenScanner
from src.trend_detector import TrendDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_application():
    logger.info("Starting integration test")
    load_dotenv()
    
    # Initialize API clients
    rpc_url = os.getenv('ALCHEMY_RPC_URL')
    wallet_address = os.getenv('WALLET_ADDRESS')
    
    if not rpc_url or not wallet_address:
        raise ValueError("Missing required environment variables")
    
    alchemy = AlchemyClient(rpc_url)
    jupiter = JupiterClient()
    wallet = PhantomWallet(alchemy)
    
    try:
        # Test API connections
        logger.info("Testing Alchemy connection...")
        alchemy_ok = await alchemy.test_connection()
        assert alchemy_ok, "Alchemy connection failed"
        
        logger.info("Testing Jupiter connection...")
        jupiter_ok = await jupiter.test_connection()
        assert jupiter_ok, "Jupiter connection failed"
        
        # Test wallet functionality
        logger.info("Testing wallet connection...")
        connected = await wallet.connect(wallet_address)
        assert connected, "Wallet connection failed"
        
        balance = await wallet.get_balance()
        assert balance is not None, "Failed to get wallet balance"
        logger.info(f"Wallet balance: {balance} SOL")
        
        # Test token scanning
        logger.info("Testing token scanner...")
        scanner = TokenScanner(jupiter, alchemy)
        new_tokens = await scanner.scan_new_listings()
        logger.info(f"Found {len(new_tokens)} new tokens")
        
        if new_tokens:
            # Test trend detection
            logger.info("Testing trend detection...")
            detector = TrendDetector(jupiter)
            token = new_tokens[0]
            trend = await detector.analyze_trend(token['address'])
            assert trend is not None, "Trend analysis failed"
            logger.info(f"Trend analysis: {trend}")
            
        logger.info("Integration test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        await wallet.disconnect()
        await alchemy.close()
        await jupiter.close()

if __name__ == "__main__":
    asyncio.run(test_application())