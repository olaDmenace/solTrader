import asyncio
from main import TradingBot

async def test():
    bot = TradingBot()
    try:
        # Test connections
        assert await bot.test_connections(), "Connection test failed"

        # Test wallet connection
        assert await bot.connect_wallet(), "Wallet connection failed"

        # Test telegram bot setup
        assert await bot.setup_telegram_bot(), "Telegram bot setup failed"

        # Test strategy initialization
        assert bot.strategy is not None, "Strategy not initialized"

        print("All tests passed!")

    except Exception as e:
        print(f"Test failed: {str(e)}")
    finally:
        await bot.shutdown()
        # Add extra cleanup
        await asyncio.sleep(0.1)  # Give time for connections to close

if __name__ == "__main__":
    asyncio.run(test())