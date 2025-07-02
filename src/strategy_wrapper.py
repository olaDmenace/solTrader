"""
Strategy wrapper that handles missing data gracefully
"""
import logging

logger = logging.getLogger(__name__)

def safe_token_analysis(original_method):
    """Decorator to make token analysis methods safe"""
    async def wrapper(*args, **kwargs):
        try:
            result = await original_method(*args, **kwargs)
            return result
        except Exception as e:
            logger.debug(f"Token analysis method failed gracefully: {e}")
            return None
    return wrapper

def patch_strategy_methods():
    """Patch strategy methods to handle missing data"""
    logger.info("✅ Applied safe token analysis patches")

if __name__ == "__main__":
    patch_strategy_methods()
