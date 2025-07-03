"""
Simplified Token Scanner Integration
Uses basic APIs only - no complex holder analysis or contract verification
"""
import logging
from src.simple_token_scanner import SimpleTokenScanner

logger = logging.getLogger(__name__)

# Make SimpleTokenScanner available as TokenScanner
TokenScanner = SimpleTokenScanner

# For backward compatibility, create the expected class
class TokenScannerCompat(SimpleTokenScanner):
    """Compatibility wrapper for the simplified scanner"""
    
    def __init__(self, jupiter_client, alchemy_client, settings):
        super().__init__(jupiter_client, alchemy_client, settings)
        logger.info("✅ Using simplified token scanner (no complex analysis)")
    
    async def scan_for_tokens(self):
        """Compatibility method for existing strategy code"""
        return await self._scan_tokens()
    
    async def analyze_token(self, token_address):
        """Compatibility method for token analysis"""
        return await self.get_token_metrics(token_address)

# Export the compatibility class as the main TokenScanner
TokenScanner = TokenScannerCompat
