#!/usr/bin/env python3
"""
Quick test script to verify Solana Tracker API key is working
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import aiohttp

# Load environment variables
load_dotenv(override=True)

class SolanaTrackerTester:
    def __init__(self):
        self.api_key = os.getenv('SOLANA_TRACKER_KEY')
        self.base_url = "https://data.solanatracker.io"
        
    async def test_api_key(self):
        """Test if the API key works"""
        print("🔑 Testing Solana Tracker API Key...")
        print("=" * 50)
        
        # Check if key exists
        if not self.api_key:
            print("❌ No API key found in environment variables")
            print("   Make sure SOLANA_TRACKER_KEY is set in your .env file")
            return False
        
        # Show masked key
        masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        print(f"✅ API key found: {masked_key}")
        
        # Test API request
        headers = {
            'User-Agent': 'SolTrader-Test/1.0',
            'Accept': 'application/json',
            'x-api-key': self.api_key
        }
        
        test_url = f"{self.base_url}/tokens/trending"
        params = {'limit': 1}
        
        print(f"🌐 Testing API request to: {test_url}")
        print(f"📋 Headers: {dict(headers)}")  # Don't show the actual key
        headers_display = dict(headers)
        headers_display['x-api-key'] = masked_key
        print(f"📋 Headers: {headers_display}")
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(test_url, params=params, headers=headers) as response:
                    print(f"📊 Response Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print("🎉 SUCCESS! API key is working correctly")
                        # Handle both dict and list responses
                        if isinstance(data, dict):
                            token_count = len(data.get('data', []))
                        elif isinstance(data, list):
                            token_count = len(data)
                        else:
                            token_count = 0
                        print(f"📈 Retrieved {token_count} trending tokens")
                        print(f"📄 Response type: {type(data)}")
                        return True
                    elif response.status == 401:
                        print("❌ UNAUTHORIZED (401)")
                        print("   Possible issues:")
                        print("   1. Invalid API key")
                        print("   2. API key not activated")
                        print("   3. Need to subscribe to a plan")
                        print("   4. Check your account at https://solanatracker.io/")
                        return False
                    elif response.status == 429:
                        print("⚠️  RATE LIMITED (429)")
                        print("   You've hit the rate limit")
                        return False
                    else:
                        print(f"❌ UNEXPECTED ERROR ({response.status})")
                        error_text = await response.text()
                        print(f"   Response: {error_text}")
                        return False
                        
        except Exception as e:
            print(f"❌ REQUEST FAILED: {str(e)}")
            return False

async def main():
    """Main test function"""
    tester = SolanaTrackerTester()
    success = await tester.test_api_key()
    
    print("=" * 50)
    if success:
        print("✅ API KEY TEST PASSED - Your bot should work now!")
        print("   Run: python main.py")
    else:
        print("❌ API KEY TEST FAILED")
        print("   Steps to fix:")
        print("   1. Check your .env file has: SOLANA_TRACKER_KEY=your_key")
        print("   2. Verify your key at: https://solanatracker.io/")
        print("   3. Make sure you have an active subscription")

if __name__ == "__main__":
    asyncio.run(main())