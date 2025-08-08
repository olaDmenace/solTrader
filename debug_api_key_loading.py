#!/usr/bin/env python3

"""
Debug API Key Loading - No Unicode
"""

import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_environment():
    """Debug environment variable loading"""
    
    print("=" * 50)
    print("API KEY LOADING DEBUG")
    print("=" * 50)
    
    # Load .env file explicitly
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    print(f"Loading .env from: {env_path}")
    print(f".env file exists: {os.path.exists(env_path)}")
    
    # Load environment
    load_dotenv(env_path)
    
    # Check API keys
    st_key = os.getenv('SOLANA_TRACKER_KEY')
    api_strategy = os.getenv('API_STRATEGY', 'not_set')
    
    print(f"\nEnvironment Variables:")
    print(f"SOLANA_TRACKER_KEY: {st_key[:20] if st_key else 'NOT FOUND'}...")
    print(f"API_STRATEGY: {api_strategy}")
    
    # Test direct reading from .env
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()
            
        st_lines = [line.strip() for line in lines if 'SOLANA_TRACKER_KEY=' in line and not line.strip().startswith('#')]
        
        print(f"\nDirect .env file reading:")
        print(f"Found SOLANA_TRACKER_KEY lines: {len(st_lines)}")
        for line in st_lines:
            key_part = line.split('=')[1] if '=' in line else 'invalid'
            print(f"  Line: {key_part[:20]}...")
    
    return st_key

def test_solana_tracker_client():
    """Test Solana Tracker client initialization"""
    
    try:
        from api.solana_tracker import SolanaTrackerClient
        
        print(f"\n" + "=" * 50)
        print("SOLANA TRACKER CLIENT TEST")
        print("=" * 50)
        
        # Test client initialization
        client = SolanaTrackerClient()
        
        print(f"Client API key set: {bool(client.api_key)}")
        if client.api_key:
            print(f"API key length: {len(client.api_key)}")
            print(f"API key preview: {client.api_key[:20]}...")
        else:
            print("ERROR: No API key loaded")
            
        return client.api_key is not None
        
    except Exception as e:
        print(f"Error testing client: {e}")
        return False

def main():
    """Main debug function"""
    
    # Debug environment loading
    api_key = debug_environment()
    
    # Test client
    client_ok = test_solana_tracker_client()
    
    print(f"\n" + "=" * 50)
    print("DEBUG SUMMARY")
    print("=" * 50)
    print(f"Environment API key loaded: {bool(api_key)}")
    print(f"Client API key loaded: {client_ok}")
    
    if api_key and client_ok:
        print("SUCCESS: API key loading works correctly")
        return True
    else:
        print("ISSUE: API key not loading properly")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)