print("Starting test...")
try:
    import solana
    print("\nAll available attributes in solana module:")
    for item in dir(solana):
        if not item.startswith('__'):  # Skip internal attributes
            print(f"- {item}")
            
    print("\nTrying to import specific modules:")
    from solana import rpc
    print("Successfully imported rpc")
    
    from solana import transaction
    print("Successfully imported transaction")
    
    from solana.rpc import async_api
    print("Successfully imported async_api")
    
except Exception as e:
    print("Error:", str(e))
print("\nTest complete.")