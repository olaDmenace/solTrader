"""
Quick Time Sync Fix Application
Run this to apply basic fixes to your trading modules
"""

import os
import re

def apply_quick_fixes():
    """Apply quick time sync fixes to key files"""
    
    fixes = [
        ('datetime.utcnow()', 'trading_time.now()'),
        ('datetime.now()', 'trading_time.now()'),
    ]
    
    key_files = [
        'src/trading/risk_engine.py',
        'src/trading/paper_trading_engine.py',
        'src/portfolio/portfolio_manager.py',
        'src/database/db_manager.py',
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply fixes
                for old, new in fixes:
                    if old in content:
                        content = content.replace(old, new)
                        print(f"  Replaced {old} with {new}")
                
                # Add import if needed
                if 'trading_time.now()' in content and 'from utils.trading_time import trading_time' not in content:
                    # Add import at top
                    lines = content.split('\n')
                    import_line = 'from utils.trading_time import trading_time'
                    
                    # Find where to insert import
                    insert_index = 0
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            insert_index = i + 1
                    
                    lines.insert(insert_index, import_line)
                    content = '\n'.join(lines)
                    print(f"  Added trading_time import")
                
                # Write back if changed
                if content != original_content:
                    with open(file_path + '.backup', 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"  Updated {file_path} (backup created)")
                else:
                    print(f"  No changes needed for {file_path}")
                    
            except Exception as e:
                print(f"  Error processing {file_path}: {str(e)}")
        else:
            print(f"  File not found: {file_path}")

if __name__ == "__main__":
    print("APPLYING QUICK TIME SYNC FIXES...")
    apply_quick_fixes()
    print("FIXES APPLIED - Review changes and test")
