#!/usr/bin/env python3
"""
Debug script to isolate the exact source of the 'str' object has no attribute 'get' error
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_config_step_by_step():
    """Test each step of config loading to find where it breaks"""
    
    print("🔍 Step 1: Testing basic imports...")
    try:
        from utils.config_loader import load_config
        print("✅ config_loader imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config_loader: {e}")
        return False
    
    print("\n🔍 Step 2: Testing config loading...")
    try:
        config = load_config('configs/main_config.yaml')
        print(f"✅ Config loaded successfully - type: {type(config)}")
        print(f"✅ Config has get method: {hasattr(config, 'get')}")
        if hasattr(config, 'keys'):
            print(f"✅ Config keys: {list(config.keys())}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🔍 Step 3: Testing basic config access...")
    try:
        # Test the specific access patterns that are failing
        paths = config.get('paths', {})
        print(f"✅ Paths access: {type(paths)} - {paths}")
        
        data_dir = paths.get('data_dir', 'fallback') if hasattr(paths, 'get') else 'fallback'
        print(f"✅ Data dir: {data_dir}")
        
    except Exception as e:
        print(f"❌ Failed config access: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🔍 Step 4: Testing HybridVideoAI import...")
    try:
        from core.hybrid_ai import HybridVideoAI
        print("✅ HybridVideoAI imported successfully")
    except Exception as e:
        print(f"❌ Failed to import HybridVideoAI: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🔍 Step 5: Testing HybridVideoAI initialization...")
    try:
        print("Initializing HybridVideoAI...")
        model = HybridVideoAI(config)
        print("✅ HybridVideoAI initialized successfully")
    except Exception as e:
        print(f"❌ HybridVideoAI initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting step-by-step debug...")
    success = test_config_step_by_step()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed - error identified")