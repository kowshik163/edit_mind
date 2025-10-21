#!/usr/bin/env python3
"""
Debug script to isolate the config loading issue
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_config_loading():
    try:
        print("🔍 Testing config loading...")
        from utils.config_loader import load_config
        config = load_config('configs/main_config.yaml')
        print(f"✅ Config loaded successfully")
        print(f"Config type: {type(config)}")
        print(f"Config keys: {list(config.keys())}")
        
        # Test the specific access patterns that are failing
        print("\n🔍 Testing config access patterns...")
        
        # Test teachers access
        teachers = config.get('teachers', {})
        print(f"Teachers: {type(teachers)} - {teachers}")
        
        if isinstance(teachers, dict):
            text_model = teachers.get('text_model', 'fallback')
            print(f"Text model: {text_model}")
        else:
            print(f"❌ Teachers is not a dict: {teachers}")
        
        # Test model access
        model_config = config.get('model', {})
        print(f"Model config: {type(model_config)} - {model_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_ai_init():
    try:
        print("\n🔍 Testing HybridVideoAI initialization...")
        from utils.config_loader import load_config
        config = load_config('configs/main_config.yaml')
        
        # Try to import the class
        from core.hybrid_ai import HybridVideoAI
        print("✅ HybridVideoAI imported successfully")
        
        # Try to initialize (this might fail due to missing models, but we want to see where)
        try:
            model = HybridVideoAI(config)
            print("✅ HybridVideoAI initialized successfully")
        except Exception as init_error:
            print(f"❌ HybridVideoAI initialization failed: {init_error}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting debug session...")
    
    success1 = test_config_loading()
    success2 = test_hybrid_ai_init()
    
    if success1 and success2:
        print("\n✅ All tests passed")
    else:
        print("\n❌ Some tests failed")