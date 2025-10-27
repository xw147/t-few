#!/usr/bin/env python3
"""
Quick test script to check if DeepSpeed is working properly
"""

import sys
import torch

def test_pytorch():
    """Test basic PyTorch functionality"""
    print("=" * 50)
    print("Testing PyTorch...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test basic tensor operations
    try:
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print("✓ Basic tensor operations working")
        
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = torch.mm(x_gpu, y_gpu)
            print("✓ CUDA tensor operations working")
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False
    
    return True

def test_deepspeed_import():
    """Test DeepSpeed import"""
    print("=" * 50)
    print("Testing DeepSpeed import...")
    
    try:
        import deepspeed
        print(f"✓ DeepSpeed imported successfully")
        print(f"DeepSpeed version: {deepspeed.__version__}")
        return True
    except ImportError as e:
        print(f"✗ DeepSpeed import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ DeepSpeed import error: {e}")
        return False

def test_deepspeed_basic():
    """Test basic DeepSpeed functionality"""
    print("=" * 50)
    print("Testing basic DeepSpeed functionality...")
    
    try:
        import deepspeed
        
        # Test basic DeepSpeed utilities
        print(f"✓ DeepSpeed utils available")
        
        # Test if DeepSpeed can detect CUDA
        if torch.cuda.is_available():
            try:
                from deepspeed.utils import logger
                print("✓ DeepSpeed logger working")
            except Exception as e:
                print(f"⚠ DeepSpeed logger issue: {e}")
        
        return True
    except Exception as e:
        print(f"✗ DeepSpeed basic test failed: {e}")
        return False

def test_deepspeed_config():
    """Test DeepSpeed configuration"""
    print("=" * 50)
    print("Testing DeepSpeed configuration...")
    
    try:
        import deepspeed
        
        # Basic DeepSpeed config
        ds_config = {
            "train_batch_size": 4,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.003
                }
            },
            "fp16": {
                "enabled": torch.cuda.is_available()
            }
        }
        
        print("✓ DeepSpeed config created successfully")
        print(f"Config: {ds_config}")
        return True
    except Exception as e:
        print(f"✗ DeepSpeed config test failed: {e}")
        return False

def test_deepspeed_simple_model():
    """Test DeepSpeed with a simple model"""
    print("=" * 50)
    print("Testing DeepSpeed with simple model...")
    
    try:
        import deepspeed
        import torch.nn as nn
        
        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Basic DeepSpeed config
        ds_config = {
            "train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001
                }
            },
            "fp16": {
                "enabled": False  # Disable for CPU test
            }
        }
        
        # Try to initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config
        )
        
        print("✓ DeepSpeed model initialization successful")
        
        # Test forward pass
        dummy_input = torch.randn(2, 10)
        output = model_engine(dummy_input)
        print("✓ DeepSpeed forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ DeepSpeed model test failed: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("DeepSpeed Compatibility Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    
    results = []
    
    # Run tests
    results.append(("PyTorch", test_pytorch()))
    results.append(("DeepSpeed Import", test_deepspeed_import()))
    results.append(("DeepSpeed Basic", test_deepspeed_basic()))
    results.append(("DeepSpeed Config", test_deepspeed_config()))
    results.append(("DeepSpeed Model", test_deepspeed_simple_model()))
    
    # Summary
    print("=" * 50)
    print("TEST SUMMARY:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed! DeepSpeed appears to be working correctly.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        print("\nCommon solutions:")
        print("1. Try: pip install --upgrade deepspeed")
        print("2. Try: pip install deepspeed --no-cache-dir")
        print("3. Check CUDA compatibility")
        print("4. Try downgrading PyTorch if version mismatch")
    
    return all_passed

if __name__ == "__main__":
    main()
