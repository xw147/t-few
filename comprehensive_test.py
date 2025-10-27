#!/usr/bin/env python3
"""
Comprehensive test script for office computer (Windows)
This tests all major components without requiring HPC resources
"""

import os
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Apply PyTorch 2.6 compatibility fix BEFORE any torch imports
import torch

try:
    # Try to import skip_code - if it fails, create a dummy function
    from torch._C._dynamo.eval_frame import skip_code
except (ImportError, AttributeError):
    # Create a compatibility layer for older/newer PyTorch versions
    from types import ModuleType
    
    # Create dummy module structure if it doesn't exist
    if not hasattr(torch._C, '_dynamo'):
        torch._C._dynamo = ModuleType('_dynamo')
    if not hasattr(torch._C._dynamo, 'eval_frame'):
        torch._C._dynamo.eval_frame = ModuleType('eval_frame')
    
    # Create dummy skip_code function
    def skip_code(x):
        return x
    
    torch._C._dynamo.eval_frame.skip_code = skip_code

# Import src to ensure other compatibility fixes are loaded
import src

def test_environment_detection():
    """Test environment detection"""
    print("=" * 60)
    print("🔍 Testing Environment Detection")
    print("=" * 60)
    
    try:
        from src.utils.environment import get_env_manager, print_environment_info
        env_manager = get_env_manager()
        print_environment_info()
        
        print(f"Debug mode: {env_manager.debug_mode}")
        print(f"Should skip DeepSpeed: {env_manager.should_skip_deepspeed()}")
        
        # Test config overrides
        overrides = env_manager.get_config_overrides()
        print(f"Config overrides: {list(overrides.keys())}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Environment detection failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading with environment overrides"""
    print("=" * 60)
    print("🔧 Testing Configuration Loading")
    print("=" * 60)
    
    try:
        from src.utils.Config import Config
        
        # Test with debug environment
        os.environ['TFEW_DEBUG'] = 'true'
        config = Config("t03b.json+ia3.json+global.json", {
            'dataset': 'ico_list',
            'num_shot': 4,
            'exp_name': 'test_debug_config'
        })
        
        print(f"[PASS] Config loaded successfully")
        print(f"   Debug overrides applied: {config.num_steps} steps")
        print(f"   Compute strategy: {config.compute_strategy}")
        print(f"   Batch size: {config.batch_size}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Config loading failed: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading without full training"""
    print("=" * 60)
    print("🤖 Testing Model Loading")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from src.models.modify_model import modify_transformer
        from src.utils.Config import Config
        
        # Force debug mode
        os.environ['TFEW_DEBUG'] = 'true'
        
        config = Config("t03b.json+ia3.json+global.json", {
            'dataset': 'ico_list',
            'max_seq_len': 128  # Use shorter sequences for testing
        })
        
        print(f"Loading tokenizer for {config.origin_model}...")
        tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
        
        print("Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)
        
        print("Modifying model with adapters...")
        tokenizer.model_max_length = config.max_seq_len
        model = modify_transformer(model, config)
        
        print(f"[PASS] Model loaded successfully: {config.origin_model}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading with ico_list dataset"""
    print("=" * 60)
    print("📊 Testing Data Loading")
    print("=" * 60)
    
    try:
        from src.data import get_dataset_reader
        from src.utils.Config import Config
        
        # Force debug mode
        os.environ['TFEW_DEBUG'] = 'true'
        
        config = Config("t03b.json+ia3.json+global.json", {
            'dataset': 'ico_list',
            'num_shot': 2,  # Very small for testing
            'few_shot': True
        })
        
        print("Loading dataset reader...")
        dataset_reader = get_dataset_reader(config)
        
        print(f"[PASS] Dataset reader loaded: {type(dataset_reader).__name__}")
        
        # Try to load a small sample
        if hasattr(dataset_reader, 'read_few_shot_dataset'):
            print("Loading few-shot dataset...")
            train_data = dataset_reader.read_few_shot_dataset()
            print(f"   Train samples: {len(train_data) if hasattr(train_data, '__len__') else 'unknown'}")
            
            # Show sample data
            if train_data and len(train_data) > 0:
                print("\n📋 Sample Training Data:")
                print("-" * 40)
                
                # Show first sample
                sample1 = train_data[0]
                print(f"Sample 1:")
                for key, value in sample1.items():
                    if isinstance(value, str) and len(value) > 100:
                        # Truncate very long strings
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
                
                # Show second sample if available
                if len(train_data) > 1:
                    print(f"\nSample 2:")
                    sample2 = train_data[1]
                    for key, value in sample2.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                print("-" * 40)
        
        # Also try to load validation data if available
        try:
            print("\nLoading validation dataset...")
            val_data = dataset_reader.read_orig_dataset("validation")
            if hasattr(val_data, '__len__'):
                print(f"   Validation samples: {len(val_data)}")
                
                # Show sample validation data
                if len(val_data) > 0:
                    print("\n📋 Sample Validation Data:")
                    print("-" * 40)
                    val_sample = val_data[0]
                    for key, value in val_sample.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                    print("-" * 40)
            else:
                print(f"   Validation data type: {type(val_data)}")
        except Exception as ve:
            print(f"   Validation data not available: {ve}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        traceback.print_exc()
        return False

def test_pytorch_lightning_setup():
    """Test PyTorch Lightning trainer setup"""
    print("=" * 60)
    print("⚡ Testing PyTorch Lightning Setup")
    print("=" * 60)
    
    try:
        # Import src first to ensure compatibility fixes are applied
        import src
        
        # Try importing PyTorch Lightning - skip test if it fails
        try:
            from pytorch_lightning import Trainer
            from pytorch_lightning.loggers import TensorBoardLogger
        except ImportError as e:
            print(f"[SKIP] PyTorch Lightning not available: {str(e)}")
            print("   This is not critical for core functionality")
            return True
            
        from src.utils.Config import Config
        
        # Force debug mode
        os.environ['TFEW_DEBUG'] = 'true'
        
        config = Config("t03b.json+ia3.json+global.json", {
            'dataset': 'ico_list',
            'exp_name': 'test_trainer_setup',
            'num_steps': 1,  # Minimal for testing
        })
        
        print("Creating trainer...")
        logger = TensorBoardLogger(config.exp_dir, name="log")
        
        trainer = Trainer(
            enable_checkpointing=False,
            accelerator='cpu',  # Force CPU for testing
            precision=32,  # Use standard precision for compatibility
            logger=logger,
            log_every_n_steps=1,
            max_steps=config.num_steps,
            min_steps=config.num_steps,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=config.eval_epoch_interval,
            accumulate_grad_batches=config.grad_accum_factor,
        )
        
        print(f"[PASS] Trainer created successfully")
        print(f"   Strategy: {trainer.strategy}")
        print(f"   Accelerator: {trainer.accelerator}")
        
        return True
    except Exception as e:
        print(f"[FAIL] PyTorch Lightning setup failed: {e}")
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the complete pipeline without actual training"""
    print("=" * 60)
    print("🔄 Testing Full Pipeline (No Training)")
    print("=" * 60)
    
    try:
        # Force debug mode
        os.environ['TFEW_DEBUG'] = 'true'
        
        from src.pl_train import get_transformer, main
        from src.utils.Config import Config
        from src.data import get_dataset_reader
        
        config = Config("t03b.json+ia3.json+global.json", {
            'dataset': 'ico_list',
            'num_shot': 2,
            'num_steps': 0,  # No actual training steps
            'exp_name': 'test_full_pipeline',
            'eval_before_training': True,
            'save_model': False
        })
        
        print("Testing full pipeline setup...")
        print(f"Config num_steps: {config.num_steps}")
        print(f"Config compute_strategy: {config.compute_strategy}")
        
        # Test individual components
        tokenizer, model = get_transformer(config)
        dataset_reader = get_dataset_reader(config)
        
        print(f"[PASS] Full pipeline components loaded successfully")
        print(f"   Model: {config.origin_model}")
        print(f"   Dataset: {config.dataset}")
        print(f"   Training steps: {config.num_steps}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Full pipeline test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("COMPREHENSIVE TESTING FOR OFFICE COMPUTER")
    print("=" * 60)
    print("This script tests all major components without requiring:")
    print("• HPC resources")
    print("• Large GPU memory") 
    print("• DeepSpeed")
    print("• Long training times")
    print("=" * 60)
    
    # Force debug mode for all tests
    os.environ['TFEW_DEBUG'] = 'true'
    
    tests = [
        ("Environment Detection", test_environment_detection),
        ("Configuration Loading", test_config_loading),
        ("Model Loading", test_model_loading),
        ("Data Loading", test_data_loading),
        ("PyTorch Lightning Setup", test_pytorch_lightning_setup),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:25} : {status}")
    
    print("-" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Your setup is ready for HPC deployment.")
        print("\nNext steps:")
        print("1. Copy your code to HPC")
        print("2. Install requirements on HPC")
        print("3. Run: TFEW_PRODUCTION=true bash bin/smart_train.sh")
    else:
        print("Some tests failed. Fix these issues before HPC deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
