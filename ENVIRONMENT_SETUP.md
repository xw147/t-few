# Simplified Training Setup - One Script, Three Modes

This setup allows you to use a single script (`few-shot-pretrained-100k.sh`) for all scenarios: debug testing, HPC validation, and comprehensive experiments.

## Quick Start

### Step 1: Test on Your Office Computer (Windows)

```bash
# 1. Edit few-shot-pretrained-100k.sh and set:
DEBUG_MODE=true
SIMPLE_HPC_TEST=false  
COMPREHENSIVE_MODE=false

# 2. Run comprehensive test first
python comprehensive_test.py

# 3. Run debug training (2 steps, minimal resources)
bash bin/few-shot-pretrained-100k.sh
```

### Step 2: Validate on University HPC (Linux)

```bash
# 1. Edit few-shot-pretrained-100k.sh and set:
DEBUG_MODE=false
SIMPLE_HPC_TEST=true
COMPREHENSIVE_MODE=false

# 2. Run simple validation (100 steps, 1 seed)
bash bin/few-shot-pretrained-100k.sh
```

### Step 3: Full Experiments on HPC

```bash
# 1. Edit few-shot-pretrained-100k.sh and set:
DEBUG_MODE=false
SIMPLE_HPC_TEST=false
COMPREHENSIVE_MODE=true

# 2. Gradually uncomment lines to expand experiments:
# seeds="42 1024 0 1 32"           # Uncomment for all seeds
# shot_numbers="4 8 16 32 64 128 256 512"  # Uncomment for all shots
# models="t03b t011b"              # Uncomment for both models

bash bin/few-shot-pretrained-100k.sh
```

## Environment Detection

The system automatically detects your environment and applies appropriate settings:

- **Windows** → Debug mode settings (no DeepSpeed, minimal resources)
- **HPC clusters** → Production mode settings (with DeepSpeed, full resources)
- **Unknown** → Debug mode settings (safe default)

## Mode Settings Comparison

| Setting | Debug Mode | Simple HPC | Comprehensive |
|---------|------------|------------|---------------|
| Steps | 2 | 100 | Full (dataset-specific) |
| Batch Size | 2 | 4 | 4 |
| Seeds | 1 (42) | 1 (42) | Expandable |
| Shots | 1 (2) | 1 (4) | Expandable |
| Models | 1 (t03b) | 1 (t03b) | Expandable |
| DeepSpeed | No | Yes | Yes |
| Purpose | Quick test | HPC validation | Full experiments |

## Experiment Grid Expansion

Start small and expand gradually by uncommenting lines:

```bash
# Start with minimal experiment
models="t03b"           # 1 model
shot_numbers="4"        # 1 shot setting  
seeds="42"              # 1 seed
datasets="ico_list"     # 1 dataset
# Total: 1 experiment

# Expand to multiple seeds
seeds="42 1024 0 1 32"  # 5 seeds
# Total: 5 experiments

# Expand to multiple shots
shot_numbers="4 8 16 32 64 128 256 512"  # 8 shot settings
# Total: 5 seeds × 8 shots = 40 experiments

# Expand to multiple models  
models="t03b t011b"     # 2 models
# Total: 2 models × 8 shots × 5 seeds = 80 experiments
```

```bash
# Force debug mode (even on HPC)
TFEW_DEBUG=true bash bin/smart_train.sh

# Force production mode (even on Windows, but DeepSpeed might fail)
TFEW_PRODUCTION=true bash bin/smart_train.sh
```

## Files in This Setup

### Core Files
- `bin/few-shot-pretrained-100k.sh` - Main training script (3 modes in one)
- `comprehensive_test.py` - Complete test suite for office computer
- `configs/debug_config.json` - Debug vs production settings
- `src/utils/environment.py` - Environment detection logic

### Configuration Files
- `src/utils/Config.py` - Enhanced configuration with environment overrides
- `src/pl_train.py` - Smart DeepSpeed handling based on environment
- `src/models/EncoderDecoder.py` - Environment-aware DeepSpeed usage

## Your Workflow

```
Office Computer (Windows)
├── 1. Edit few-shot-pretrained-100k.sh → DEBUG_MODE=true
├── 2. python comprehensive_test.py → Test all components
└── 3. bash bin/few-shot-pretrained-100k.sh → Quick validation (2 steps)

Copy to HPC (Linux)
├── 4. Edit few-shot-pretrained-100k.sh → SIMPLE_HPC_TEST=true  
├── 5. bash bin/few-shot-pretrained-100k.sh → HPC validation (100 steps)
└── 6. Edit → COMPREHENSIVE_MODE=true → Full experiments

Expand Experiments (Gradually)
├── 7. Uncomment seeds line → Multiple random seeds
├── 8. Uncomment shots line → Multiple shot numbers
├── 9. Uncomment models line → Multiple models
└── 10. Run comprehensive grid → Full experimental suite
```

## Testing Your Setup

Before deploying to HPC, run the comprehensive test:

```bash
python comprehensive_test.py
```

This tests:
- Environment detection
- Configuration loading with overrides
- Model loading with adapters
- Data loading with your dataset
- PyTorch Lightning setup
- Full pipeline without training

## Expected Behavior

| Environment | Mode | DeepSpeed | Steps | Time | GPU Memory |
|-------------|------|-----------|-------|------|------------|
| Windows Office | Debug | No | 2 | <5 min | <2GB |
| Linux HPC | Simple Test | Yes | 100 | ~10 min | 4-8GB |
| Linux HPC | Comprehensive | Yes | 15000+ | 2-4 hours | 8-16GB |

## Troubleshooting

**Office Computer Issues:**
- Run `python comprehensive_test.py` to identify problems
- Check Python/PyTorch installation
- Verify transformers/pytorch-lightning versions

**HPC Issues:**
- Check DeepSpeed: `python -c "import deepspeed; print(deepspeed.__version__)"`
- Verify CUDA compatibility
- Check GPU memory availability
- Review SLURM/scheduler logs

## Benefits of This Setup

1. **Single Script** - No need to manage multiple training scripts
2. **Progressive Expansion** - Start small, grow the experiment grid
3. **Environment Aware** - Automatically adapts to Windows vs HPC
4. **Clear Configuration** - Easy to see and modify experimental parameters
5. **Predictable Scaling** - Know exactly how many experiments you're running

Your office computer handles quick validation, while HPC handles the heavy computational work!
