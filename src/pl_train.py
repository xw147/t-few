import os
import torch
import argparse
from datetime import datetime

# PyTorch 2.6 compatibility fix for skip_code import error
try:
    # Try to import skip_code - if it fails, create a dummy function
    from torch._C._dynamo.eval_frame import skip_code
except (ImportError, AttributeError):
    # Create a compatibility layer for older/newer PyTorch versions
    import sys
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
    print("Warning: Using compatibility skip_code function for PyTorch version", torch.__version__)

# DeepSpeed compatibility fix for Windows
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
    print(f"DeepSpeed loaded successfully: {deepspeed.__version__}")
except ImportError as e:
    DEEPSPEED_AVAILABLE = False
    print(f"Warning: DeepSpeed not available: {e}")
    print("Continuing without DeepSpeed acceleration...")
except Exception as e:
    DEEPSPEED_AVAILABLE = False
    print(f"Warning: DeepSpeed failed to load: {e}")
    print("This might be due to Windows compilation issues. Continuing without DeepSpeed...")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import FinetuneDataModule, get_dataset_reader, PretrainDataModule
from src.models.EncoderDecoder import EncoderDecoder
from src.models.modify_model import modify_transformer
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds
from src.utils.environment import get_env_manager


def get_transformer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)

    tokenizer.model_max_length = config.max_seq_len
    model = modify_transformer(model, config)
    return tokenizer, model


def main(config):
    """
    Trains the model

    :param config:
    :return:
    """

    tokenizer, model = get_transformer(config)
    dataset_reader = get_dataset_reader(config)
    if config.dataset == "T0Mixture":
        datamodule = PretrainDataModule(config, tokenizer, dataset_reader)
    else:
        datamodule = FinetuneDataModule(config, tokenizer, dataset_reader)
    model = EncoderDecoder(config, tokenizer, model, dataset_reader)
    logger = TensorBoardLogger(config.exp_dir, name="log")

    # Create trainer with environment-appropriate settings
    trainer_kwargs = {
        "enable_checkpointing": False,
        "devices": "auto" if torch.cuda.is_available() else None,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "precision": config.compute_precision,
        "logger": logger,
        "log_every_n_steps": 4,
        "max_steps": config.num_steps,
        "min_steps": config.num_steps,
        "num_sanity_val_steps": -1 if config.eval_before_training else 0,
        "check_val_every_n_epoch": config.eval_epoch_interval,
        "accumulate_grad_batches": config.grad_accum_factor,
        "gradient_clip_val": config.grad_clip_norm,
    }
    
    # Only add strategy if it's not "none"
    if config.compute_strategy != "none":
        trainer_kwargs["strategy"] = config.compute_strategy
    
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    print(f"Start experiment {config.exp_name}")
    # Setup config
    env_manager = get_env_manager()
    
    # Handle DeepSpeed based on environment
    if env_manager.should_skip_deepspeed():
        if config.compute_strategy.startswith("deepspeed"):
            print("[DEBUG MODE] DeepSpeed strategy detected but disabled for compatibility.")
            print(f"   Changing compute_strategy from '{config.compute_strategy}' to 'none'")
            config.compute_strategy = "none"
        
        # Allowed strategies for debug/Windows
        allowed_strategies = ["none", "ddp"]
        assert config.compute_strategy in allowed_strategies, f"In debug mode, only {allowed_strategies} are supported"
    else:
        # Production mode - allow all strategies including DeepSpeed
        allowed_strategies = ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]
        assert config.compute_strategy in allowed_strategies, f"Supported strategies: {allowed_strategies}"
    if config.fishmask_mode == "create":
        print("Detecting fishmask_mode=create, override batch_size, num_step, fishmask_path")
        config.batch_size = 1
        config.num_steps = config.num_shot
        config.eval_before_training = False
        config.fishmask_path = None

    print(config.to_json())

    if config.allow_skip_exp and os.path.exists(config.finish_flag_file):
        print(f"Skip finished experiment {config.exp_name}")
    else:
        print(f"Mark experiment {config.exp_name} as claimed")
        with open(config.finish_flag_file, "a+") as f:
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
        set_seeds(config.seed)
        main(config)
