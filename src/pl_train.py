import os
import torch
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import FinetuneDataModule, get_dataset_reader, PretrainDataModule
from src.models.EncoderDecoder import EncoderDecoder
from src.models.modify_model import modify_transformer
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds


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

    # Handle PyTorch Lightning 2.x API changes
    # In Lightning 2.x, 'gpus' is deprecated, use 'devices' and 'accelerator'
    import pytorch_lightning as pl
    pl_version = int(pl.__version__.split('.')[0])
    
    if pl_version >= 2:
        # PyTorch Lightning 2.x API
        trainer_kwargs = {
            "enable_checkpointing": False,
            "accelerator": "mps" if torch.backends.mps.is_available() else "cpu",
            "devices": 1,  # Use 1 device (MPS or CPU)
            "precision": "16-mixed" if config.compute_precision == 16 else 32,
            "strategy": config.compute_strategy if config.compute_strategy != "none" else "auto",
            "logger": logger,
            "log_every_n_steps": 4,
            "max_steps": config.num_steps,
            "min_steps": config.num_steps,
            "num_sanity_val_steps": -1 if config.eval_before_training else 0,
            "check_val_every_n_epoch": config.eval_epoch_interval,
            "accumulate_grad_batches": config.grad_accum_factor,
            "gradient_clip_val": config.grad_clip_norm,
        }
    else:
        # PyTorch Lightning 1.x API (legacy)
        trainer_kwargs = {
            "enable_checkpointing": False,
            "gpus": torch.cuda.device_count(),
            "precision": config.compute_precision,
            "amp_backend": "native",
            "strategy": config.compute_strategy if config.compute_strategy != "none" else None,
            "logger": logger,
            "log_every_n_steps": 4,
            "max_steps": config.num_steps,
            "min_steps": config.num_steps,
            "num_sanity_val_steps": -1 if config.eval_before_training else 0,
            "check_val_every_n_epoch": config.eval_epoch_interval,
            "accumulate_grad_batches": config.grad_accum_factor,
            "gradient_clip_val": config.grad_clip_norm,
        }
    
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
    assert config.compute_strategy in ["none", "ddp"], \
        "Only 'none' and 'ddp' strategies are supported on macOS. DeepSpeed requires Linux."
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
