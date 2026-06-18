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

    # Build Trainer kwargs in a way that works for both Lightning 1.x and 2.x.
    import pytorch_lightning as pl
    pl_version = int(pl.__version__.split('.')[0])

    if pl_version >= 2:
        # ---- PyTorch Lightning 2.x API --------------------------------------
        # Pick the best available accelerator: CUDA (HPC) > MPS (mac) > CPU.
        # (The original only chose mps/cpu, which silently fell back to CPU on
        #  a CUDA cluster — i.e. it would NOT use the GPU.)
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

        # Map compute_precision -> Lightning 2.x precision string.
        # bf16 is strongly preferred for T5/T0 (fp16 tends to produce NaNs).
        if config.compute_precision in (16, "16", "fp16"):
            precision = "16-mixed"
        elif config.compute_precision in ("bf16", "bfloat16"):
            precision = "bf16-mixed"
        else:
            precision = "32-true"

        trainer_kwargs = {
            "enable_checkpointing": False,
            "accelerator": accelerator,
            "devices": 1,
            "precision": precision,
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
        # ---- PyTorch Lightning 1.x API (this is what you run: 1.9.5) ---------
        # gpus=device_count() uses the GPU Slurm allocated (1 with --gres=gpu:1).
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

    # Clamp log_every_n_steps so it never exceeds the number of training batches
    # (num_shot // batch_size).  This avoids a PL 1.9.x bug where
    # log_every_n_steps > num_batches causes the Adafactor closure() to receive
    # a detached loss tensor → RuntimeError: element 0 of tensors does not require grad.
    # Works for any num_shot value (4, 8, 16, 32, 64, 128, ...).
    num_batches_per_epoch = max(1, config.num_shot // config.batch_size)
    trainer_kwargs["log_every_n_steps"] = max(1, min(config.log_every_n_steps, num_batches_per_epoch))

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
        "compute_strategy must be 'none' or 'ddp' here. To use DeepSpeed (e.g. for " \
        "T0-11B that won't fit otherwise), add the deepspeed strategy to this assert " \
        "and pass it through trainer_kwargs."
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
