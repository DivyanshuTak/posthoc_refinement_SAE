"""Train SAE adapter on precomputed OpenMidnight embeddings (sharded .npy). Uses config_openmidnight.yml."""
import os
import glob
import yaml
import numpy as np
import torch
import pytorch_lightning as pl
import wandb
import re
from monai.utils import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import ShardedMemmapEmbeddingDataset
from model_mlp_grounded import SAEAdapterMLPGrounded


# helpers to fill in {model.topk}, {optim.lr:.0e}, in logger run_name and save_name from config
def _resolve_cfg_value(cfg, key):
    current = cfg
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current

def _format_with_cfg(template, cfg):
    def replace(match):
        token = match.group(1)
        if ":" in token:
            key, spec = token.split(":", 1)
        else:
            key, spec = token, ""
        value = _resolve_cfg_value(cfg, key)
        if value is None:
            return match.group(0)
        try:
            return format(value, spec) if spec else str(value)
        except Exception:
            return str(value)

    return re.sub(r"\{([^{}]+)\}", replace, template)


if __name__ == "__main__":
    with open("config_openmidnight.yml", "r") as file:
        cfg = yaml.safe_load(file)

    # expand {model.topk} etc. in logger strings
    if "logger" in cfg:
        if "run_name" in cfg["logger"]:
            cfg["logger"]["run_name"] = _format_with_cfg(cfg["logger"]["run_name"], cfg)
        if "save_name" in cfg["logger"]:
            cfg["logger"]["save_name"] = _format_with_cfg(cfg["logger"]["save_name"], cfg)

    os.makedirs(cfg["logger"]["save_dir"], exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["trainer"]["devices"][0])
    wandb.init(project=cfg["logger"]["project"], name=cfg["logger"]["run_name"])
    wandb.config.update(cfg)
    wandb_logger = WandbLogger()
    set_determinism(seed=0)
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    torch.set_float32_matmul_precision("medium")

    if not cfg["data"].get("use_embeddings", False):
        raise ValueError("Training is on precomputed embeddings only")
    

    # point to sharded .npy files 
    embedding_npy = cfg["data"].get("embedding_npy")
    embeddings_glob = cfg["data"].get("embeddings_glob")
    embeddings_dir = cfg["data"].get("embeddings_dir")

    if embeddings_glob:
        glob_base = embeddings_dir if embeddings_dir else ""
        glob_path = os.path.join(glob_base, embeddings_glob)
        npy_files = sorted(glob.glob(glob_path))
        if not npy_files:
            raise FileNotFoundError(f"No embedding shards found for pattern: {glob_path}")
        train_dataset = ShardedMemmapEmbeddingDataset(npy_files)
    elif embedding_npy:  # for single npy file 
        paths = embedding_npy if isinstance(embedding_npy, (list, tuple)) else [embedding_npy]
        train_dataset = ShardedMemmapEmbeddingDataset(paths)
    else:
        raise ValueError("Provide data.embeddings_glob or data.embedding_npy")

    # spinup dataloader 
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # load model 
    model = SAEAdapterMLPGrounded(cfg)


    # setup callbacks and trainer 
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        dirpath=cfg["logger"]["save_dir"],
        filename=cfg["logger"]["save_name"],
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=cfg["trainer"]["devices"],
        precision=cfg["trainer"]["precision"],
        logger=wandb_logger,
        log_every_n_steps=100,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
    )

    # run training 
    trainer.fit(model, train_loader)
