"""
SAE adapter class that runs on precomputed CLS embeddings.
SAEAdapterMLPGrounded: encoder -> top-k/l1 sparsity -> decoder.
build/load helpers used by extract_om_embeddings and EVA wrapper.
"""
import os
import sys
import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _ensure_openmidnight_on_path(cfg):
    repo_root = cfg["backbone"].get("openmidnight_repo")
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _load_state_dict(backbone, checkpoint_path, checkpoint_key=None, override_pos_embed=False, strict=True):
    """Load state dict into backbone, optionally from a key and with pos_embed override."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint_key:
        if checkpoint_key not in state_dict:
            raise KeyError(f"Checkpoint key '{checkpoint_key}' not found in {checkpoint_path}.")
        state_dict = state_dict[checkpoint_key]
    # Strip DDP / wrapper prefixes so keys match
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
    if override_pos_embed and "pos_embed" in state_dict and hasattr(backbone, "pos_embed"):
        backbone.pos_embed = nn.Parameter(state_dict["pos_embed"])
    load_msg = backbone.load_state_dict(state_dict, strict=strict)
    missing = getattr(load_msg, "missing_keys", [])
    unexpected = getattr(load_msg, "unexpected_keys", [])
    logger.info("Loaded checkpoint from %s (strict=%s).", checkpoint_path, strict)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)
    return load_msg


def build_openmidnight_backbone(cfg):
    """Build OpenMidnight ViT-G/14 from config (no weights loaded here)."""
    _ensure_openmidnight_on_path(cfg)
    from dinov2.models.vision_transformer import vit_giant2

    patch_size = cfg["backbone"].get("patch_size", 14)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    img_size = cfg["backbone"].get("img_size", 224)
    if isinstance(img_size, (list, tuple)):
        img_size = tuple(img_size)

    return vit_giant2(
        patch_size=patch_size,
        img_size=img_size,
        in_chans=cfg["backbone"].get("in_channels", 3),
        num_register_tokens=cfg["backbone"].get("num_register_tokens", 4),
        ffn_layer=cfg["backbone"].get("ffn_layer", "swiglufused"),
        block_chunks=cfg["backbone"].get("block_chunks", 0),
        init_values=cfg["backbone"].get("init_values", 1.0),
    )


class SAEAdapterMLPGrounded(pl.LightningModule):
    """SAE on backbone CLS embeddings"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embedding_dim = cfg["backbone"]["hidden_size"]
        self.latent_dim = cfg["model"]["latent_dim"]
        self.sparsity_type = cfg["model"].get("sparsity_type", "topk")
        self.topk = int(cfg["model"].get("topk", 64))

        self.l1_weight = float(cfg["model"].get("l1_weight", 0.0))  # only for sparsity l1

        self.encoder = nn.Linear(self.embedding_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, self.embedding_dim, bias=True)

        # small std for better initialization 
        enc_std = float(cfg["model"].get("enc_init_std", 0.02))
        dec_std = float(cfg["model"].get("dec_init_std", 0.02))
        nn.init.normal_(self.encoder.weight, std=enc_std)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=dec_std)
        nn.init.zeros_(self.decoder.bias)

    def apply_sparsity(self, z_raw: torch.Tensor):
        if self.sparsity_type == "topk":
            k = min(self.topk, z_raw.size(1))
            idx = z_raw.abs().topk(k=k, dim=1, largest=True, sorted=False).indices  # (B, k)
            z = torch.zeros_like(z_raw)
            z.scatter_(1, idx, z_raw.gather(1, idx))
            l1_loss = z_raw.new_zeros(())
            return z, l1_loss, idx
        elif self.sparsity_type == "l1":
            z = z_raw
            l1_loss = z_raw.abs().sum(dim=1).mean()
            return z, l1_loss, None
        else:
            raise ValueError(f"Unknown sparsity_type: {self.sparsity_type}")

    def training_step(self, batch, batch_idx):
        if "embedding" not in batch:
            raise KeyError("Training uses precomputed embeddings only.")
        x = batch["embedding"]

        # SAE
        z_raw = self.encoder(x)
        z, l1_loss, idx = self.apply_sparsity(z_raw)
        x_hat = self.decoder(z)

        rec_loss = F.mse_loss(x_hat, x)

        total_loss = rec_loss
        if self.sparsity_type == "l1" and self.l1_weight > 0:
            total_loss = total_loss + self.l1_weight * l1_loss

        # Usage Fraction
        with torch.no_grad():
            nnz = (z != 0).float().mean()  # fraction nonzero over all entries
            z_active_abs_mean = z.abs().sum(dim=1).div((z != 0).sum(dim=1).clamp_min(1)).mean()

            # how many unique latents were used in this batch
            if idx is not None:
                unique_used = torch.unique(idx).numel()
                usage_frac = unique_used / float(self.latent_dim)
            else:
                unique_used = torch.tensor(0.0, device=self.device)
                usage_frac = torch.tensor(0.0, device=self.device)

        # log metrics 
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("rec_loss", rec_loss, prog_bar=True)
        self.log("z_nnz_frac", nnz)
        self.log("z_active_abs_mean", z_active_abs_mean)
        self.log("latents_used_frac_batch", usage_frac)

        if self.sparsity_type == "l1":
            self.log("l1_loss", l1_loss)

        return total_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg["optim"]["lr"]),
            weight_decay=float(self.cfg["optim"]["weight_decay"]),
        )
        return opt
