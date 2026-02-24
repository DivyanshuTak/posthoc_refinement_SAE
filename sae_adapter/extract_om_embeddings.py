#!/usr/bin/env python3
"""Extract OM CLS embeddings from patches """
import os

import numpy as np
import cv2
import torch
import yaml
import torchvision.transforms as T
from tqdm import tqdm

from dataset import SlideListPatchDataset
from model_mlp_grounded import build_openmidnight_backbone, _load_state_dict


def build_patch_transform(size):
    if isinstance(size, (list, tuple)) and len(size) == 2:
        resize_size = tuple(size)
    else:
        resize_size = (224, 224)
    return T.Compose(
        [
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# Skip patches that are mostly background (HSV tissue filter)
def _accept_patch_hsv(tile_rgb, min_ratio, lower_bound, upper_bound):
    tile = np.asarray(tile_rgb)
    tile = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(tile, lower_bound, upper_bound)
    ratio = np.count_nonzero(mask) / mask.size
    return ratio > min_ratio


def extract_embeddings(
    backbone,
    dataloader,
    output_npy: str,
    dtype: str = "float16",
    filter_tissue: bool = False,
    min_ratio: float = 0.6,
    hsv_lower: tuple = (90, 8, 103),
    hsv_upper: tuple = (180, 255, 255),
):
    """Run backbone on dataloader; optionally filter by HSV tissue ratio. Writes embeddings to .npy."""
    total = len(dataloader.dataset)
    embed_dim = 1536
    temp_npy = output_npy + ".tmp"
    out = np.lib.format.open_memmap(
        temp_npy,
        mode="w+",
        dtype=dtype,
        shape=(total, embed_dim),
    )

    accepted = 0
    skipped = 0
    lower_bound = np.array(hsv_lower, dtype=np.uint8)
    upper_bound = np.array(hsv_upper, dtype=np.uint8)

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        images = batch["image"]
        raw_images = batch.get("image_raw")
        orig_batch = images.shape[0]

        if filter_tissue and raw_images is not None:
            keep = [i for i, img in enumerate(raw_images) if _accept_patch_hsv(img, min_ratio, lower_bound, upper_bound)]
            if not keep:
                skipped += orig_batch
                continue
            images = images[keep]
            skipped += orig_batch - len(keep)

        images = images.to(next(backbone.parameters()).device)
        with torch.no_grad():
            features = backbone(images)
            if isinstance(features, dict) and "x_norm_clstoken" in features:
                embedding = features["x_norm_clstoken"]
            elif isinstance(features, (list, tuple)):
                embedding = features[0][:, 0, :]
            elif isinstance(features, torch.Tensor):
                embedding = features if features.dim() == 2 else features[:, 0, :]
            else:
                raise RuntimeError(f"Unexpected backbone output type: {type(features)}")

        batch_np = embedding.detach().cpu().to(torch.float16 if dtype == "float16" else torch.float32).numpy()
        out[accepted : accepted + batch_np.shape[0]] = batch_np
        accepted += batch_np.shape[0]

    out.flush()

    if accepted < total:
        final_out = np.lib.format.open_memmap(
            output_npy,
            mode="w+",
            dtype=dtype,
            shape=(accepted, embed_dim),
        )
        final_out[:] = out[:accepted]
        final_out.flush()
        os.remove(temp_npy)
    else:
        os.replace(temp_npy, output_npy)
    return accepted, skipped


def main():
   
    with open("config_openmidnight.yml", "r") as file:
        cfg = yaml.safe_load(file)

    output_dir = cfg["data"].get("embeddings_output_dir", "./embeddings")
    output_name = cfg["data"].get("embeddings_file", "embeddings.npy")
    sample_list_path = cfg["data"]["sample_list_path"]

    # use a shard of the patch list and name outputs with shard suffix for runnign on cluster
    shard_id = os.environ.get("SHARD_ID")
    num_shards = os.environ.get("NUM_SHARDS")
    if shard_id is not None and num_shards is not None:
        shard_id = int(shard_id)
        num_shards = int(num_shards)
        shard_list_dir = os.environ.get("SHARD_LIST_DIR") or cfg["data"].get(
            "shard_list_dir", os.path.join(output_dir, "shards")
        )
        base_name = os.path.basename(sample_list_path).replace(".txt", "")
        shard_file = f"{base_name}_shard{shard_id:02d}-of-{num_shards:02d}.txt"
        sample_list_path = os.path.join(shard_list_dir, shard_file)

        def _with_shard_suffix(name, suffix):
            if "{shard}" in name:
                return name.format(shard=suffix)
            root, ext = os.path.splitext(name)
            return f"{root}_{suffix}{ext}"

        shard_suffix = f"shard{shard_id:02d}-of-{num_shards:02d}"
        output_name = _with_shard_suffix(output_name, shard_suffix)

    output_name = os.environ.get("EMBEDDINGS_FILE", output_name)
    os.makedirs(output_dir, exist_ok=True)
    output_npy = os.path.join(output_dir, output_name)

    # Build backbone and load weights
    backbone = build_openmidnight_backbone(cfg)
    checkpoint_path = cfg["backbone"].get("checkpoint")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_key = cfg["backbone"].get("checkpoint_key")
        if checkpoint_key in (None, "", "none", "None"):
            checkpoint_key = None
        _load_state_dict(
            backbone,
            checkpoint_path=checkpoint_path,
            checkpoint_key=checkpoint_key,
            override_pos_embed=bool(cfg["backbone"].get("override_pos_embed", False)),
            strict=bool(cfg["backbone"].get("load_strict", False)),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    backbone.eval()

    # Dataset / dataloader
    transform = build_patch_transform(cfg["data"].get("size", [224, 224]))
    dataset = SlideListPatchDataset(
        sample_list_path=sample_list_path,
        transform=transform,
        patch_size=cfg["data"].get("patch_size", 224),
        return_raw=bool(cfg["data"].get("filter_tissue", False)),
        return_metadata=False,
    )
    batch_size = cfg["data"].get("batch_size", 64)
    num_workers = cfg["data"].get("num_workers", 4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    accepted, skipped = extract_embeddings(
        backbone,
        dataloader,
        output_npy,
        dtype=cfg["data"].get("embeddings_dtype", "float16"),
        filter_tissue=bool(cfg["data"].get("filter_tissue", False)),
        min_ratio=float(cfg["data"].get("filter_min_ratio", 0.6)),
        hsv_lower=tuple(cfg["data"].get("filter_hsv_lower", [90, 8, 103])),
        hsv_upper=tuple(cfg["data"].get("filter_hsv_upper", [180, 255, 255])),
    )

    print(f"Saved embeddings to: {output_npy} | Accepted: {accepted} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
