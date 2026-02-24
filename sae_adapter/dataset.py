"""Datasets for embeddings (memmap/sharded) and WSI patch lists. 
SlideListPatchDataset for extract_om_embeddings; 
ShardedMemmapEmbeddingDataset for training."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from openslide import OpenSlide


class ShardedMemmapEmbeddingDataset(Dataset):
    """This dataset provides a global index over several .npy embedding shards, loading each entry as a writeable tensor for PyTorch."""

    def __init__(self, embedding_npy_list):
        if not embedding_npy_list:
            raise ValueError("embedding_npy_list is required for ShardedMemmapEmbeddingDataset.")
        if isinstance(embedding_npy_list, str):
            embedding_npy_list = [embedding_npy_list]
        self.embedding_paths = list(embedding_npy_list)
        self.embeddings = [np.load(path, mmap_mode="r") for path in self.embedding_paths]
        self.lengths = [arr.shape[0] for arr in self.embeddings]
        self.cum_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return int(self.cum_lengths[-1])

    def __getitem__(self, idx):
        idx = int(idx)
        # Map global index to which file and which row inside it
        file_idx = int(np.searchsorted(self.cum_lengths, idx, side="right"))
        prev = 0 if file_idx == 0 else int(self.cum_lengths[file_idx - 1])
        local_idx = idx - prev
        # .copy() so torch doesn't complain about memmap read-only
        return {"embedding": torch.from_numpy(self.embeddings[file_idx][local_idx].copy())}


class SlideListPatchDataset(Dataset):
    """WSI patch dataset from a text file of lines 'path x y level'."""

    def __init__(self, sample_list_path, transform=None, patch_size=224, return_raw=False, return_metadata=False):
        self.sample_list_path = sample_list_path
        self.transform = transform
        self.patch_size = patch_size
        self.return_raw = return_raw
        self.return_metadata = return_metadata
        if not os.path.isfile(self.sample_list_path):
            raise FileNotFoundError(f"Sample list not found: {self.sample_list_path}")
        with open(self.sample_list_path, "r") as f:
            self.samples = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        parts = self.samples[idx].split(" ")
        if len(parts) != 4:
            raise ValueError(f"Invalid sample line: {self.samples[idx]}")
        path, x, y, level = parts
        x, y, level = int(x), int(y), int(level)

        # One OpenSlide per patch (no caching) to keep memory sane
        slide = OpenSlide(path)
        patch = slide.read_region((x, y), level=level, size=(self.patch_size, self.patch_size))
        slide.close()
        image = patch.convert("RGB")
        metadata = {"path": path, "x": x, "y": y, "level": level}
        raw_image = np.array(image) if self.return_raw else None
        if self.transform:
            image = self.transform(image)
        sample = {"image": image}
        if self.return_raw:
            sample["image_raw"] = raw_image
        if self.return_metadata:
            sample["metadata"] = metadata
        return sample
