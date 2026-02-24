#!/usr/bin/env python3
"""
Split the patch list into N shards for parallel embedding extraction.
Run: shard_embeddings.py <shard_id> <num_shards>
"""
import os
import sys

import yaml


def main():
   
    if len(sys.argv) != 3:
        raise SystemExit("Run: python shard_embeddings.py <shard_id> <num_shards>")
    shard_id = int(sys.argv[1])
    num_shards = int(sys.argv[2])
    

    with open("config_openmidnight.yml", "r") as file:
        cfg = yaml.safe_load(file)

    sample_list_path = cfg["data"]["sample_list_path"]
  
    shard_dir = cfg["data"].get("shard_list_dir")
    if not shard_dir:
        base_dir = cfg["data"].get("embeddings_output_dir", "./embeddings")
        shard_dir = os.path.join(base_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    output_name = os.path.basename(sample_list_path).replace(".txt", "")
    shard_path = os.path.join(shard_dir, f"{output_name}_shard{shard_id:02d}-of-{num_shards:02d}.txt")

    with open(sample_list_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

   
    with open(shard_path, "w") as out:
        for idx, line in enumerate(lines):
            if idx % num_shards == shard_id:
                out.write(line + "\n")

    print(f"Wrote shard {shard_id}/{num_shards} to: {shard_path}")


if __name__ == "__main__":
    main()
