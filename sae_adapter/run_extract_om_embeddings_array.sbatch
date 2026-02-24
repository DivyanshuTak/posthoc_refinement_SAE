#!/bin/bash
#SBATCH --job-name=om-embeds
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/om_embeds_%A_%a.out
#SBATCH --error=logs/om_embeds_%A_%a.err

set -euo pipefail

# Repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NUM_SHARDS=8
export SHARD_ID=${SLURM_ARRAY_TASK_ID}
export NUM_SHARDS=${NUM_SHARDS}

mkdir -p "${SCRIPT_DIR}/logs"

cd "${SCRIPT_DIR}"
source "${REPO_ROOT}/OpenMidnight/.venv/bin/activate"

python extract_om_embeddings.py
