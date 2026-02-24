#!/usr/bin/env bash
#SBATCH --job-name=sae_om
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/sae_om_%j.out
#SBATCH --error=logs/sae_om_%j.err
#SBATCH --partition=main

set -euo pipefail

# Repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${SCRIPT_DIR}/logs"

cd "${SCRIPT_DIR}"
source "${REPO_ROOT}/OpenMidnight/.venv/bin/activate"

python train_sae_adapter.py
