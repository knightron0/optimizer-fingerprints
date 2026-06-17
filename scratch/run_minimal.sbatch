#!/bin/bash
#SBATCH --job-name=opt-fingerprinting
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=4:00:00
#SBATCH --account=cgoergen
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=50G

cd /home/mangla/fingerprinting
uv run python -m fingerprinting run --optimizer adamw
uv run python -m fingerprinting run --optimizer muon
uv run python -m fingerprinting run --optimizer shampoo_default
uv run python -m fingerprinting run --optimizer shampoo_pinv_one_sided