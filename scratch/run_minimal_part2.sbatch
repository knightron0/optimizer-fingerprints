#!/bin/bash
#SBATCH --job-name=opt-fingerprinting-2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=4:00:00
#SBATCH --account=cgoergen
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=a30
#SBATCH --mem=50G

cd /home/mangla/fingerprinting
uv run python -m fingerprinting run --optimizer sgd
uv run python -m fingerprinting run --optimizer sgd_momentum
uv run python -m fingerprinting run --optimizer lion
uv run python -m fingerprinting run --optimizer adafactor
