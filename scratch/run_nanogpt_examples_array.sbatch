#!/bin/bash
#SBATCH --job-name=nanogpt-fp
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --account=cgoergen
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=a100-40gb,a100-80gb
#SBATCH --mem=50G
#SBATCH --array=0-48

set -euo pipefail

cd /home/mangla/fingerprinting
source /scratch/gilbreth/mangla/venvs/fingerprint/bin/activate
export NANOGPT_SEED="${NANOGPT_SEED:-0}"

mapfile -t scripts < <(
  {
    printf '%s\n' nanogpt/examples/aurora.py
    find nanogpt/examples -maxdepth 1 -name 'record_*.py' | sort
  }
)

script="${scripts[$SLURM_ARRAY_TASK_ID]}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "script=${script}"
echo "NANOGPT_SEED=${NANOGPT_SEED}"

torchrun --standalone --nproc_per_node=2 "${script}"
