#!/bin/bash
#SBATCH --job-name=uni_test
#SBATCH --account=<your-account>
#SBATCH --partition=gpu
#SBATCH --constraint=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/beagle3/<user>/<workdir>/logs/uni_test_%j.out
#SBATCH --error=/scratch/beagle3/<user>/<workdir>/logs/uni_test_%j.err

set -eo pipefail
mkdir -p /scratch/beagle3/<user>/<workdir>/logs

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/beagle3/<user>/dinov2_env

export XFORMERS_DISABLED=1
export TORCH_HOME=/scratch/beagle3/<user>/<workdir>/torch_hub_cache
export HF_HOME=/scratch/beagle3/<user>/<workdir>/torch_hub_cache/hf

python /scratch/beagle3/<user>/<workdir>/infer_dataset.py \
  --dataset crc_val_7k \
  --data-dir /scratch/beagle3/<user>/<workdir>/data/CRC-VAL-HE-7K \
  --format file \
  --models uni \
  --batch-size 16 \
  --max-patches 500
