#!/bin/bash
#SBATCH --job-name=hoptimus0_full
#SBATCH --account=<your-account>
#SBATCH --partition=gpu
#SBATCH --constraint=rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/beagle3/<user>/<workdir>/logs/hoptimus0_full_%j.out
#SBATCH --error=/scratch/beagle3/<user>/<workdir>/logs/hoptimus0_full_%j.err

set -eo pipefail
mkdir -p /scratch/beagle3/<user>/<workdir>/logs

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/beagle3/<user>/dinov2_env

export XFORMERS_DISABLED=1
export TORCH_HOME=/scratch/beagle3/<user>/<workdir>/torch_hub_cache
export HF_HOME=/scratch/beagle3/<user>/<workdir>/torch_hub_cache/hf

echo "=== CRC-VAL-HE-7K ==="
python /scratch/beagle3/<user>/<workdir>/infer_dataset.py \
  --dataset crc_val_7k \
  --data-dir /scratch/beagle3/<user>/<workdir>/data/CRC-VAL-HE-7K \
  --format file \
  --models hoptimus0 \
  --batch-size 32

echo ""
echo "=== NCT-CRC-HE-100K ==="
python /scratch/beagle3/<user>/<workdir>/infer_dataset.py \
  --dataset nct_crc_100k \
  --data-dir /scratch/beagle3/<user>/<workdir>/data/NCT-CRC-HE-100K \
  --format file \
  --models hoptimus0 \
  --batch-size 32

echo ""
echo "=== PCam ==="
python /scratch/beagle3/<user>/<workdir>/infer_dataset.py \
  --dataset pcam \
  --data-dir /scratch/beagle3/<user>/<workdir>/data \
  --format pcam \
  --models hoptimus0 \
  --batch-size 32
