#!/bin/bash
#SBATCH --job-name=dinov2_crc7k
#SBATCH --account=<your-account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/beagle3/<user>/<workdir>/logs/crc7k_%j.out
#SBATCH --error=/scratch/beagle3/<user>/<workdir>/logs/crc7k_%j.err

set -eo pipefail
mkdir -p /scratch/beagle3/<user>/<workdir>/logs
module load cuda/11.7
source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/beagle3/<user>/dinov2_env

python /scratch/beagle3/<user>/<workdir>/infer_dataset.py \
  --dataset crc_val_7k \
  --data-dir /scratch/beagle3/<user>/<workdir>/data/CRC-VAL-HE-7K \
  --format file \
  --models vits14 vitb14 vitl14 \
  --batch-size 64
