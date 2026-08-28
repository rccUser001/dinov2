#!/bin/bash
#SBATCH --job-name=dinov2_nct100k
#SBATCH --account=<your-account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/beagle3/<user>/<workdir>/logs/nct100k_%j.out
#SBATCH --error=/scratch/beagle3/<user>/<workdir>/logs/nct100k_%j.err

set -eo pipefail
mkdir -p /scratch/beagle3/<user>/<workdir>/logs
module load cuda/11.7
source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/beagle3/<user>/dinov2_env

python /scratch/beagle3/<user>/<workdir>/infer_dataset.py \
  --dataset nct_crc_100k \
  --data-dir /scratch/beagle3/<user>/<workdir>/data/NCT-CRC-HE-100K \
  --format file \
  --models vits14 vitb14 vitl14 \
  --batch-size 64
