#!/bin/bash
#SBATCH --job-name=dinov2_infer
#SBATCH --account=<your-account>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/beagle3/<user>/<workdir>/logs/infer_%j.out
#SBATCH --error=/scratch/beagle3/<user>/<workdir>/logs/infer_%j.err

set -eo pipefail

mkdir -p /scratch/beagle3/<user>/<workdir>/logs

module load cuda/11.7

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/beagle3/<user>/dinov2_env

python /scratch/beagle3/<user>/<workdir>/infer_basic.py
