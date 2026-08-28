#!/bin/bash
#SBATCH --job-name=dinov2_eval
#SBATCH --account=<your-account>
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/beagle3/<user>/<workdir>/logs/eval_%j.out
#SBATCH --error=/scratch/beagle3/<user>/<workdir>/logs/eval_%j.err

set -eo pipefail
mkdir -p /scratch/beagle3/<user>/<workdir>/logs

source /software/python-miniforge-25.3.0-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/beagle3/<user>/dinov2_env

echo "=== t-SNE ==="
python /scratch/beagle3/<user>/<workdir>/visualize_tsne.py

echo ""
echo "=== KNN + Linear Probe ==="
python /scratch/beagle3/<user>/<workdir>/eval_knn.py
