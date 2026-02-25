#!/bin/bash
#BSUB -J vae_mog
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 2:00
#BSUB -o logs/vae_mog_%J.out
#BSUB -e logs/vae_mog_%J.err

# mkdir -p logs
# mkdir -p models

# Load modules (adjust if your HPC uses different module names)
module load cuda/11.8
module load python3/3.9.11

# Activate your conda/venv environment if you have one
source .venv/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "GPU info:"
nvidia-smi

cd /zhome/47/6/135800/Desktop/AML/MiniProject_1

for i in 1 2 3 4 5 6 7 8 9 10
do
    echo "==============================="
    echo "Starting MoG run $i at $(date)"
    echo "==============================="
    python src/vae_mog_prior.py --run $i --device cuda
done

echo "All MoG runs finished at $(date)"
