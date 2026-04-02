#!/bin/bash
#BSUB -J ddpm_mnist
#BSUB -q gpua100
#BSUB -n 4
#BSUB -W 04:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:aff=no:mode=exclusive_process"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail
mkdir -p logs

# Start clean (optional but recommended on HPC)
module purge

# Load a Python module (choose the one you used to CREATE the venv)
module load python3/3.11.13

# (Optional) load CUDA if your site requires it for runtime linkage / you installed CUDA wheels
# module load cuda/12.4.1

# Activate your venv (path must be visible on compute nodes)
source /path/to/your/project/venv_1/bin/activate

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('torch:', torch.__version__)"
python -u vae_geo.py calculate_CoV --device cuda