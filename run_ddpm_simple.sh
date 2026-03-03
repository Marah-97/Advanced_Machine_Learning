#!/bin/bash

ROOT="$(dirname "$0")"
MODEL=models/ddpm_unet_fast.pt

python3 "$ROOT/src/ddpm.py" train \
  --data mnist \
  --arch unet \
  --timesteps 200 \
  --device cpu \
  --epochs 5 \
  --batch-size 128 \
  --lr 3e-4 \
  --model "$MODEL"
