#!/bin/bash
# General-purpose DDPM runner.
# Usage:
#   ./run_ddpm.sh train [options]
#   ./run_ddpm.sh sample [options]
#
# Example:
#   ./run_ddpm.sh train --epochs 50 --device cuda
#   ./run_ddpm.sh sample --model models/ddpm.pt --n-samples 16

set -e

ROOT="$(dirname "$0")"
MODE=$1
shift

if [[ "$MODE" != "train" && "$MODE" != "sample" && "$MODE" != "test" ]]; then
  echo "Usage:"
  echo "  ./run_ddpm.sh train [ddpm options]"
  echo "  ./run_ddpm.sh sample [ddpm options]"
  echo "  ./run_ddpm.sh test [ddpm options]"
  exit 1
fi

python3 "$ROOT/src/ddpm.py" "$MODE" "$@"
™¡™