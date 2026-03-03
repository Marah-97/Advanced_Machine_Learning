  #!/bin/bash
  # Simple utility to generate a single DDPM sample from a saved model.
  # Usage:
  #   ./run_ddpm_test.sh            # use defaults
  #   ./run_ddpm_test.sh model.pt out.png

  MODEL=${1:-models/ddpm_unet_fast.pt}
  BASE_DIR="ddpm_samples"
  BASE_NAME="ddpm_test_sample"
  EXT="png"

  mkdir -p "$BASE_DIR"

  i=1
  while [ -f "$BASE_DIR/${BASE_NAME}_$i.$EXT" ]; do
    i=$((i+1))
  done

  OUT="$BASE_DIR/${BASE_NAME}_$i.$EXT"

  ROOT="$(dirname "$0")"

  python3 "$ROOT/src/ddpm.py" sample \
    --data mnist \
    --arch unet \
    --timesteps 200 \
    --device cpu \
    --model "$MODEL" \
    --samples "$OUT" \
    --n-samples 1

  echo "Wrote one sample image to $OUT"