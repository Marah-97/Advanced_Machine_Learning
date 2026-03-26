# Mini-project 2 — Advanced Machine Learning (02460)
## Part A: Pull-back Geodesics

This project trains a Variational Autoencoder (VAE) on a subset of MNIST and computes
geodesics in the latent space under the pull-back metric induced by the decoder.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to run

### 1. Train the VAE
Trains a VAE with a 2D latent space on 2048 MNIST images (digits 0, 1, 2) for 60 epochs.
Results are saved to the experiment folder.

```bash
python vae_geo.py train \
    --experiment-folder exp_run1 \
    --epochs-per-decoder 60 \
    --device cuda
```

---

### 2. Compute geodesics
Loads the trained model, encodes the test set, picks 25 random point pairs,
computes geodesics by minimizing the discrete curve energy, and saves the results.

```bash
python vae_geo.py geodesics \
    --experiment-folder exp_run1 \
    --num-curves 25 \
    --num-t 20 \
    --geodesic-steps 1000 \
    --geodesic-lr 5e-3 \
    --device cuda
```

This saves:
- `exp_run1/geodesics_partA.png` — the latent space plot with geodesics
- `exp_run1/geodesics_data.pt`  — the computed geodesics and latent points for replotting

---

### 3. Replot without recomputing
Edit plotting options (e.g. title, endpoints) and replot instantly from saved data.

```bash
python vae_geo.py plot \
    --experiment-folder exp_run1 \
    --experiment-folder-2 exp_run2 \
    --device cuda
```

This generates a two-panel figure (`geodesics_two_panel.png`) comparing two training runs
side by side — used for the report Figure 1.

---

### 4. Evaluate ELBO on test set
```bash
python vae_geo.py eval \
    --experiment-folder exp_run1 \
    --device cuda
```

---

### 5. Sample from the model
```bash
python vae_geo.py sample \
    --experiment-folder exp_run1 \
    --samples samples.png \
    --device cuda
```

---

## Reproducing the report results (Figure 1)

To reproduce the two-panel figure used in the report, train and compute geodesics
for two independent runs using the same random seed for point pair selection:

```bash
# Run 1
python vae_geo.py train --experiment-folder exp_run1 --epochs-per-decoder 60 --device cuda
python vae_geo.py geodesics --experiment-folder exp_run1 --num-curves 25 --num-t 20 --geodesic-steps 1000 --geodesic-lr 5e-3 --device cuda

# Run 2
python vae_geo.py train --experiment-folder exp_run2 --epochs-per-decoder 60 --device cuda
python vae_geo.py geodesics --experiment-folder exp_run2 --num-curves 25 --num-t 20 --geodesic-steps 1000 --geodesic-lr 5e-3 --device cuda

# Generate two-panel plot
python vae_geo.py plot --experiment-folder exp_run1 --experiment-folder-2 exp_run2 --device cuda
```

---

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment-folder` | `experiment` | Folder to save/load model and results |
| `--epochs-per-decoder` | `50` | Number of training epochs |
| `--latent-dim` | `2` | Latent space dimension |
| `--batch-size` | `32` | Training batch size |
| `--num-curves` | `25` | Number of geodesics to compute |
| `--num-t` | `20` | Number of points along each curve |
| `--geodesic-steps` | `500` | Optimisation steps for geodesic solver |
| `--geodesic-lr` | `0.01` | Learning rate for geodesic solver |
| `--device` | `cpu` | Device: `cpu`, `cuda`, or `mps` |


