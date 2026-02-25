# plot_all_runs.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads all 10 saved VAE models (vae_gaussian_run1.pt ... run10.pt) and
# produces two figures:
#
#   1. combined_prior_posterior.png
#      A 2x5 grid — one subplot per run — each showing prior (blue) vs
#      aggregate posterior (red) in 2D PCA space.
#
#   2. elbo_summary.png
#      Bar chart of test ELBO across runs + mean ± std annotation.
#
# Usage:
#   python plot_all_runs.py
#   python plot_all_runs.py --model-dir models --n-runs 10 --device cpu

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import argparse
import os


# ─── Copy model classes here (needed to load state dicts) ────────────────────

class GaussianPrior(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(M), requires_grad=False)
        self.std  = nn.Parameter(torch.ones(M),  requires_grad=False)
    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net
    def forward(self, x):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net
    def forward(self, z):
        return td.Independent(td.Bernoulli(logits=self.decoder_net(z)), 2)

class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior = prior; self.decoder = decoder; self.encoder = encoder
    def forward(self, x):
        return -self.elbo(x)
    def elbo(self, x):
        q = self.encoder(x); z = q.rsample()
        return torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)


def build_model(M, device):
    enc = nn.Sequential(nn.Flatten(), nn.Linear(784,512), nn.ReLU(),
                        nn.Linear(512,512), nn.ReLU(), nn.Linear(512, M*2))
    dec = nn.Sequential(nn.Linear(M,512), nn.ReLU(), nn.Linear(512,512),
                        nn.ReLU(), nn.Linear(512,784), nn.Unflatten(-1,(28,28)))
    return VAE(GaussianPrior(M), BernoulliDecoder(dec), GaussianEncoder(enc)).to(device)


# ─── Named binarize (Windows-safe) ───────────────────────────────────────────

def binarize(x):
    return (x > 0.5).float().squeeze()


# ─── Collect posterior means for one model ───────────────────────────────────

@torch.no_grad()
def get_posterior_means(model, data_loader, device, max_samples=3000):
    model.eval()
    z_list = []
    for x, _ in data_loader:
        x = x.to(device)
        z_list.append(model.encoder(x).mean.cpu())
        if sum(z.shape[0] for z in z_list) >= max_samples:
            break
    return torch.cat(z_list, dim=0)[:max_samples].numpy()


# ─── Main plotting function ───────────────────────────────────────────────────

def plot_all_runs(model_dir, n_runs, M, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    transform   = transforms.Compose([transforms.ToTensor(), binarize])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=256, shuffle=False, num_workers=0
    )

    # ── Figure 1: 2×5 grid of prior vs posterior per run ─────────────────────
    n_cols = 5
    n_rows = (n_runs + n_cols - 1) // n_cols   # = 2 for 10 runs
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    fig.suptitle(
        "Standard Gaussian Prior vs Aggregate Posterior\n(2D PCA projection, all 10 runs)",
        fontsize=14, fontweight='bold', y=1.01
    )

    elbo_values = []

    for run in range(1, n_runs + 1):
        model_path = os.path.join(model_dir, f'vae_gaussian_run{run}.pt')
        elbo_path  = os.path.join(model_dir, f'vae_gaussian_run{run}_elbo.txt')

        if not os.path.exists(model_path):
            print(f"  [!] Missing: {model_path} — skipping run {run}")
            axes[run-1].set_visible(False)
            continue

        print(f"Loading run {run} ...")
        model = build_model(M, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Aggregate posterior means
        z_post = get_posterior_means(model, test_loader, device, max_samples=3000)

        # Prior samples (fit PCA on posterior, project both)
        pca      = PCA(n_components=2)
        post_2d  = pca.fit_transform(z_post)
        ev       = pca.explained_variance_ratio_.sum() * 100

        prior_raw = model.prior().sample(torch.Size([3000])).cpu().numpy()
        prior_2d  = pca.transform(prior_raw)

        # Plot
        ax = axes[run - 1]
        ax.scatter(prior_2d[:, 0], prior_2d[:, 1], s=3, alpha=0.3,
                   color='steelblue', label='Prior p(z)', rasterized=True)
        ax.scatter(post_2d[:, 0],  post_2d[:, 1],  s=3, alpha=0.3,
                   color='tomato',    label='Posterior q(z)', rasterized=True)
        ax.set_title(f"Run {run}  ({ev:.1f}% var)", fontsize=10)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='datalim')

        # Read ELBO
        if os.path.exists(elbo_path):
            with open(elbo_path) as f:
                elbo_values.append(float(f.read().strip()))

    # Shared legend
    blue_patch = mpatches.Patch(color='steelblue', label='Prior p(z)')
    red_patch  = mpatches.Patch(color='tomato',    label='Aggregate posterior q(z)')
    fig.legend(handles=[blue_patch, red_patch], loc='lower center',
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    # Hide any unused subplots
    for i in range(n_runs, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    out1 = os.path.join(out_dir, 'combined_prior_posterior.png')
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved -> {out1}")

    # ── Figure 2: ELBO bar chart ──────────────────────────────────────────────
    if elbo_values:
        mean_elbo = np.mean(elbo_values)
        std_elbo  = np.std(elbo_values)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        runs = list(range(1, len(elbo_values) + 1))
        bars = ax2.bar(runs, elbo_values, color='steelblue', edgecolor='white', alpha=0.85)
        ax2.axhline(mean_elbo, color='tomato', linewidth=2, linestyle='--',
                    label=f'Mean = {mean_elbo:.2f}')
        ax2.fill_between(
            [0.5, len(elbo_values) + 0.5],
            mean_elbo - std_elbo, mean_elbo + std_elbo,
            color='tomato', alpha=0.15, label=f'±1 std = {std_elbo:.2f}'
        )
        ax2.set_xlabel("Run", fontsize=12)
        ax2.set_ylabel("Test ELBO", fontsize=12)
        ax2.set_title(f"Test ELBO across {len(elbo_values)} runs — Standard Gaussian Prior\n"
                      f"Mean = {mean_elbo:.2f}  ±  {std_elbo:.2f}", fontsize=12)
        ax2.set_xticks(runs)
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        out2 = os.path.join(out_dir, 'elbo_summary.png')
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved -> {out2}")
        print(f"\nTest ELBO — Mean: {mean_elbo:.4f}  Std: {std_elbo:.4f}")
    else:
        print("No ELBO files found — skipping ELBO plot.")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='models',  help='Folder with .pt and .txt files')
    parser.add_argument('--out-dir',   type=str, default='results', help='Folder to save plots')
    parser.add_argument('--n-runs',    type=int, default=10)
    parser.add_argument('--latent-dim',type=int, default=32)
    parser.add_argument('--device',    type=str, default='cpu')
    args = parser.parse_args()

    plot_all_runs(
        model_dir=args.model_dir,
        n_runs=args.n_runs,
        M=args.latent_dim,
        device=args.device,
        out_dir=args.out_dir,
    )
