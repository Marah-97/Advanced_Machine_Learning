# plot_digit_colored.py
# ─────────────────────────────────────────────────────────────────────────────
# digit-colored aggregate posterior
# scattered on top of prior density contours, in 2D PCA space.
#
# Averages posterior means across all 10 runs for a cleaner visualization.
#
# Usage:
#   python src/plot_digit_colored.py
#   python src/plot_digit_colored.py --model-dir models --n-runs 10 --device cuda

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from torchvision import datasets, transforms
import argparse
import os


# ─── Model classes ───────────────────────────────────────────────────────────

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


def binarize(x):
    return (x > 0.5).float().squeeze()


# ─── Collect posterior means + labels for one model ──────────────────────────

@torch.no_grad()
def get_posterior_means_with_labels(model, data_loader, device):
    model.eval()
    z_list, y_list = [], []
    for x, y in data_loader:
        x = x.to(device)
        z_list.append(model.encoder(x).mean.cpu())
        y_list.append(y)
    return torch.cat(z_list, dim=0).numpy(), torch.cat(y_list, dim=0).numpy()


# ─── Main ─────────────────────────────────────────────────────────────────────

def plot_digit_colored(model_dir, n_runs, M, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    transform   = transforms.Compose([transforms.ToTensor(), binarize])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=256, shuffle=False, num_workers=0
    )

    # 10 distinct colors for digits 0-9 (matches the style in the example)
    COLORS = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#808000'
    ]
    DIGIT_NAMES = [str(i) for i in range(10)]

    # ── Collect posterior means from all runs, average them ──────────────────
    print("Loading models and collecting posterior means...")
    all_z   = []   # list of (N, M) arrays, one per run
    labels  = None

    for run in range(1, n_runs + 1):
        model_path = os.path.join(model_dir, f'vae_gaussian_run{run}.pt')
        if not os.path.exists(model_path):
            print(f"  [!] Missing run {run}, skipping")
            continue
        model = build_model(M, device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        z, y = get_posterior_means_with_labels(model, test_loader, device)
        all_z.append(z)
        if labels is None:
            labels = y
        print(f"  Loaded run {run}")

    # Average posterior means across runs (same label order guaranteed by shuffle=False)
    z_mean = np.mean(np.stack(all_z, axis=0), axis=0)  # (N, M)

    # ── PCA: fit on averaged posterior means ─────────────────────────────────
    pca    = PCA(n_components=2)
    z_2d   = pca.fit_transform(z_mean)
    ev     = pca.explained_variance_ratio_.sum() * 100

    # ── Prior density via KDE on prior samples projected into same PCA space ─
    # Sample many prior points, project, estimate density
    model = build_model(M, device)  # just need prior, weights don't matter
    prior_samples = td.Independent(
        td.Normal(torch.zeros(M), torch.ones(M)), 1
    ).sample(torch.Size([10000])).numpy()
    prior_2d = pca.transform(prior_samples)

    # KDE on prior projected samples
    kde      = gaussian_kde(prior_2d.T, bw_method=0.15)
    x_min, x_max = z_2d[:, 0].min() - 0.5, z_2d[:, 0].max() + 0.5
    y_min, y_max = z_2d[:, 1].min() - 0.5, z_2d[:, 1].max() + 0.5
    xx, yy   = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])
    density  = kde(grid_pts).reshape(xx.shape)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    # Prior contours (blue, like in the example)
    ax.contour(xx, yy, density, levels=15, colors='steelblue',
               linewidths=0.8, alpha=0.8)

    # Aggregate posterior — one color per digit
    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            s=4, alpha=0.4, color=COLORS[digit],
            label=str(digit), rasterized=True
        )

    ax.set_xlabel(f"PC1", fontsize=11) # ax.set_xlabel(f"PC1 ({ev/2:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2", fontsize=11) # ax.set_ylabel(f"PC2 ({ev/2:.1f}% var)", fontsize=11)
    # ax.set_title(
    #     f"Standard Gaussian Prior — Aggregate Posterior\n"
    #     f"(averaged over {len(all_z)} runs, 2D PCA, {ev:.1f}% var explained)",
    #     fontsize=11, fontweight='bold'
    # )

    # Legend for digits
    # legend = ax.legend(
    #     title="Digit", loc='upper right', markerscale=3,
    #     fontsize=9, title_fontsize=10,
    #     framealpha=0.8, ncol=2
    # )

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'prior_posterior_digit_colored.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved -> {out_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',  type=str, default='models')
    parser.add_argument('--out-dir',    type=str, default='results')
    parser.add_argument('--n-runs',     type=int, default=10)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--device',     type=str, default='cpu')
    args = parser.parse_args()

    plot_digit_colored(
        model_dir=args.model_dir,
        n_runs=args.n_runs,
        M=args.latent_dim,
        device=args.device,
        out_dir=args.out_dir,
    )