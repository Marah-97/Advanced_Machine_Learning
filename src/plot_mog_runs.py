# plot_mog_runs.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads all 10 saved MoG VAE models and produces:
#
#   1. mog_combined_prior_posterior.png  — 2x5 grid, one subplot per run
#   2. mog_elbo_summary.png             — bar chart of test ELBO across runs
#   3. mog_digit_colored.png            — digit-colored posterior + prior contours
#                                         (averaged across all 10 runs)
#
# Usage:
#   python src/plot_mog_runs.py
#   python src/plot_mog_runs.py --model-dir models --n-runs 10 --device cuda

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from torchvision import datasets, transforms
import argparse
import os


# ─── Model classes ───────────────────────────────────────────────────────────

class MoGPrior(nn.Module):
    def __init__(self, M, K=10):
        super().__init__()
        self.M = M; self.K = K
        self.means    = nn.Parameter(torch.randn(K, M) * 0.5)
        self.log_stds = nn.Parameter(torch.zeros(K, M))
        self.logits   = nn.Parameter(torch.zeros(K))

    def forward(self):
        mix  = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(self.means, torch.exp(self.log_stds)), 1)
        return td.MixtureSameFamily(mix, comp)


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


class VAE_MoG(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior = prior; self.decoder = decoder; self.encoder = encoder
    def elbo(self, x):
        q = self.encoder(x); z = q.rsample()
        return torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z), dim=0
        )
    def forward(self, x):
        return -self.elbo(x)


def build_model(M, K, device):
    enc = nn.Sequential(nn.Flatten(), nn.Linear(784,512), nn.ReLU(),
                        nn.Linear(512,512), nn.ReLU(), nn.Linear(512, M*2))
    dec = nn.Sequential(nn.Linear(M,512), nn.ReLU(), nn.Linear(512,512),
                        nn.ReLU(), nn.Linear(512,784), nn.Unflatten(-1,(28,28)))
    return VAE_MoG(MoGPrior(M,K), BernoulliDecoder(dec), GaussianEncoder(enc)).to(device)


def binarize(x):
    return (x > 0.5).float().squeeze()


# ─── Data helpers ─────────────────────────────────────────────────────────────

def get_test_loader():
    transform = transforms.Compose([transforms.ToTensor(), binarize])
    return torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=256, shuffle=False, num_workers=0
    )


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


@torch.no_grad()
def get_posterior_means_with_labels(model, data_loader, device):
    model.eval()
    z_list, y_list = [], []
    for x, y in data_loader:
        x = x.to(device)
        z_list.append(model.encoder(x).mean.cpu())
        y_list.append(y)
    return torch.cat(z_list).numpy(), torch.cat(y_list).numpy()


# ─── Plot 1: 2x5 grid prior vs posterior ─────────────────────────────────────

def plot_combined_grid(models_data, n_runs, out_dir):
    n_cols = 5
    n_rows = (n_runs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.flatten()

    fig.suptitle(
        "MoG Prior vs Aggregate Posterior\n(2D PCA projection, all 10 runs)",
        fontsize=14, fontweight='bold', y=1.01
    )

    for run_idx, (z_post, model, device) in enumerate(models_data):
        pca      = PCA(n_components=2)
        post_2d  = pca.fit_transform(z_post)
        ev       = pca.explained_variance_ratio_.sum() * 100

        # Sample from MoG prior
        prior_raw = model.prior().sample(torch.Size([3000])).cpu().detach().numpy()
        prior_2d  = pca.transform(prior_raw)

        ax = axes[run_idx]
        ax.scatter(prior_2d[:,0], prior_2d[:,1], s=3, alpha=0.3,
                   color='steelblue', label='Prior p(z)', rasterized=True)
        ax.scatter(post_2d[:,0],  post_2d[:,1],  s=3, alpha=0.3,
                   color='tomato', label='Posterior q(z)', rasterized=True)
        ax.set_title(f"Run {run_idx+1}  ({ev:.1f}% var)", fontsize=10)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='datalim')

    for i in range(len(models_data), len(axes)):
        axes[i].set_visible(False)

    blue_patch = mpatches.Patch(color='steelblue', label='Prior p(z)')
    red_patch  = mpatches.Patch(color='tomato',    label='Aggregate posterior q(z)')
    fig.legend(handles=[blue_patch, red_patch], loc='lower center',
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = os.path.join(out_dir, 'mog_combined_prior_posterior.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out}")


# ─── Plot 2: ELBO bar chart ───────────────────────────────────────────────────

def plot_elbo_summary(elbo_values, out_dir):
    mean_e = np.mean(elbo_values)
    std_e  = np.std(elbo_values)

    fig, ax = plt.subplots(figsize=(10, 4))
    runs = list(range(1, len(elbo_values)+1))
    ax.bar(runs, elbo_values, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axhline(mean_e, color='tomato', linewidth=2, linestyle='--',
               label=f'Mean = {mean_e:.2f}')
    ax.fill_between([0.5, len(elbo_values)+0.5],
                    mean_e - std_e, mean_e + std_e,
                    color='tomato', alpha=0.15, label=f'±1 std = {std_e:.2f}')
    ax.set_xlabel("Run", fontsize=12)
    ax.set_ylabel("Test ELBO", fontsize=12)
    ax.set_title(f"Test ELBO across {len(elbo_values)} runs — MoG Prior\n"
                 f"Mean = {mean_e:.2f}  ±  {std_e:.2f}", fontsize=12)
    ax.set_xticks(runs)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, 'mog_elbo_summary.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out}")
    print(f"Test ELBO — Mean: {mean_e:.4f}  Std: {std_e:.4f}")


# ─── Plot 3: Digit-colored posterior + MoG prior contours ────────────────────

def plot_digit_colored(all_z, all_labels, model, device, pca, out_dir):
    COLORS = [
        '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
        '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#808000'
    ]

    # Average posterior means across runs
    z_mean  = np.mean(np.stack(all_z, axis=0), axis=0)
    labels  = all_labels
    z_2d    = pca.transform(z_mean)
    ev      = pca.explained_variance_ratio_.sum() * 100

    # MoG prior density via KDE on projected prior samples
    with torch.no_grad():
        prior_samples = model.prior().sample(torch.Size([10000])).cpu().numpy()
    prior_2d = pca.transform(prior_samples)

    kde     = gaussian_kde(prior_2d.T, bw_method=0.2)
    x_min, x_max = z_2d[:,0].min()-0.5, z_2d[:,0].max()+0.5
    y_min, y_max = z_2d[:,1].min()-0.5, z_2d[:,1].max()+0.5
    xx, yy  = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Prior contours
    ax.contour(xx, yy, density, levels=15, colors='steelblue',
               linewidths=0.8, alpha=0.8)

    # Digit-colored posterior
    for digit in range(10):
        mask = labels == digit
        ax.scatter(z_2d[mask,0], z_2d[mask,1], s=4, alpha=0.4,
                   color=COLORS[digit], label=str(digit), rasterized=True)

    ax.set_xlabel(f"PC1", fontsize=11) # ({ev/2:.1f}% var)
    ax.set_ylabel(f"PC2", fontsize=11) # ({ev/2:.1f}% var)
    # ax.set_title(
    #     f"MoG Prior — Aggregate Posterior\n"
    #     f"(averaged over runs, 2D PCA, {ev:.1f}% var explained)",
    #     fontsize=11, fontweight='bold'
    # )
    # ax.legend(title="Digit", loc='upper right', markerscale=3,
    #           fontsize=9, title_fontsize=10, framealpha=0.8, ncol=2)

    plt.tight_layout()
    out = os.path.join(out_dir, 'mog_digit_colored.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',   type=str, default='models')
    parser.add_argument('--out-dir',     type=str, default='results')
    parser.add_argument('--n-runs',      type=int, default=10)
    parser.add_argument('--latent-dim',  type=int, default=32)
    parser.add_argument('--n-components',type=int, default=10)
    parser.add_argument('--device',      type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_loader = get_test_loader()

    models_data  = []   # (z_post, model, device) per run
    all_z        = []   # posterior means per run (for digit plot)
    all_labels   = None
    elbo_values  = []
    last_model   = None

    for run in range(1, args.n_runs + 1):
        model_path = os.path.join(args.model_dir, f'vae_mog_run{run}.pt')
        elbo_path  = os.path.join(args.model_dir, f'vae_mog_run{run}_elbo.txt')

        if not os.path.exists(model_path):
            print(f"  [!] Missing: {model_path} — skipping run {run}")
            continue

        print(f"Loading run {run} ...")
        model = build_model(args.latent_dim, args.n_components, args.device)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model.eval()

        z_post = get_posterior_means(model, test_loader, args.device, max_samples=3000)
        models_data.append((z_post, model, args.device))

        z_all, y_all = get_posterior_means_with_labels(model, test_loader, args.device)
        all_z.append(z_all)
        if all_labels is None:
            all_labels = y_all

        last_model = model

        if os.path.exists(elbo_path):
            with open(elbo_path) as f:
                elbo_values.append(float(f.read().strip()))

    if not models_data:
        print("No models found! Check --model-dir path.")
        exit(1)

    print(f"\nLoaded {len(models_data)} runs. Generating plots...")

    # Fit PCA once on averaged posterior for digit plot
    z_mean_all = np.mean(np.stack(all_z, axis=0), axis=0)
    pca = PCA(n_components=2)
    pca.fit(z_mean_all)

    plot_combined_grid(models_data, args.n_runs, args.out_dir)

    if elbo_values:
        plot_elbo_summary(elbo_values, args.out_dir)

    plot_digit_colored(all_z, all_labels, last_model, args.device, pca, args.out_dir)

    print("\nAll MoG plots done!")
