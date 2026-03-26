# VAE with Standard Gaussian Prior - Part A
# DTU 02460 Advanced Machine Learning Mini-Project 1
#
# Usage:
#   Single run:    python vae_gaussian_prior.py --run 1
#   Windows loop:  for %i in (1 2 3 4 5 6 7 8 9 10) do python vae_gaussian_prior.py --run %i
# bash :
# for i in 1 2 3 4 5 6 7 8 9 10
# do
#     echo "Starting run $i"
#     python vae_gaussian_prior.py --run $i --device cuda
# done
#
# Architecture summary:
#   Encoder: Flatten -> Linear(784,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512, 2*M)
#   Decoder: Linear(M,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512,784) -> Unflatten(28,28)
#   Prior:   Standard Gaussian N(0, I), M=32
#
# Training config:
#   Epochs: 20 | Batch size: 128 | LR: 1e-3 | Latent dim: 32

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from torchvision import datasets, transforms
import argparse
import os


# ─── Model Definitions ───────────────────────────────────────────────────────

class GaussianPrior(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std  = nn.Parameter(torch.ones(self.M),  requires_grad=False)

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
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior   = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        q    = self.encoder(x)
        z    = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0
        )
        return elbo

    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        return -self.elbo(x)


# ─── Build model ─────────────────────────────────────────────────────────────

def build_model(M, device):
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M * 2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),   nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28)),
    )
    return VAE(GaussianPrior(M), BernoulliDecoder(decoder_net), GaussianEncoder(encoder_net)).to(device)


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    progress_bar = tqdm(range(len(data_loader) * epochs), desc="Training")
    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_elbo(model, data_loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, _ in data_loader:
        x = x.to(device)
        total += model.elbo(x).item() * x.size(0)
        n     += x.size(0)
    return total / n


# ─── Named binarize fn (lambda can't be pickled on Windows) ──────────────────

def binarize(x):
    return (x > 0.5).float().squeeze()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',        type=int,   default=1,       help='Run index (e.g. 1-10)')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int,   default=32)
    parser.add_argument('--device',     type=str,   default='cpu')
    parser.add_argument('--save-dir',   type=str,   default='models', help='Folder to save .pt and .txt files')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    DEVICE     = args.device
    M          = args.latent_dim
    MODEL_PATH = os.path.join(args.save_dir, f'vae_gaussian_run{args.run}.pt')
    ELBO_PATH  = os.path.join(args.save_dir, f'vae_gaussian_run{args.run}_elbo.txt')

    print(f"\n{'='*55}")
    print(f"  Run {args.run}  |  Device: {DEVICE}  |  Latent dim: {M}")
    print(f"  Epochs={args.epochs}  Batch={args.batch_size}  LR={args.lr}")
    print(f"  Saving to: {MODEL_PATH}")
    print(f"{'='*55}\n")

    transform    = transforms.Compose([transforms.ToTensor(), binarize])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=True,  download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader  = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model     = build_model(M, DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, train_loader, args.epochs, DEVICE)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

    test_elbo = evaluate_elbo(model, test_loader, DEVICE)
    print(f"Test ELBO (run {args.run}): {test_elbo:.4f}")

    with open(ELBO_PATH, 'w') as f:
        f.write(str(test_elbo))
    print(f"ELBO saved -> {ELBO_PATH}")
