# VAE with Mixture of Gaussians (MoG) Prior - Part A
# DTU 02460 Advanced Machine Learning Mini-Project 1
#
# Usage:
#   Single run:   python src/vae_mog_prior.py --run 1 --device cuda
#   Linux loop:   for i in $(seq 1 10); do python src/vae_mog_prior.py --run $i --device cuda; done

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from torchvision import datasets, transforms
import argparse
import os


# ─── MoG Prior ───────────────────────────────────────────────────────────────

class MoGPrior(nn.Module):
    """
    Mixture of K Gaussians prior. All parameters are learned during training.
    p(z) = sum_k pi_k * N(z | mu_k, diag(sigma_k^2))
    """
    def __init__(self, M, K=10):
        super().__init__()
        self.means    = nn.Parameter(torch.randn(K, M) * 0.5)  # (K, M)
        self.log_stds = nn.Parameter(torch.zeros(K, M))         # (K, M)
        self.logits   = nn.Parameter(torch.zeros(K))            # mixture weights

    def forward(self):
        mix  = td.Categorical(logits=self.logits)
        comp = td.Independent(td.Normal(self.means, torch.exp(self.log_stds)), 1)
        return td.MixtureSameFamily(mix, comp)


# ─── Encoder ─────────────────────────────────────────────────────────────────

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


# ─── Decoder ─────────────────────────────────────────────────────────────────

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        return td.Independent(td.Bernoulli(logits=self.decoder_net(z)), 2)


# ─── VAE ─────────────────────────────────────────────────────────────────────

class VAE_MoG(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior   = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        KL divergence has no closed form for MoG prior, so we use
        a Monte Carlo estimate of the ELBO:
            ELBO = E_q[log p(x|z)] - E_q[log q(z|x)] + E_q[log p(z)]
        """
        q = self.encoder(x)
        z = q.rsample()
        return torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z), dim=0
        )

    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        return -self.elbo(x)


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


# ─── Named binarize (Windows-safe) ───────────────────────────────────────────

def binarize(x):
    return (x > 0.5).float().squeeze()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',          type=int,   default=1)
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch-size',   type=int,   default=128)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--latent-dim',   type=int,   default=32)
    parser.add_argument('--n-components', type=int,   default=10)
    parser.add_argument('--device',       type=str,   default='cpu')
    parser.add_argument('--save-dir',     type=str,   default='models')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    M          = args.latent_dim
    K          = args.n_components
    MODEL_PATH = os.path.join(args.save_dir, f'vae_mog_run{args.run}.pt')
    ELBO_PATH  = os.path.join(args.save_dir, f'vae_mog_run{args.run}_elbo.txt')

    print(f"\n{'='*55}")
    print(f"  MoG Prior  |  Run {args.run}  |  Device: {args.device}")
    print(f"  Epochs={args.epochs}  Batch={args.batch_size}  LR={args.lr}  M={M}  K={K}")
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

    model = VAE_MoG(
        prior   = MoGPrior(M, K),
        decoder = BernoulliDecoder(decoder_net),
        encoder = GaussianEncoder(encoder_net),
    ).to(args.device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, train_loader, args.epochs, args.device)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    test_elbo = evaluate_elbo(model, test_loader, args.device)
    print(f"Test ELBO (run {args.run}): {test_elbo:.4f}")

    with open(ELBO_PATH, 'w') as f:
        f.write(str(test_elbo))