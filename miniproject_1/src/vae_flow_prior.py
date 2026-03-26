# VAE with Flow-based Prior - Part A
# DTU 02460 Advanced Machine Learning Mini-Project 1
#
# Usage:
#   Single run:   python src/vae_flow_prior.py --run 1 --device cuda
#   Linux loop:   for i in $(seq 1 10); do python src/vae_flow_prior.py --run $i --device cuda; done
#
# Architecture summary:
#   Encoder:    Flatten -> Linear(784,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512, 2*M)
#   Decoder:    Linear(M,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512,784) -> Unflatten(28,28)
#   Prior:      RealNVP normalizing flow (n_flows coupling layers) transforming N(0,I)
#               Each coupling layer: scale and translate networks (Linear(M,256)->ReLU->Linear(256,M/2))
#
# Training config:
#   Epochs: 20 | Batch size: 128 | LR: 1e-3 | Latent dim: 32 | Flow layers: 8

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from torchvision import datasets, transforms
import argparse
import os


# ─── RealNVP Coupling Layer ───────────────────────────────────────────────────

class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP.
    Splits z into two halves [z1, z2]:
      - z1 passes through unchanged
      - z2 is transformed: z2' = z2 * exp(s(z1)) + t(z1)
    The log determinant of the Jacobian is sum(s(z1)).
    """
    def __init__(self, M, hidden_dim=256):
        super().__init__()
        half = M // 2
        rest = M - half

        # Scale and translate networks — take first half, output second half
        self.scale_net = nn.Sequential(
            nn.Linear(half, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, rest), nn.Tanh()   # Tanh keeps scale bounded
        )
        self.translate_net = nn.Sequential(
            nn.Linear(half, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, rest)
        )
        self.half = half

    def forward(self, z):
        """Forward pass: z -> z' (used for sampling from prior)"""
        z1, z2 = z[:, :self.half], z[:, self.half:]
        s = self.scale_net(z1)
        t = self.translate_net(z1)
        z2_prime = z2 * torch.exp(s) + t
        return torch.cat([z1, z2_prime], dim=1)

    def inverse(self, z_prime):
        """Inverse pass: z' -> z (used for computing log prob)"""
        z1, z2_prime = z_prime[:, :self.half], z_prime[:, self.half:]
        s = self.scale_net(z1)
        t = self.translate_net(z1)
        z2 = (z2_prime - t) * torch.exp(-s)
        return torch.cat([z1, z2], dim=1), -s.sum(dim=1)  # return z and log_det


# ─── RealNVP Flow Prior ───────────────────────────────────────────────────────

class FlowPrior(nn.Module):
    """
    Normalizing flow prior using stacked RealNVP coupling layers.
    The base distribution is N(0, I). The flow transforms samples
    from the base into a more expressive distribution p(z).

    To alternate which dimensions are transformed, we permute
    (reverse) the dimensions between coupling layers.
    """
    def __init__(self, M, n_flows=8, hidden_dim=256):
        super().__init__()
        self.M       = M
        self.n_flows = n_flows

        self.flows = nn.ModuleList([CouplingLayer(M, hidden_dim) for _ in range(n_flows)])

        # Base distribution: standard Gaussian (not a parameter, just used for log_prob)
        self.register_buffer('base_mean', torch.zeros(M))
        self.register_buffer('base_std',  torch.ones(M))

    @property
    def base_dist(self):
        return td.Independent(td.Normal(self.base_mean, self.base_std), 1)

    def forward(self):
        """Return self so we can call .log_prob() and .sample() on the prior."""
        return self

    def log_prob(self, z):
        """
        Compute log p(z) using the change of variables formula:
            log p(z) = log p_base(f^{-1}(z)) + sum of log|det J_f^{-1}|
        We apply coupling layers in reverse order for the inverse pass.
        """
        log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in reversed(self.flows):
            z, ld = flow.inverse(z)
            log_det += ld
        return self.base_dist.log_prob(z) + log_det

    def sample(self, sample_shape=torch.Size([])):
        """
        Sample from the flow prior by sampling from base and
        applying the forward transformation.
        """
        with torch.no_grad():
            z = self.base_dist.sample(sample_shape)
            for flow in self.flows:
                z = flow.forward(z)
        return z


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


# ─── VAE with Flow Prior ──────────────────────────────────────────────────────

class VAE_Flow(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior   = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        KL has no closed form for flow prior — use MC estimate:
            ELBO = E_q[log p(x|z)] - E_q[log q(z|x)] + E_q[log p(z)]
        where log p(z) is computed via the flow's change-of-variables formula.
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
    parser.add_argument('--run',        type=int,   default=1)
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch-size', type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int,   default=32)
    parser.add_argument('--n-flows',    type=int,   default=8,    help='Number of coupling layers')
    parser.add_argument('--device',     type=str,   default='cpu')
    parser.add_argument('--save-dir',   type=str,   default='models')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    M          = args.latent_dim
    MODEL_PATH = os.path.join(args.save_dir, f'vae_flow_run{args.run}.pt')
    ELBO_PATH  = os.path.join(args.save_dir, f'vae_flow_run{args.run}_elbo.txt')

    print(f"\n{'='*55}")
    print(f"  Flow Prior  |  Run {args.run}  |  Device: {args.device}")
    print(f"  Epochs={args.epochs}  Batch={args.batch_size}  LR={args.lr}")
    print(f"  Latent dim={M}  Flow layers={args.n_flows}")
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

    model = VAE_Flow(
        prior   = FlowPrior(M, n_flows=args.n_flows),
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
