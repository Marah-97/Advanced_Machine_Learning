# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024
# Part A: Pull-back geodesics implemented by [your group]

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.func import jacrev, vmap
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model definitions (unchanged from handout)
# ---------------------------------------------------------------------------

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

        Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters:
        decoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M)`
           as input and outputs a tensor of dimension (batch_size, 1, 28, 28).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        return -self.elbo(x)


# ---------------------------------------------------------------------------
# Training loop (unchanged from handout)
# ---------------------------------------------------------------------------

def train(model, optimizer, data_loader, epochs, device):
    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at epoch {epoch}, loss={loss:.1f}"
                )
                break


# ---------------------------------------------------------------------------
# Part A: Pull-back metric geodesics
# ---------------------------------------------------------------------------

def curve_energy(curve_points, decoder_net):
    """
    Compute the discrete energy of a curve in latent space under the
    pull-back metric induced by the decoder mean function f: Z -> X.

    The pull-back metric tensor at z is  G(z) = J(z)^T J(z),
    where J(z) = df/dz is the Jacobian of the decoder mean.

    The discrete energy approximation is:
        E(c) ≈ sum_{i=0}^{N-2} (c_{i+1} - c_i)^T G(c_i) (c_{i+1} - c_i)
             = sum_{i=0}^{N-2} || J(c_i) (c_{i+1} - c_i) ||^2

    Parameters:
    curve_points: [torch.Tensor]  shape (N, M)
        The N points along the curve, including fixed endpoints.
    decoder_net: [torch.nn.Module]
        The decoder network f: R^M -> R^(1 x 28 x 28).

    Returns:
    energy: [torch.Tensor]  scalar
        The total discrete curve energy.
    """
    N = curve_points.shape[0]

    # Compute Jacobians at all interior + start points: shape (N-1, D, M)
    # where D = 1*28*28 = 784
    def f(z):
        # z: (M,) -> output: (D,)  — needed for jacrev (single input)
        return decoder_net(z.unsqueeze(0)).squeeze(0).flatten()

    # vmap over the first N-1 curve points (segment starts)
    J = vmap(jacrev(f))(curve_points[:-1])   # (N-1, D, M)

    # Finite differences along the curve
    dz = curve_points[1:] - curve_points[:-1]  # (N-1, M)

    # || J_i @ dz_i ||^2  for each segment
    # J_i @ dz_i  has shape (D,), so we batch-matmul: (N-1, D, M) x (N-1, M, 1)
    Jdz = torch.bmm(J, dz.unsqueeze(-1)).squeeze(-1)   # (N-1, D)
    segment_energies = (Jdz ** 2).sum(dim=-1)           # (N-1,)

    return segment_energies.sum()


def compute_geodesic(z_start, z_end, decoder_net, n_points=20,
                     lr=1e-2, n_steps=500, device="cpu"):
    """
    Compute a geodesic in latent space between z_start and z_end by
    minimizing the discrete curve energy under the pull-back metric.

    The endpoints are fixed; only the n_points-2 interior points are optimised.

    Parameters:
    z_start: [torch.Tensor]  shape (M,)
    z_end:   [torch.Tensor]  shape (M,)
    decoder_net: [torch.nn.Module]
    n_points: [int]  total number of points on the curve (including endpoints)
    lr: [float]  learning rate for Adam
    n_steps: [int]  number of optimisation steps

    Returns:
    curve: [torch.Tensor]  shape (n_points, M)  — the optimised curve
    """
    M = z_start.shape[0]

    # Initialise interior points as a straight line between the endpoints
    t = torch.linspace(0, 1, n_points, device=device)       # (n_points,)
    init = z_start.unsqueeze(0) + t.unsqueeze(1) * (z_end - z_start).unsqueeze(0)
    # Interior points are learnable; endpoints are fixed
    interior = nn.Parameter(init[1:-1].clone())              # (n_points-2, M)

    optimizer = torch.optim.Adam([interior], lr=lr)

    for _ in range(n_steps):
        optimizer.zero_grad()
        # Assemble full curve: fixed start + learnable interior + fixed end
        curve = torch.cat([
            z_start.unsqueeze(0),
            interior,
            z_end.unsqueeze(0)
        ], dim=0)                                            # (n_points, M)

        energy = curve_energy(curve, decoder_net)
        energy.backward()
        optimizer.step()

    with torch.no_grad():
        curve = torch.cat([
            z_start.unsqueeze(0),
            interior,
            z_end.unsqueeze(0)
        ], dim=0)

    return curve.detach()


# ---------------------------------------------------------------------------
# Plotting helper — separated so we can call it from both geodesics + plot modes
# ---------------------------------------------------------------------------

def _plot_geodesics(geodesics, all_z, all_y, num_classes,
                    out_path, show_endpoints=True, title=""):
    """
    Plot latent space scatter with geodesic curves overlaid.

    Parameters:
    geodesics: list of (n_points, 2) tensors
    all_z: (N, 2) tensor of latent means
    all_y: (N,) tensor of class labels
    num_classes: int
    out_path: str — where to save the png
    show_endpoints: bool — whether to mark curve endpoints as red dots
    title: str — plot title (empty string = no title)
    """
    colors      = ["tab:blue", "tab:orange", "tab:green"]
    class_names = ["0", "1", "2"]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Scatter latent means coloured by class
    for c in range(num_classes):
        mask = all_y == c
        ax.scatter(
            all_z[mask, 0], all_z[mask, 1],
            s=6, alpha=0.4, color=colors[c], label=f"Class {class_names[c]}"
        )

    # Overlay geodesic curves
    for curve in geodesics:
        ax.plot(curve[:, 0].numpy(), curve[:, 1].numpy(),
                color="black", linewidth=0.8, alpha=0.7)
        if show_endpoints:
            ax.scatter(curve[0, 0].item(),  curve[0, 1].item(),
                       color="red", s=20, zorder=5)
            ax.scatter(curve[-1, 0].item(), curve[-1, 1].item(),
                       color="red", s=20, zorder=5)

    if title:
        ax.set_title(title)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.legend(loc="upper right", markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "plot"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument("--experiment-folder", type=str, default="experiment")
    parser.add_argument("--samples", type=str, default="samples.png")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=32, metavar="N")
    parser.add_argument("--epochs-per-decoder", type=int, default=50, metavar="N")
    parser.add_argument("--latent-dim", type=int, default=2, metavar="N")
    parser.add_argument("--num-decoders", type=int, default=3, metavar="N")
    parser.add_argument("--num-reruns", type=int, default=10, metavar="N")
    parser.add_argument("--num-curves", type=int, default=25, metavar="N",
                        help="number of geodesics to plot (default: %(default)s)")
    parser.add_argument("--num-t", type=int, default=20, metavar="N",
                        help="number of points along the curve (default: %(default)s)")
    parser.add_argument("--geodesic-lr", type=float, default=1e-2,
                        help="learning rate for geodesic optimiser (default: %(default)s)")
    parser.add_argument("--geodesic-steps", type=int, default=500,
                        help="optimisation steps for geodesic (default: %(default)s)")

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]
        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST("data/", train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))
    test_tensors = datasets.MNIST("data/", train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))
    train_data = subsample(train_tensors.data, train_tensors.targets,
                           num_train_data, num_classes)
    test_data = subsample(test_tensors.data, test_tensors.targets,
                          num_train_data, num_classes)

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # Model factories
    # ------------------------------------------------------------------
    M = args.latent_dim

    def new_encoder():
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )

    def new_decoder():
        return nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------
    if args.mode == "train":
        os.makedirs(args.experiment_folder, exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, mnist_train_loader, args.epochs_per_decoder, device)
        torch.save(model.state_dict(), f"{args.experiment_folder}/model.pt")
        print(f"Model saved to {args.experiment_folder}/model.pt")

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt",
                                         map_location=device))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)
            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(torch.cat([data.cpu(), recon.cpu()], dim=0),
                       "reconstruction_means.png")

    elif args.mode == "eval":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt",
                                         map_location=device))
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbos.append(model.elbo(x))
        print("Mean test ELBO:", torch.tensor(elbos).mean().item())

    elif args.mode == "geodesics":
        # ---------------------------------------------------------------
        # Load model
        # ---------------------------------------------------------------
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt",
                                         map_location=device))
        model.eval()
        decoder_net = model.decoder.decoder_net

        # ---------------------------------------------------------------
        # Encode all test data to get latent means + labels
        # ---------------------------------------------------------------
        all_z, all_y = [], []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                z_mean = model.encoder(x).mean
                all_z.append(z_mean.cpu())
                all_y.append(y)
        all_z = torch.cat(all_z, dim=0)   # (N_test, 2)
        all_y = torch.cat(all_y, dim=0)   # (N_test,)

        # ---------------------------------------------------------------
        # Randomly pick pairs of latent points for geodesics
        # ---------------------------------------------------------------
        torch.manual_seed(42)
        n_curves = args.num_curves
        idx = torch.randperm(len(all_z))[:2 * n_curves]
        starts = all_z[idx[:n_curves]].to(device)    # (n_curves, 2)
        ends   = all_z[idx[n_curves:]].to(device)    # (n_curves, 2)

        # ---------------------------------------------------------------
        # Compute geodesics
        # ---------------------------------------------------------------
        print(f"Computing {n_curves} geodesics "
              f"({args.num_t} points, {args.geodesic_steps} steps each)...")

        geodesics = []
        for i in tqdm(range(n_curves), desc="Geodesics"):
            curve = compute_geodesic(
                starts[i], ends[i],
                decoder_net,
                n_points=args.num_t,
                lr=args.geodesic_lr,
                n_steps=args.geodesic_steps,
                device=device,
            )
            geodesics.append(curve.cpu())

        # ---------------------------------------------------------------
        # Save geodesics + latent data so we can replot without rerunning
        # ---------------------------------------------------------------
        torch.save({
            "geodesics": geodesics,
            "all_z": all_z,
            "all_y": all_y,
        }, f"{args.experiment_folder}/geodesics_data.pt")
        print(f"Geodesic data saved to {args.experiment_folder}/geodesics_data.pt")

        # ---------------------------------------------------------------
        # Plot: latent space scatter + geodesics
        # ---------------------------------------------------------------
        _plot_geodesics(
            geodesics, all_z, all_y,
            num_classes=num_classes,
            out_path=f"{args.experiment_folder}/geodesics_partA.png",
            show_endpoints=True,
            title="Latent space with pull-back geodesics (Part A)",
        )

    elif args.mode == "plot":
        # ---------------------------------------------------------------
        # Just reload saved data and replot — no recomputation needed
        # ---------------------------------------------------------------
        data_path = f"{args.experiment_folder}/geodesics_data.pt"
        assert os.path.exists(data_path), \
            f"No saved data found at {data_path}. Run 'geodesics' mode first."

        saved = torch.load(data_path)
        geodesics = saved["geodesics"]
        all_z     = saved["all_z"]
        all_y     = saved["all_y"]
        num_classes = 3

        _plot_geodesics(
            geodesics, all_z, all_y,
            num_classes=num_classes,
            out_path=f"{args.experiment_folder}/geodesics_partA.png",
            show_endpoints=False,       # ← change to True to show red dots
            title="",                   # ← set your title here, or leave empty
        )
        print(f"Replot saved to {args.experiment_folder}/geodesics_partA.png")