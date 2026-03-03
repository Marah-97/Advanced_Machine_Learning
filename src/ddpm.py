import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=1000):
        super(DDPM, self).__init__()
        self.network = network
        self.T = T

        betas = torch.linspace(beta_1, beta_T, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion q(x_t | x_0).

        x0: (B, D)
        t: (B,) long
        noise: optional noise (B, D)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x0.ndim - 1)))
        return sqrt_a * x0 + sqrt_om * noise

    def p_mean_variance(self, xt, t):
        """Predict mean and variance of p(x_{t-1} | x_t).

        Returns mean, variance.
        """
        # predict noise epsilon_theta
        t_input = t.float().unsqueeze(1) / float(self.T)
        predicted_noise = self.network(xt, t_input)

        beta_t = self.betas[t].view(-1, *([1] * (xt.ndim - 1)))
        alpha_t = self.alphas[t].view(-1, *([1] * (xt.ndim - 1)))
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, *([1] * (xt.ndim - 1)))
        alpha_cumprod_prev = torch.cat([self.alphas_cumprod.new_tensor([1.0]), self.alphas_cumprod[:-1]])[t].view(-1, *([1] * (xt.ndim - 1)))

        # posterior mean
        mean = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise)

        # variance according to Ho et al.
        var = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
        return mean, var

    def p_sample(self, xt, t):
        mean, var = self.p_mean_variance(xt, t)
        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).view(-1, *([1] * (xt.ndim - 1))).float()
        return mean + nonzero_mask * torch.sqrt(var) * noise

    def sample(self, shape, device=None):
        if device is None:
            device = self.sqrt_alphas_cumprod.device
        xt = torch.randn(*shape, device=device)
        for time in reversed(range(self.T)):
            t = torch.full((shape[0],), time, dtype=torch.long, device=device)
            xt = self.p_sample(xt, t)
        return xt

    def negative_elbo(self, x):
        """Compute the simplified DDPM loss (noise prediction MSE) per sample."""
        batch = x.size(0)
        t = torch.randint(0, self.T, (batch,), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        xt = self.q_sample(x, t, noise)
        t_input = t.float().unsqueeze(1) / float(self.T)
        predicted_noise = self.network(xt, t_input)
        mse = F.mse_loss(predicted_noise, noise, reduction='none')
        return mse.view(batch, -1).mean(dim=1)

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use {tg: two Gaussians, cb: chequerboard, mnist} (default: %(default)s)')
    parser.add_argument('--arch', type=str, default='fc', choices=['fc','unet'], help='network architecture to use for MNIST: fc (default) or unet')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of diffusion timesteps (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--n-samples', type=int, default=4, help='number of generated samples to write (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Handle MNIST separately (no ToyData required)
    if args.data == 'mnist':
        # Use MNIST images in [-1,1], flattened to vectors
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x*2.0-1.0), transforms.Lambda(lambda x: x.view(-1))])
        train_dataset = datasets.MNIST('data/', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data/', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Get the dimension of the dataset (flattened 28*28)
        D = train_dataset[0][0].shape[0]

        # Use the fully connected network for flattened MNIST
        num_hidden = 1024
        network = FcNetwork(D, num_hidden)

    else:
        # Toy datasets require the ToyData module (only import when needed)
        try:
            import ToyData
        except Exception as e:
            raise ImportError('ToyData module is required for toy datasets (tg/cb). Use --data mnist to run on MNIST.') from e

        # Generate the toy data
        n_data = 10000000
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)

        # Get the dimension of the dataset
        D = next(iter(train_loader)).shape[1]

        # Define the network for toy data
        num_hidden = 64
        network = FcNetwork(D, num_hidden)

    # Set the number of steps in the diffusion process from CLI
    T = int(args.timesteps)

    # Optionally use a U-Net architecture for MNIST
    if args.data == 'mnist' and args.arch == 'unet':
        try:
            from unet import Unet
            network = Unet()
        except Exception:
            # fall back to FC if Unet import fails
            print('Warning: failed to import Unet; falling back to FcNetwork')

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
    model.eval()
    with torch.no_grad():
        if args.data == 'mnist':
            ns = args.n_samples

            # --- Timing starts ---
            import time
            start_time = time.time()
            samples = model.sample((ns, D)).cpu()
            end_time = time.time()
            # --- Timing ends ---

            elapsed = end_time - start_time
            print(f"Sampling time: {elapsed:.4f} seconds")
            print(f"Samples per second: {ns / elapsed:.2f}")

            # reshape to images in [0,1]
            samples = (samples.view(-1, 1, 28, 28) + 1.0) / 2.0

            # arrange into a square grid (fallback to 1xN)
            nrow = int(ns**0.5)
            if nrow * nrow != ns:
                nrow = ns

            save_image(samples[:ns], args.samples, nrow=nrow)
            print(f"{ns} samples saved -> {args.samples}")

        else:
            ns = 10000

            # --- Timing starts ---
            import time
            start_time = time.time()
            samples = model.sample((ns, D)).cpu()
            end_time = time.time()
            # --- Timing ends ---

            elapsed = end_time - start_time
            print(f"Sampling time: {elapsed:.4f} seconds")
            print(f"Samples per second: {ns / elapsed:.2f}")

            # Transform back to original space
            samples = samples / 2 + 0.5

            # Plot toy density
            coordinates = [[[x, y] for x in np.linspace(*toy.xlim, 1000)]
                        for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob,
                        extent=[toy.xlim[0], toy.xlim[1],
                                toy.ylim[0], toy.ylim[1]],
                        origin='lower',
                        cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()