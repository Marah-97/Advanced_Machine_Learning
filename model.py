import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GNNEncoder(nn.Module):
    """Encoder using Message Passing (GIN) to map graphs to latent space."""
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3, dropout=0.1):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        in_dim = input_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), 
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim
            
        self.dropout = nn.Dropout(dropout)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
            h = self.dropout(h)
        h_graph = global_mean_pool(h, batch)
        return self.fc_mean(h_graph), self.fc_log_var(h_graph)

class GNNDecoder(nn.Module):
    """Decoder that reconstructs the adjacency matrix from the latent vector z."""
    def __init__(self, latent_dim, hidden_dim, max_nodes):
        super(GNNDecoder, self).__init__()
        self.max_nodes = max_nodes
        output_dim = max_nodes * max_nodes
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        logits = self.mlp(z)
        return logits.view(-1, self.max_nodes, self.max_nodes)

class GraphVAE(nn.Module):
    """The Variational Autoencoder (VAE) for Graphs."""
    def __init__(self, input_dim, hidden_dim, latent_dim, max_nodes):
        super(GraphVAE, self).__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = GNNDecoder(latent_dim, hidden_dim, max_nodes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar = self.encoder(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        adj_recon = self.decoder(z)
        return adj_recon, mu, logvar