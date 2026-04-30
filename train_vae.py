import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import pickle
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_networkx
from model import GraphVAE

# Configs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_NODES = 28 
LATENT_DIM = 32
EPOCHS = 150

# 1. Load Data
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
train_graphs = [to_networkx(d, to_undirected=True) for d in dataset]

# 2. Init Model
model = GraphVAE(input_dim=7, hidden_dim=64, latent_dim=LATENT_DIM, max_nodes=MAX_NODES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop (Task 2.3)
print("Training the Deep Generative Model...")
model.train()
for epoch in range(EPOCHS + 1):
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        adj_recon, mu, logvar = model(data)
        adj_target = to_dense_adj(data.edge_index, data.batch, max_num_nodes=MAX_NODES)
        
        recon_loss = F.binary_cross_entropy_with_logits(adj_recon, adj_target, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (recon_loss + kl_loss) / data.num_graphs
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

# 4. Sampling 1000 graphs
print("\nSampling 1000 graphs...")
model.eval()
vae_generated_graphs = []
with torch.no_grad():
    z = torch.randn(1000, LATENT_DIM).to(device)
    adj_logits = model.decoder(z)
    adjs = (torch.sigmoid(adj_logits) > 0.5).float().cpu().numpy()
    
    for i in range(1000):
        G = nx.from_numpy_array(adjs[i])
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() > 0:
            vae_generated_graphs.append(G)

# 5. Compute Metrics (Novelty & Uniqueness)
print("\nComputing Evaluation Metrics...")
train_hashes = {nx.weisfeiler_lehman_graph_hash(g) for g in train_graphs}
gen_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in vae_generated_graphs]

novel_count = sum(1 for h in gen_hashes if h not in train_hashes)
unique_hashes = set(gen_hashes)
novel_unique_count = len({h for h in gen_hashes if h not in train_hashes})
n = len(vae_generated_graphs)

print("\n--- Task 2.4: Novelty & Uniqueness Table ---")
print(f"{'Model':<25} | {'Novel':<7} | {'Unique':<7} | {'Novel+Unique':<12}")
print("-" * 60)
print(f"{'Deep Generative Model':<25} | {(novel_count/n)*100:>6.1f}% | {(len(unique_hashes)/n)*100:>6.1f}% | {(novel_unique_count/n)*100:>11.1f}%")

# 6. Save results for the final plot script
with open('team2_vae_graphs.pkl', 'wb') as f:
    pickle.dump(vae_generated_graphs, f)

print(f"\nSuccess! Graphs saved to 'team2_vae_graphs.pkl'.")