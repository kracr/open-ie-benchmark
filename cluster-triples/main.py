import torch
from torch_geometric.loader import DataLoader
from GNN.dataset import GraphData
from torch_geometric.nn import GAE, VGAE

from GNN.autoencoder import VariationalLinearEncoder, VariationalGCNEncoder, LinearEncoder, GCNEncoder
from GNN.cluster import cluster

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 100
lr = 3e-3


dataset = GraphData(root="./data")
loader = DataLoader(dataset, batch_size=2000)

in_channels, out_channels = dataset.num_features, 128

# model = GAE(GCNEncoder(in_channels, out_channels))

model = GAE(LinearEncoder(in_channels, out_channels))

# model = VGAE(VariationalGCNEncoder(in_channels, out_channels))

# model = VGAE(VariationalLinearEncoder(in_channels, out_channels))


model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def train():
    l=0
    model.train()
    optimizer.zero_grad()

    for batch in loader:

        batch = batch.to(device)
        z = model.encode(batch.x, batch.edge_index)
        
        loss = model.recon_loss(z, batch.edge_index)/cluster(model, device)
        # if args.variational:
        # loss = loss + (1 / batch.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        l+=loss.item()
    return float(l)



for epoch in range(1, epochs + 1):
    loss = train()
    print(f'Epoch: {epoch:03d}, loss: {loss:.4f}')
    torch.save(model, './checkpoint/model.pt')