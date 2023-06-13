import torch
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GAE, VGAE

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score

from GNN.dataset import GraphData
from GNN.autoencoder import VariationalLinearEncoder, VariationalGCNEncoder, LinearEncoder, GCNEncoder

from torchclustermetrics import silhouette


def get_data(dataset, device, encoder):
    graph_embeddings = []
    files = []

    for i in range(9999):
        batch, file_name = dataset.get_test(i)
        files.append(file_name)
        batch = batch.to(device)
        try:
            out = encoder(batch.x, batch.edge_index)
        except:
            print(file_name)
            continue
        
        d = out.view(-1,71,128)
        edge_indexes = batch.edge_index.flatten()  #reshape(2*71,-1).transpose(1,0)

        # for idx, edges in enumerate(edge_indexes):
        x = torch.sum(d[:, edge_indexes, :], dim=1)  #calulating dimensions by summing all the nodes.

        graph_embeddings.append(x)
    
    return graph_embeddings


def cluster(model, device):
    dataset = GraphData(root="./data")
    in_channels, out_channels = dataset.num_features, 128

    model = GAE(LinearEncoder(in_channels, out_channels))

    model = torch.load('checkpoint/model.pt', map_location=device)
    model.eval()
    encoder = model.encoder
    sil = silhouette()
    graph_embeddings = get_data(dataset, device, encoder)
    graph_embeddings = torch.stack(graph_embeddings).squeeze().detach()

    clustering = AgglomerativeClustering(n_clusters=6).fit(graph_embeddings.cpu())
    clustering_labels = clustering.labels_

    return  torch.tensor(davies_bouldin_score(graph_embeddings.cpu(), clustering_labels), device=device)   #sil.score(graph_embeddings, clustering_labels)