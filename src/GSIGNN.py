import torch
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
import os
import random
from losses import square_contrastive_loss
from utils import build_sims_matrix
from utils import build_sims_matrix2
from utils import build_sims_matrix_similarity
from utils import cosine_similarity_matrix
from utils import preprocess_embeddings
from utils import create_adjacency_matrix
from model import GSIGNN



# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# set_seed(42)
#
# torch.use_deterministic_algorithms(True)

data_path = '../Dataset/University Course/pairs.csv'
node_path = '../Dataset/University Course/emb.csv'
edges_path= '../Dataset/University Course/edges.csv'

nodes_df = pd.read_csv(node_path)

num_nodes = nodes_df['node_id'].nunique()

# 应用预处理
nodes_df['lm_emb'] = nodes_df['lm_emb'].apply(preprocess_embeddings)
nodes_df['node_emb'] = nodes_df['node_emb'].apply(preprocess_embeddings)
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

edges_df = pd.read_csv(edges_path)
num_nodes = nodes_df['node_id'].nunique()
edge_index = torch.tensor([edges_df['source_node_id'].values, edges_df['target_node_id'].values], dtype=torch.long)
node_features = torch.stack(nodes_df['node_emb'].tolist())
data = Data(x=node_features, edge_index=edge_index)
lm_embeddings = torch.stack(nodes_df['lm_emb'].tolist())
data.lm_emb = lm_embeddings

#sims_matrix = build_sims_matrix(data_path, node_path)
#sims_matrix = build_sims_matrix2(data_path, node_path)
sims_matrix = build_sims_matrix_similarity(node_path)

opt = {
        'state_dim': 1024,
        'annotation_dim': 0,
        'n_edge_types': 1,
        'n_node': num_nodes,
        'n_steps': 2
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GSIGNN(opt).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
A_in_tensor, A_out_tensor, A = create_adjacency_matrix(edges_path, num_nodes, 'cuda')


for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, A)
    data.lm_emb = data.lm_emb.to(device)
    logits = cosine_similarity_matrix(out, data.lm_emb.to(device))
    sims_tensor = torch.tensor(sims_matrix, dtype=logits.dtype)

    if logits.is_cuda:
        sims_tensor = sims_tensor.to(logits.device)
    loss = square_contrastive_loss(logits,sims_tensor,sim_weights='identity', alpha=0.1)
    #loss = square_contrastive_loss(logits)
    loss.backward()
    optimizer.step()

out = out.to(device)
data.lm_emb = data.lm_emb.to(device)

first_node_feature = out[0]

# dot_products = torch.matmul(first_node_feature, data.lm_emb.t())
# top_values, top_indices = torch.topk(dot_products, 10)
# for i in range(10):
#     print(f" {top_indices[i].item()}, : {top_values[i].item()}")


out_cpu = out.detach().cpu().numpy()


node_embs_str = [",".join(map(str, emb)) for emb in out_cpu]


final_df = pd.DataFrame({
    "node_name": nodes_df["node_name"],
    "node_emb": node_embs_str
})


final_df.to_csv("../Dataset/University Course/final_node_embeddings.csv", index=False)
print("Embedding has saved to final_node_embeddings.csv")


out = out.cpu()
data.lm_emb = data.lm_emb.cpu()
combined_emb = torch.cat((out, data.lm_emb), dim=1)
combined_emb_str = [",".join(map(str, emb.tolist())) for emb in combined_emb]
final_df['combined_emb'] = combined_emb_str

final_df.to_csv("../Dataset/University Course/final_node_embeddings_with_combined_emb.csv", index=False)


