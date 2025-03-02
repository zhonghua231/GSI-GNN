import pandas as pd
import numpy as np
import torch

def create_adjacency_matrix(edges_path, num_nodes, device='cuda'):
    edges_df = pd.read_csv(edges_path)
    A_in = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    A_out = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    out_degree = np.zeros(num_nodes, dtype=np.float32)

    for _, row in edges_df.iterrows():
        src = int(row['source_node_id'])
        out_degree[src] += 1

    for _, row in edges_df.iterrows():
        src = int(row['source_node_id'])
        dst = int(row['target_node_id'])

        A_out[src, dst] = 1.0 / out_degree[src] if out_degree[src] > 0 else 0

        A_in[dst, src] = 1

    A_in_tensor = torch.tensor(A_in, device=device)
    A_out_tensor = torch.tensor(A_out, device=device)

    A = torch.cat([A_in_tensor.unsqueeze(-1), A_out_tensor.unsqueeze(-1)], dim=-1)

    print(A_in.shape)
    print(A_out.shape)
    print(A.shape)

    return A_in_tensor, A_out_tensor, A


def preprocess_embeddings(embeddings_str):
    embeddings = [float(x) for x in embeddings_str.strip('[]').split(',')]
    return torch.tensor(embeddings, dtype=torch.float)

def build_sims_matrix(data_path, node_path):
    node_df = pd.read_csv(node_path)

    unique_node_ids = node_df['node_id'].unique()
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(unique_node_ids)}

    sims_matrix = np.zeros((len(unique_node_ids), len(unique_node_ids)))
    np.fill_diagonal(sims_matrix, 1)

    geometry_df = pd.read_csv(data_path)

    node_name_to_id = node_df.set_index('node_name')['node_id'].to_dict()

    for _, row in geometry_df.iterrows():
        if row['RefD'] > 0:
            a_node_id = node_name_to_id.get(row['A'])
            b_node_id = node_name_to_id.get(row['B'])
            if a_node_id in node_id_to_index and b_node_id in node_id_to_index:
                a_index = node_id_to_index[a_node_id]
                b_index = node_id_to_index[b_node_id]
                sims_matrix[a_index, b_index] = 1
                sims_matrix[b_index, a_index] = 1

    print("sims_matrix shape is :")
    print(sims_matrix.shape)

    return sims_matrix

def build_sims_matrix2(data_path, node_path):
    node_df = pd.read_csv(node_path)

    unique_node_ids = node_df['node_id'].unique()
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(unique_node_ids)}

    sims_matrix = np.zeros((len(unique_node_ids), len(unique_node_ids)))
    np.fill_diagonal(sims_matrix, 1)

    geometry_df = pd.read_csv(data_path)

    node_name_to_id = node_df.set_index('node_name')['node_id'].to_dict()

    for _, row in geometry_df.iterrows():
        if row['result'] > 0:
            a_node_id = node_name_to_id.get(row['A'])
            b_node_id = node_name_to_id.get(row['B'])
            if a_node_id in node_id_to_index and b_node_id in node_id_to_index:
                a_index = node_id_to_index[a_node_id]
                b_index = node_id_to_index[b_node_id]
                sims_matrix[a_index, b_index] = 1
                #sims_matrix[b_index, a_index] = 1

    print("sims_matrix shape is :")
    print(sims_matrix.shape)

    return sims_matrix

def cosine_similarity_matrix(x, y):
    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)
    similarity = torch.matmul(x_norm, y_norm.t())
    return similarity


def build_sims_matrix_similarity(node_path):
    df = pd.read_csv(node_path)
    embeddings = torch.tensor(df['lm_emb'].apply(lambda x: list(map(float, x.split(',')))).tolist())

    sims_matrix = cosine_similarity_matrix(embeddings, embeddings)

    return sims_matrix

# data_path = '../data/ALCPL_data_geometry/geometry.csv'
# node_path = '../data/ALCPL_data_geometry/geometry(id+name+lm+node).csv'
# sims_matrix = build_sims_matrix(data_path, node_path)
# print(sims_matrix)
# print(sims_matrix.shape)
# np.set_printoptions(threshold=np.inf)
# print(sims_matrix)