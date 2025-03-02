from random import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wandb.integration.torch.wandb_torch import torch


def split_dataset(filepath, save_dir, test_size=0.1, val_size=0.1, random_state=0):
    data = pd.read_csv(filepath)
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    adjusted_val_size = val_size / (1.0 - test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=adjusted_val_size, random_state=random_state)

    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_data.to_csv(f"{save_dir}/train_data.csv", index=False)
    val_data.to_csv(f"{save_dir}/val_data.csv", index=False)
    test_data.to_csv(f"{save_dir}/test_data.csv", index=False)
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
def process_and_save_edges(train_data_path, node_path, edges_output_path, use_real_label):

    train_data_df = pd.read_csv(train_data_path)
    node_df = pd.read_csv(node_path)

    name_to_id_map = node_df.set_index('node_name')['node_id'].to_dict()

    edges = []

    for index, row in train_data_df.iterrows():
        if (use_real_label and row['result'] > 0) or (not use_real_label and row['RefD'] > 0):
            source_node_name = row['A']
            target_node_name = row['B']
            source_node_id = name_to_id_map.get(source_node_name, None)
            target_node_id = name_to_id_map.get(target_node_name, None)

            if source_node_id is not None and target_node_id is not None:
                edges.append((source_node_id, target_node_id))

    edges_df = pd.DataFrame(edges, columns=['source_node_id', 'target_node_id'])

    edges_df.to_csv(edges_output_path, index=False)

if __name__ == "__main__":
    split_dataset('../Dataset/University Course/pairs.csv', '../Dataset/University Course/')

    process_and_save_edges('../Dataset/University Course/train_data.csv',
                           '../Dataset/University Course/emb.csv',
                           '../Dataset/University Course/edges.csv', True)





