import os
import pandas as pd
import torch
import pickle
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def load_strict_dataset(csv_path, graph_dir):
    df = pd.read_csv(csv_path)
    
    inter_list = []
    intra1_list = []
    intra2_list = []
    
    inter_dir = os.path.join(graph_dir, 'inter_graph')
    indi_dir = os.path.join(graph_dir, 'individual_graph')
    
    print(f"Checking {csv_path}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pdb = str(row['pdb_code']).lower()
        
        path_inter = os.path.join(inter_dir, pdb)
        path_intra1 = os.path.join(indi_dir, f"{pdb}_1")
        path_intra2 = os.path.join(indi_dir, f"{pdb}_2")
        
        if os.path.exists(path_inter) and os.path.exists(path_intra1) and os.path.exists(path_intra2):
            
            with open(path_inter, 'rb') as f:
                g_inter = pickle.load(f)
                g_inter.edge_attr = g_inter.edge_attr.float()
                
            with open(path_intra1, 'rb') as f:
                g_intra1 = pickle.load(f)
                g_intra1.edge_attr = g_intra1.edge_attr.float()
                
            with open(path_intra2, 'rb') as f:
                g_intra2 = pickle.load(f)
                g_intra2.edge_attr = g_intra2.edge_attr.float()
                
            label = float(row['proaffinity_label'])
            g_inter.y = torch.tensor([label], dtype=torch.float)
            
            inter_list.append(g_inter)
            intra1_list.append(g_intra1)
            intra2_list.append(g_intra2)
            
        else:
            print(f" Error：{pdb} file missed，skipped")

    print(f" {len(inter_list)} perfect data found\n")
    
    loader_inter = DataLoader(inter_list, batch_size=16, shuffle=False)
    loader_intra1 = DataLoader(intra1_list, batch_size=16, shuffle=False)
    loader_intra2 = DataLoader(intra2_list, batch_size=16, shuffle=False)
    
    return loader_inter, loader_intra1, loader_intra2

graph_base_dir = '/root/autodl-tmp/5/ProAffinity_Test/ProAffinity-GNN/data/graph'
csv_dir = '/root/autodl-tmp/5/ProAffinity_Test'

train_csv = os.path.join(csv_dir, 'train_split.csv')
train_inter, train_intra1, train_intra2 = load_strict_dataset(train_csv, graph_base_dir)