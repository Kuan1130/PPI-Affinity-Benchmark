import os
import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

def load_strict_dataset(csv_path, graph_dir, batch_size=64): 
    df = pd.read_csv(csv_path)
    inter_list, intra1_list, intra2_list = [], [], []
    inter_dir = os.path.join(graph_dir, 'inter_graph')
    indi_dir = os.path.join(graph_dir, 'individual_graph')
    
    for idx, row in df.iterrows():
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
                
            g_inter.y = torch.tensor([float(row['proaffinity_label'])], dtype=torch.float)
            
            inter_list.append(g_inter)
            intra1_list.append(g_intra1)
            intra2_list.append(g_intra2)

    loader_inter = DataLoader(inter_list, batch_size=batch_size, shuffle=False)
    loader_intra1 = DataLoader(intra1_list, batch_size=batch_size, shuffle=False)
    loader_intra2 = DataLoader(intra2_list, batch_size=batch_size, shuffle=False)
    
    return loader_inter, loader_intra1, loader_intra2, len(inter_list)

class AttentiveFPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout):
        super(AttentiveFPModel, self).__init__()
        self.model = AttentiveFP(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        return self.model(x, edge_index, edge_attr, batch)

class GraphNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout, linear_out1, linear_out2):
        super(GraphNetwork, self).__init__()
        self.graph1 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        self.graph2 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        self.graph3 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        
        self.fc1 = torch.nn.Linear(out_channels * 3, linear_out1)
        self.fc2 = torch.nn.Linear(linear_out1, linear_out2)

    def forward(self, inter_data, intra_data1, intra_data2):
        inter_graph = self.graph1(inter_data)
        intra_graph1 = self.graph2(intra_data1)
        intra_graph2 = self.graph3(intra_data2)
        x = torch.cat([inter_graph, intra_graph1, intra_graph2], dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, dl_inter, dl_intra1, dl_intra2, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for d_inter, d_intra1, d_intra2 in zip(dl_inter, dl_intra1, dl_intra2):
        d_inter, d_intra1, d_intra2 = d_inter.to(device), d_intra1.to(device), d_intra2.to(device)
        optimizer.zero_grad()
        out = model(d_inter, d_intra1, d_intra2).squeeze()
        loss = criterion(out, d_inter.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * d_inter.num_graphs
    return total_loss / len(dl_inter.dataset)

def evaluate_metrics(model, dl_inter, dl_intra1, dl_intra2, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for d_inter, d_intra1, d_intra2 in zip(dl_inter, dl_intra1, dl_intra2):
            d_inter, d_intra1, d_intra2 = d_inter.to(device), d_intra1.to(device), d_intra2.to(device)
            out = model(d_inter, d_intra1, d_intra2).squeeze()
            loss = criterion(out, d_inter.y)
            total_loss += loss.item() * d_inter.num_graphs
            
            if out.dim() == 0: out = out.unsqueeze(0)
            
            all_preds.extend(out.cpu().numpy())
            all_trues.extend(d_inter.y.cpu().numpy())
            
    avg_loss = total_loss / len(dl_inter.dataset)
    
    rp, _ = pearsonr(all_trues, all_preds)
    rs, _ = spearmanr(all_trues, all_preds)
    rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
    mae = mean_absolute_error(all_trues, all_preds)
    
    return avg_loss, rp, rs, rmse, mae

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    # dir
    graph_base_dir = '/root/autodl-tmp/5/ProAffinity_Test/ProAffinity-GNN/data/graph'
    base_csv_dir = '/root/autodl-tmp/5/ProAffinity_Test/random_seeds_splits' 
    base_model_dir = 'model'

    seed_folders = ['seed_0', 'seed_1', 'seed_42', 'seed_142', 'seed_4242']
    
    bs = 64
    ep = 100
    patience = 15
    
    final_results = {'Rp': [], 'Rs': [], 'RMSE': [], 'MAE': []}

    for seed in seed_folders:
        print("\n" + "="*20)
        print(f"Running seed {seed}...")
        print("="*20)

        csv_dir = os.path.join(base_csv_dir, seed)
        model_dir = os.path.join(base_model_dir, seed)
        os.makedirs(model_dir, exist_ok=True)
        model_save = os.path.join(model_dir, 'best_model.pkl')

        print(f"\n Loading Datasets for {seed}...")
        tr_inter, tr_in1, tr_in2, tr_len = load_strict_dataset(os.path.join(csv_dir, 'train_split.csv'), graph_base_dir, bs)
        val_inter, val_in1, val_in2, val_len = load_strict_dataset(os.path.join(csv_dir, 'val_split.csv'), graph_base_dir, bs)
        te_inter, te_in1, te_in2, te_len = load_strict_dataset(os.path.join(csv_dir, 'test_split.csv'), graph_base_dir, bs)

        sample_data = next(iter(tr_inter))
        model = GraphNetwork(
            in_channels=sample_data.num_node_features, hidden_channels=256, out_channels=64,
            edge_dim=sample_data.num_edge_features, num_layers=3, num_timesteps=2,
            dropout=0.5, linear_out1=32, linear_out2=1
        ).to(device)

        # Adam + MSELoss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True) # 改 mode='max', patience=10
        criterion = torch.nn.MSELoss()

        best_rp = -float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []

        print(f"\n Start training {seed}...")
        for epoch in range(ep):
            t_loss = train(model, tr_inter, tr_in1, tr_in2, optimizer, criterion, device)
            v_loss, v_rp, v_rs, v_rmse, v_mae = evaluate_metrics(model, val_inter, val_in1, val_in2, criterion, device)
            
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | Train MSE: {t_loss:.4f} | Val MSE: {v_loss:.4f} | Val Rp: {v_rp:.3f}")
            
            scheduler.step(v_rp) 
            
            if v_rp > best_rp:
                best_rp = v_rp
                patience_counter = 0
                torch.save(model.state_dict(), model_save)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early Stopping Triggered, stopped at Epoch {epoch+1}")
                break

        print(f"\n Loading the best model for {seed}...")
        model.load_state_dict(torch.load(model_save))
        te_loss, te_rp, te_rs, te_rmse, te_mae = evaluate_metrics(model, te_inter, te_in1, te_in2, criterion, device)
        
        print(f"\n Test Score for {seed} ")
        print(f"Rp: {te_rp:.4f} | Rs: {te_rs:.4f} | RMSE: {te_rmse:.4f} | MAE: {te_mae:.4f}")
        
        final_results['Rp'].append(te_rp)
        final_results['Rs'].append(te_rs)
        final_results['RMSE'].append(te_rmse)
        final_results['MAE'].append(te_mae)

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss (MSE)')
        plt.plot(val_losses, label='Validation Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curve ({seed})')
        plt.legend()
        plt.savefig(os.path.join(model_dir, f'loss_curve_{seed}.png'))
        plt.close() 

    print("\n" + "="*50)
    print(" FINAL 5-SEED AVERAGE RESULTS ")
    print("="*50)
    print(f"Pearson (Rp):  {np.mean(final_results['Rp']):.4f} ± {np.std(final_results['Rp']):.4f}")
    print(f"Spearman (Rs): {np.mean(final_results['Rs']):.4f} ± {np.std(final_results['Rs']):.4f}")
    print(f"RMSE:          {np.mean(final_results['RMSE']):.4f} ± {np.std(final_results['RMSE']):.4f}")
    print(f"MAE:           {np.mean(final_results['MAE']):.4f} ± {np.std(final_results['MAE']):.4f}")
    print("="*50)