import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shutil
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchinfo import summary
import esm
import model_final
import tqdm
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Check if CUDA is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Current device: {device}")

# Parameters
epochs = 30
pro_len = 2000  
d_embed = 64  
d_ff = 128  
d_k = d_v = 32  
n_layers_en = 1  
n_heads = 8  
batch_size = 8

# Data Processing
print('Initializing ESM Model...')
_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

def load_batches_from_disk(output_path):
    protein_names, seqs, chain_id_res, enc_tokens, seq_features = [], [], [], [], []
    coor_features, interface_atoms, affinity, interaction_type = [], [], [], []
    interaction_matrix, res_mass_centor, hetatm_features = [], [], []

    batch_files = sorted([f for f in os.listdir(output_path) if f.startswith("batch_") and f.endswith(".pt")])
    for batch_file in tqdm.tqdm(batch_files, desc=f"Loading {os.path.basename(output_path)}"):
        batch_data = torch.load(os.path.join(output_path, batch_file), map_location=torch.device('cpu'))
        protein_names.extend(batch_data["protein_names"])
        seqs.extend(batch_data["seqs"])
        chain_id_res.extend(batch_data["chain_id_res"])
        enc_tokens.extend(batch_data["enc_tokens"])
        seq_features.extend(batch_data["seq_features"])
        coor_features.extend(batch_data["coor_features"])
        interface_atoms.extend(batch_data["interface_atoms"])
        affinity.extend(batch_data["affinity"])
        interaction_type.extend(batch_data["interaction_type"])
        interaction_matrix.extend(batch_data["interaction_matrix"])
        res_mass_centor.extend(batch_data["res_mass_centor"])
        hetatm_features.extend(batch_data["hetatm_features"])

    return {
        "protein_names": protein_names, "seqs": seqs, "chain_id_res": chain_id_res,
        "enc_tokens": enc_tokens, "seq_features": seq_features, "coor_features": coor_features,
        "interface_atoms": interface_atoms, "affinity": affinity, "interaction_type": interaction_type,
        "interaction_matrix": interaction_matrix, "res_mass_centor": res_mass_centor,
        "hetatm_features": hetatm_features
    }

class MyDataSet(Data.Dataset):
    def __init__(self, data_dict):
        super(MyDataSet, self).__init__()
        self.d = data_dict
    def __len__(self):
        return len(self.d["enc_tokens"])
    def __getitem__(self, idx):
        return (self.d["protein_names"][idx], self.d["chain_id_res"][idx], self.d["enc_tokens"][idx], 
                self.d["seq_features"][idx], self.d["coor_features"][idx], self.d["interface_atoms"][idx],
                self.d["affinity"][idx], self.d["seqs"][idx], self.d["interaction_type"][idx],
                self.d["interaction_matrix"][idx], self.d["res_mass_centor"][idx], self.d["hetatm_features"][idx])

def collate_fn(batch):
    protein_names = [item[0] for item in batch]
    chain_id_res = [item[1] for item in batch]
    enc_tokens = torch.stack([item[2] for item in batch])
    seq_features = torch.stack([item[3] for item in batch])
    coor_features = torch.stack([item[4] for item in batch])
    interface_atoms = torch.stack([item[5] for item in batch])
    affinity = torch.stack([item[6] for item in batch])
    seqs = [item[7] for item in batch]
    interaction_type = torch.stack([item[8] for item in batch])
    interaction_matrix = torch.stack([item[9] for item in batch])
    res_mass_centor = torch.stack([item[10] for item in batch])
    hetatm_features = torch.stack([item[11] for item in batch])
    return protein_names, chain_id_res, enc_tokens, seq_features, coor_features, \
           interface_atoms, affinity, seqs, interaction_type, interaction_matrix, res_mass_centor, hetatm_features

def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for it, batch_data in enumerate(loader):
        enc_tokens, seq_features = batch_data[2].type(torch.int64).to(device), batch_data[3].to(device)
        coor_features, interface_atoms, affinity = batch_data[4].to(device), batch_data[5].to(device), batch_data[6].to(device)
        interaction_type = batch_data[8].type(torch.int32).to(device)
        interaction_matrix = batch_data[9].type(torch.int32).to(device)
        res_mass_centor = batch_data[10].to(device)
        hetatm_features = batch_data[11].type(torch.float).to(device)

        optimizer.zero_grad()
        outputs = model(enc_tokens, seq_features, coor_features, hetatm_features, interface_atoms, 
                        interaction_type, interaction_matrix, res_mass_centor, batch_data[7], batch_data[0], batch_data[1])
        loss = criterion(outputs.view(-1), affinity)
        epoch_loss += loss.item()
        
        if it % 10 == 0:
            torch.cuda.empty_cache()
            
        loss.backward()
        optimizer.step()
    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    output_list, affinity_list = [], []
    with torch.no_grad():
        for it, batch_data in enumerate(loader):
            enc_tokens, seq_features = batch_data[2].type(torch.int64).to(device), batch_data[3].to(device)
            coor_features, interface_atoms, affinity = batch_data[4].to(device), batch_data[5].to(device), batch_data[6].to(device)
            interaction_type = batch_data[8].type(torch.int64).to(device)
            interaction_matrix = batch_data[9].type(torch.int32).to(device)
            res_mass_centor = batch_data[10].to(device)
            hetatm_features = batch_data[11].type(torch.float).to(device)

            val_outputs = model(enc_tokens, seq_features, coor_features, hetatm_features, interface_atoms,
                                interaction_type, interaction_matrix, res_mass_centor, batch_data[7], batch_data[0], batch_data[1])
            if it % 10 == 0:
                torch.cuda.empty_cache()
                
            output_list.append(val_outputs.view(-1))
            affinity_list.append(affinity)
            loss = criterion(val_outputs.view(-1), affinity)
            epoch_loss += loss.item()

        output_all = torch.cat(output_list, dim=0)
        affinity_all = torch.cat(affinity_list, dim=0)
    return epoch_loss / len(loader), output_all, affinity_all

# ==========================================
# Main 5-Seed Loop with Perfect Model Tracking
# ==========================================
if __name__ == '__main__':
    seed_folders = ['seed_0', 'seed_1', 'seed_42', 'seed_142', 'seed_4242']
    final_results = {'Rp': [], 'Rs': [], 'RMSE': [], 'MAE': []}
    
    base_model_dir = 'runs/5seeds_results'
    os.makedirs(base_model_dir, exist_ok=True)

    for seed in seed_folders:
        print("\n" + "="*50)
        print(f" Processing {seed} ...")
        print("="*50)

        temp_batch_dir = f"./batchs_{seed}"
        
        print(f"\n [1/3] Generating Data for {seed} (This takes a few minutes)...")
        generate_cmd = (
            f"python generate_batch.py "
            f"--data ./checked_data "
            f"--gpu_path /root/autodl-tmp/preprocess_gpu_data "
            f"--batch_path {temp_batch_dir} "
            f"--csv_dir /root/autodl-tmp/Graphomer_Test/random_seeds_splits/{seed}"
        )
        subprocess.run(generate_cmd, shell=True, check=True)

        seed_save_dir = os.path.join(base_model_dir, seed)
        os.makedirs(seed_save_dir, exist_ok=True)
        model_save_path = os.path.join(seed_save_dir, 'model_best_weights.pth')
        log_file = open(os.path.join(seed_save_dir, 'training_log.txt'), 'w')

        print(f"\n [2/3] Loading Data into High-Speed RAM for {seed}...")
        train_data_dict = load_batches_from_disk(f"{temp_batch_dir}/train")
        val_data_dict = load_batches_from_disk(f"{temp_batch_dir}/val")
        test_data_dict = load_batches_from_disk(f"{temp_batch_dir}/test")

        train_loader = Data.DataLoader(MyDataSet(train_data_dict), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = Data.DataLoader(MyDataSet(val_data_dict), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = Data.DataLoader(MyDataSet(test_data_dict), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        config = model_final.Config(
            pro_vocab_size=len(batch_converter.alphabet.all_toks), device=device, pro_len=pro_len,  
            d_embed=d_embed, d_ff=d_ff, d_k=d_k, d_v=d_v, n_layers_en=n_layers_en, n_heads=n_heads
        )
        model = model_final.Transformer(config).to(device)
        
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=9e-4, betas=(0.9, 0.98), eps=1e-09, weight_decay=2e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.93)

        best_valid_r = -float('inf') 
        train_losses, val_losses = [], []

        print(f"\n Start training {seed}...")
        for epoch in range(1, epochs + 1):
            t_loss = train(model, train_loader, optimizer, criterion)
            v_loss, val_preds, val_trues = evaluate(model, val_loader, criterion)
            
            val_preds_np = val_preds.cpu().numpy()
            val_trues_np = val_trues.cpu().numpy()
            try:
                r_val, _ = pearsonr(val_trues_np, val_preds_np)
            except:
                r_val = 0.0
            
            scheduler.step()
            train_losses.append(t_loss)
            val_losses.append(v_loss)

            current_lr = optimizer.param_groups[-1]['lr']
            log_str = f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Train L1: {t_loss:.4f} | Val L1: {v_loss:.4f} | Val R: {r_val:.4f}"
            print(log_str)
            print(log_str, file=log_file)

            if r_val > best_valid_r:
                best_valid_r = r_val
                torch.save(model.state_dict(), model_save_path)

        print(f"\nLoading the BEST model (by Val Rp) for {seed} Test...")
        
        best_model = model_final.Transformer(config).to(device)
        best_model.load_state_dict(torch.load(model_save_path))
        best_model.eval()

        _, te_output, te_affinity = evaluate(best_model, test_loader, criterion)

        all_preds = te_output.cpu().numpy()
        all_trues = te_affinity.cpu().numpy()

        te_rp, _ = pearsonr(all_trues, all_preds)
        te_rs, _ = spearmanr(all_trues, all_preds)
        te_rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
        te_mae = mean_absolute_error(all_trues, all_preds)

        test_score_text = (
            f"\n=== Test Score for {seed} ===\n"
            f"Rp: {te_rp:.4f} | Rs: {te_rs:.4f} | RMSE: {te_rmse:.4f} | MAE: {te_mae:.4f}\n"
            "==============================="
        )
        print(test_score_text)
        print(test_score_text, file=log_file)
        log_file.close()

        final_results['Rp'].append(te_rp)
        final_results['Rs'].append(te_rs)
        final_results['RMSE'].append(te_rmse)
        final_results['MAE'].append(te_mae)

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss (L1)')
        plt.plot(val_losses, label='Validation Loss (L1)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curve ({seed})')
        plt.legend()
        plt.savefig(os.path.join(seed_save_dir, f'loss_curve_{seed}.png'))
        plt.close('all')
        
        print(f"\n[3/3] Performing Deep Clean for {seed}...")
        shutil.rmtree(temp_batch_dir, ignore_errors=True)
        del train_data_dict, val_data_dict, test_data_dict
        del train_loader, val_loader, test_loader
        del model, best_model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        print("Memory & Disk successfully flushed. Ready for next seed.")

    # ==========================================
    # Final 5-Seed Average Report
    # ==========================================
    report_text = (
        "\n" + "="*50 + "\n"
        " FINAL 5-SEED AVERAGE RESULTS (Graphomer)\n"
        + "="*50 + "\n"
        f"Pearson (Rp):  {np.mean(final_results['Rp']):.4f} ± {np.std(final_results['Rp']):.4f}\n"
        f"Spearman (Rs): {np.mean(final_results['Rs']):.4f} ± {np.std(final_results['Rs']):.4f}\n"
        f"RMSE:          {np.mean(final_results['RMSE']):.4f} ± {np.std(final_results['RMSE']):.4f}\n"
        f"MAE:           {np.mean(final_results['MAE']):.4f} ± {np.std(final_results['MAE']):.4f}\n"
        + "="*50
    )
    print(report_text)
    
    with open(os.path.join(base_model_dir, 'final_average_report.txt'), 'w') as f:
        f.write(report_text)