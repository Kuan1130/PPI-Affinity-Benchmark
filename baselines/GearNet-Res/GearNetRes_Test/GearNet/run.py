import os
import re
import subprocess
import numpy as np
import shutil

base_yaml = "config/downstream/temp_seed_0.yaml"
base_split_dir = "/root/autodl-tmp/GearNetRes_Test/GearNet/random_seeds_splits"
seeds = ['seed_0', 'seed_1', 'seed_42', 'seed_142', 'seed_4242']
ckpt_path = "/root/autodl-tmp/mc_gearnet_edge.pth"

final_results = {'Rp': [], 'Rs': [], 'RMSE': [], 'MAE': []}

with open(base_yaml, 'r') as f:
    original_yaml = f.read()

for seed in seeds:
    print(f"\n{'='*50}")
    print(f" Running seed {seed} ...")
    print(f"{'='*50}\n")
    
    new_yaml = re.sub(r"train_csv:\s*.*", f"train_csv: {os.path.join(base_split_dir, seed, 'train_split.csv')}", original_yaml)
    new_yaml = re.sub(r"val_csv:\s*.*", f"val_csv: {os.path.join(base_split_dir, seed, 'val_split.csv')}", new_yaml)
    new_yaml = re.sub(r"test_csv:\s*.*", f"test_csv: {os.path.join(base_split_dir, seed, 'test_split.csv')}", new_yaml)
    new_yaml = re.sub(r"output_dir:\s*.*", f"output_dir: ./output/{seed}", new_yaml)
    
    temp_yaml = f"temp_{seed}.yaml"
    with open(temp_yaml, 'w') as f:
        f.write(new_yaml)
        
    cmd = ["python", "-u", "script/downstream.py", "-c", temp_yaml, "--gpus", "[0]"]
    if os.path.exists(ckpt_path):
        cmd.extend(["--ckpt", ckpt_path])
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    rp, rs, rmse, mae = None, None, None, None
    in_test_block = False 
    
    for line in process.stdout:
        print(line, end="") 
        
        if "Evaluate on test" in line:
            in_test_block = True
        elif "Evaluate on valid" in line or "Epoch" in line:
            in_test_block = False

        if in_test_block:
            if "pearsonr [proaffinity_label]:" in line:
                rp = float(line.split(":")[-1].strip())
            elif "spearmanr [proaffinity_label]:" in line:
                rs = float(line.split(":")[-1].strip())
            elif "root mean squared error [proaffinity_label]:" in line:
                rmse = float(line.split(":")[-1].strip())
            elif "mean absolute error [proaffinity_label]:" in line:
                mae = float(line.split(":")[-1].strip())
            
    process.wait()
    
    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)
        
    if rp is not None and rs is not None:
        print(f"\n [Test Score for {seed}]")
        print(f"Rp: {rp:.4f} | Rs: {rs:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        final_results['Rp'].append(rp)
        final_results['Rs'].append(rs)
        final_results['RMSE'].append(rmse)
        final_results['MAE'].append(mae)
    else:
        print(f"\n [Error] Seed :{seed}")
        
    target_del_dir = f"./output/{seed}"
    if os.path.exists(target_del_dir):
        shutil.rmtree(target_del_dir)
        print(f"Remove {seed} weight\n")

print("\n" + "="*50)
print(" FINAL 5-SEED AVERAGE RESULTS (GearNet) ")
print("="*50)
if len(final_results['Rp']) > 0:
    print(f"Pearson (Rp):  {np.mean(final_results['Rp']):.4f} ± {np.std(final_results['Rp']):.4f}")
    print(f"Spearman (Rs): {np.mean(final_results['Rs']):.4f} ± {np.std(final_results['Rs']):.4f}")
    print(f"RMSE:          {np.mean(final_results['RMSE']):.4f} ± {np.std(final_results['RMSE']):.4f}")
    print(f"MAE:           {np.mean(final_results['MAE']):.4f} ± {np.std(final_results['MAE']):.4f}")
else:
    print("Cannot get any result. Process failed.")
print("="*50)