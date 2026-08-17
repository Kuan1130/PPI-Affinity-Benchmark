import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt

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
# device = torch.device('cpu')

save_dir = 'runs/run_ffinal/attempt1/'

os.makedirs(save_dir, exist_ok=True)

# Parameters
epochs = 30
pro_len = 2000  # Encoder max sequence length
d_embed = 64  # Embedding Size
d_ff = 128  # FeedForward dimension
d_k = d_v = 32  # Dimension of K(=Q), V
n_layers_en = 1  # Number of Encoder layers
n_heads = 8  # Number of heads in Multi-Head Attention
batch_size = 8
n_fold = 5  # Number of folds for cross-validation




# Data Processing
print('数据处理...')
_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

batch_converter = alphabet.get_batch_converter()

def chunked_cat(tensors, dim=0, chunk_size=10):
    chunks = [torch.cat(tensors[i:i + chunk_size], dim=dim) for i in range(0, len(tensors), chunk_size)]
    return torch.cat(chunks, dim=dim)

'''
def load_batches_from_disk(output_path):
    protein_names = []
    seqs = []
    chain_id_res = []
    enc_tokens = []
    seq_features = []
    coor_features = []
    interface_atoms = []
    affinity = []
    interaction_type = []
    interaction_matrix = []
    res_mass_centor = []
    hetatm_features = []

    batch_files = sorted([f for f in os.listdir(output_path) if f.startswith("batch_") and f.endswith(".pt")])
    # if len(batch_files)>95:
    #     batch_files =batch_files[0:3]

    # batch_files =batch_files[0:3]
    for batch_file in tqdm.tqdm(batch_files):
        batch_data = torch.load(os.path.join(output_path, batch_file),map_location=torch.device('cpu'))
        protein_names.extend(batch_data["protein_names"])
        seqs.extend(batch_data["seqs"])
        chain_id_res.extend(batch_data["chain_id_res"])
        enc_tokens.append(torch.stack(batch_data["enc_tokens"]))
        seq_features.append(torch.stack(batch_data["seq_features"]))
        coor_features.append(torch.stack(batch_data["coor_features"]))
        interface_atoms.append(torch.stack(batch_data["interface_atoms"]))
        affinity.append(torch.stack(batch_data["affinity"]))
        interaction_type.append(torch.stack(batch_data["interaction_type"]))
        interaction_matrix.append(torch.stack(batch_data["interaction_matrix"]))
        res_mass_centor.append(torch.stack(batch_data["res_mass_centor"]))
        hetatm_features.append(torch.stack(batch_data["hetatm_features"]))

    # 合并所有批次的数据
    enc_tokens = chunked_cat(enc_tokens, dim=0)
    seq_features = chunked_cat(seq_features, dim=0)
    coor_features = chunked_cat(coor_features, dim=0)
    interface_atoms = chunked_cat(interface_atoms, dim=0)
    affinity = chunked_cat(affinity, dim=0)
    interaction_type = chunked_cat(interaction_type, dim=0)
    interaction_matrix = chunked_cat(interaction_matrix, dim=0)
    res_mass_centor = chunked_cat(res_mass_centor, dim=0)
    hetatm_features = chunked_cat(hetatm_features, dim=0)
    return {
        "protein_names": protein_names,
        "seqs": seqs,
        "chain_id_res": chain_id_res,
        "enc_tokens": enc_tokens,
        "seq_features": seq_features,
        "coor_features": coor_features,
        "interface_atoms": interface_atoms,
        "affinity": affinity,
        "interaction_type": interaction_type,
        "interaction_matrix": interaction_matrix,
        "res_mass_centor": res_mass_centor,
        "hetatm_features": hetatm_features
    }
'''

def load_batches_from_disk(output_path):
    protein_names = []
    seqs = []
    chain_id_res = []
    enc_tokens = []
    seq_features = []
    coor_features = []
    interface_atoms = []
    affinity = []
    interaction_type = []
    interaction_matrix = []
    res_mass_centor = []
    hetatm_features = []

    batch_files = sorted([f for f in os.listdir(output_path) if f.startswith("batch_") and f.endswith(".pt")])
    for batch_file in tqdm.tqdm(batch_files):
        batch_data = torch.load(os.path.join(output_path, batch_file), map_location=torch.device('cpu'))
        protein_names.extend(batch_data["protein_names"])
        seqs.extend(batch_data["seqs"])
        chain_id_res.extend(batch_data["chain_id_res"])
        
        # Not to usetorch.stack, extend to pure Python List
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
        "protein_names": protein_names,
        "seqs": seqs,
        "chain_id_res": chain_id_res,
        "enc_tokens": enc_tokens,
        "seq_features": seq_features,
        "coor_features": coor_features,
        "interface_atoms": interface_atoms,
        "affinity": affinity,
        "interaction_type": interaction_type,
        "interaction_matrix": interaction_matrix,
        "res_mass_centor": res_mass_centor,
        "hetatm_features": hetatm_features
    }


class MyDataSet(Data.Dataset):
    def __init__(self, protein_names, chain_id_res, enc_tokens, seq_features, coor_features, interface_atoms,affinity,seqs,
                 interaction_type,interaction_matrix,res_mass_centor,hetatm_features):
        super(MyDataSet, self).__init__()
        self.protein_names = protein_names
        self.chain_id_res = chain_id_res
        self.enc_tokens = enc_tokens
        self.seq_features = seq_features
        self.coor_features = coor_features
        self.interface_atoms = interface_atoms
        self.affinity = affinity
        self.seqs = seqs
        self.interaction_type=interaction_type
        self.interaction_matrix=interaction_matrix
        self.res_mass_centor=res_mass_centor
        self.hetatm_features=hetatm_features

    def __len__(self):
        return len(self.enc_tokens)

    def __getitem__(self, idx):
        return (self.protein_names[idx], self.chain_id_res[idx], self.enc_tokens[idx], self.seq_features[idx],
                self.coor_features[idx], self.interface_atoms[idx],self.affinity[idx],self.seqs[idx],
                self.interaction_type[idx],self.interaction_matrix[idx],self.res_mass_centor[idx],self.hetatm_features[idx])

def collate_fn(batch):
    # 因为token_list是一个变长的数据，所以需要用一个list来装这个batch的token_list

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


    return protein_names,chain_id_res,enc_tokens,seq_features,coor_features,\
    interface_atoms,affinity,seqs,interaction_type,interaction_matrix,res_mass_centor,hetatm_features

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for it, (protein_names, chain_id_res, enc_tokens, seq_features, coor_features, 
            interface_atoms,affinity,seqs,interaction_type,interaction_matrix,res_mass_centor,hetatm_features) in enumerate(loader):
        enc_tokens,seq_features = enc_tokens.type(torch.int64).to(device),seq_features.to(device)
        coor_features= coor_features.to(device)
        affinity=affinity.to(device)
        interface_atoms= interface_atoms.to(device)
        interaction_type,interaction_matrix,res_mass_centor=interaction_type.type(torch.int32).to(device),interaction_matrix.type(torch.int32).to(device),res_mass_centor.to(device)
        hetatm_features =hetatm_features.type(torch.float).to(device)

        optimizer.zero_grad()
        outputs= model(enc_tokens, seq_features, coor_features, hetatm_features,interface_atoms,interaction_type,interaction_matrix,res_mass_centor,seqs,\
                       protein_names,chain_id_res)
        loss = criterion(outputs.view(-1), affinity)

        # l2_lambda = 1e-4
        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + l2_lambda * l2_norm
        epoch_loss += loss.item()
        if it%10==0:
            torch.cuda.empty_cache()

        print(f'Iter {it:4d} Loss: {loss:.4f} | Iter PPL: {math.exp(loss):7.4f}')
        loss.backward()
        optimizer.step()

    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        output_list=[]
        affinity_list=[]
        protein_names_val_list=[]

        for it, (protein_names_val, chain_id_res_val, enc_tokens_val, seq_features_val, 
                 coor_features_val, interface_atoms_val,affinity_val,seqs_val,
                 interaction_type_val,interaction_matrix_val,res_mass_centor_val,hetatm_features_val) in enumerate(loader):
            enc_tokens_val, seq_features_val = enc_tokens_val.type(torch.int64).to(device), seq_features_val.to(device)
            coor_features_val= coor_features_val.to(device)
            affinity_val=affinity_val.to(device)
            interface_atoms_val=interface_atoms_val.to(device)
            interaction_type_val,interaction_matrix_val,res_mass_centor_val=interaction_type_val.type(torch.int64).to(device),interaction_matrix_val.type(torch.int32).to(device),res_mass_centor_val.to(device)
            hetatm_features_val=hetatm_features_val.type(torch.float).to(device)

            val_outputs= model(enc_tokens_val, seq_features_val, coor_features_val, hetatm_features_val,interface_atoms_val,\
                               interaction_type_val,interaction_matrix_val,res_mass_centor_val,seqs_val,protein_names_val,chain_id_res_val)
            if it%10==0:
                torch.cuda.empty_cache()
            r=torch.corrcoef(torch.stack((val_outputs.view(-1), affinity_val)))[0,1]
            output_list.append(val_outputs.view(-1))
            affinity_list.append(affinity_val)
            protein_names_val_list.extend(protein_names_val)
            
            loss = criterion(val_outputs.view(-1), affinity_val)
            epoch_loss += loss.item()

            print(f'Evaluate {it:4d} Loss: {loss:.4f} | Iter PPL: {math.exp(loss):7.4f}')
            print(f'Evaluate {it:4d} R: {r:.4f} ')
        output_all=torch.cat(output_list,dim=0)
        affinity_all=torch.cat(affinity_list,dim=0)
        df=pd.DataFrame(data=np.vstack((output_all.cpu().numpy(),affinity_all.cpu().numpy())).T,index=protein_names_val_list)
        df.to_csv('./'+save_dir+"/trains.csv")

    return epoch_loss / len(loader),output_all,affinity_all

fold = 0

print(f"第 {fold} 折：")
ff = open(save_dir + 'train'+str(fold)+'.txt', 'w')

config = model_final.Config(
    pro_vocab_size=len(batch_converter.alphabet.all_toks), device=device, pro_len=pro_len,  
    d_embed=d_embed, d_ff=d_ff, d_k=d_k, d_v=d_v, n_layers_en=n_layers_en, n_heads=n_heads
)
transformer_model = model_final.Transformer(config).to(device)




# transformer_model = torch.load('/public/mxp/xiejun/py_project/PPI_affinity/runs/run_11_final/attempt9_nompnn_en2/model_8_0.pth')
summary_txt=summary(transformer_model)

print(str(summary_txt), file=ff)

criterion = nn.L1Loss()
optimizer = optim.Adam(transformer_model.parameters(), lr=9e-4, betas=(0.9, 0.98), eps=1e-09,weight_decay=2e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.93)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, 
#             6, 
#             eta_min=1e-5, 
#             last_epoch=-1)

# print("loading train data")
# train_output_path="/public/mxp/xiejun/py_project/PPI_affinity/data_final/batchs/5fold_"+str(fold)+"_train"
# train_data_dict = load_batches_from_disk(train_output_path)
# print("loading val data")
# val_output_path="/public/mxp/xiejun/py_project/PPI_affinity/data_final/batchs/5fold_"+str(fold)+"_val"
# val_data_dict = load_batches_from_disk(val_output_path)
# print("construct data")

print("loading train data")
train_output_path="./batchs/train"
train_data_dict = load_batches_from_disk(train_output_path)


print("loading val data")
val_output_path="./batchs/val"
val_data_dict = load_batches_from_disk(val_output_path)

print("loading test data")
test_output_path="./batchs/test"
test_data_dict = load_batches_from_disk(test_output_path)

print("construct data")


loader = Data.DataLoader(
    MyDataSet(train_data_dict["protein_names"], train_data_dict["chain_id_res"], train_data_dict["enc_tokens"], 
              train_data_dict["seq_features"], train_data_dict["coor_features"], 
              train_data_dict["interface_atoms"],train_data_dict["affinity"],train_data_dict["seqs"],
              train_data_dict["interaction_type"],train_data_dict["interaction_matrix"],train_data_dict["res_mass_centor"],
              train_data_dict["hetatm_features"]
), 
    batch_size=batch_size, shuffle=True,collate_fn=collate_fn
)

val_loader = Data.DataLoader(
    MyDataSet(val_data_dict["protein_names"], val_data_dict["chain_id_res"], val_data_dict["enc_tokens"], 
              val_data_dict["seq_features"], val_data_dict["coor_features"], 
              val_data_dict["interface_atoms"],val_data_dict["affinity"],val_data_dict["seqs"],
              val_data_dict["interaction_type"],val_data_dict["interaction_matrix"],val_data_dict["res_mass_centor"],
              val_data_dict["hetatm_features"]), 

    batch_size=batch_size, shuffle=False,collate_fn=collate_fn
)

test_loader = Data.DataLoader(
    MyDataSet(test_data_dict["protein_names"], test_data_dict["chain_id_res"], test_data_dict["enc_tokens"], 
              test_data_dict["seq_features"], test_data_dict["coor_features"], 
              test_data_dict["interface_atoms"],test_data_dict["affinity"],test_data_dict["seqs"],
              test_data_dict["interaction_type"],test_data_dict["interaction_matrix"],test_data_dict["res_mass_centor"],
              test_data_dict["hetatm_features"]), 
    batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

ff.write("")
ff.write(f"d_embed: {d_embed}\n")
ff.write(f"d_ff: {d_ff}\n")
ff.write(f"d_k: {d_k}\n")
ff.write(f"d_v: {d_v}\n")
ff.write(f"n_layers_en: {n_layers_en}\n")
ff.write(f"n_heads: {n_heads}\n")
ff.write(f"batch_size: {batch_size}\n")
ff.write(f"optimizer: Adam\n")

loss_epoch = []
val_loss_epoch = []
best_valid_loss = 10
model_best = transformer_model
epoch_best = 0

print('模型训练...')
lr_list = []

for epoch in range(1, epochs + 1):
    start_time = time.time()

    train_loss = train(transformer_model, loader, optimizer, criterion)


    valid_loss,output_all,affinity_all = evaluate(transformer_model, val_loader, criterion)
    loss_all_val = criterion(output_all.view(-1), affinity_all)
    r_all=torch.corrcoef(torch.stack((output_all.view(-1), affinity_all)))[0,1]
    scheduler.step()
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    loss_epoch.append(train_loss)
    val_loss_epoch.append(valid_loss)

    cur_lr = optimizer.param_groups[-1]['lr']
    print(f"learning rate: {cur_lr}")
    lr_list.append(cur_lr)

    if epoch % 1 == 0:
        torch.save(transformer_model, save_dir + f'model_{epoch}_{fold}.pth')

    if loss_all_val < best_valid_loss:
        model_best = transformer_model
        epoch_best = epoch
        best_valid_loss = loss_all_val

    print(f'Fold: {fold:03} | Epoch: {epoch:04} | Time: {epoch_mins}m {epoch_secs}s', file=ff)
    print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}', file=ff)
    print(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {math.exp(valid_loss):7.4f}', file=ff)
    print(f'\tEvaluate all R: {r_all:.4f} ', file=ff)
    print(f'\tVal loss all : {loss_all_val:.4f} ', file=ff)

    print(f'Fold: {fold:03} | Epoch: {epoch:04} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}')
    print(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {math.exp(valid_loss):7.4f}')
    print(f'\tEvaluate all R: {r_all:.4f} ')
    print(f'\tVal loss all : {loss_all_val:.4f} ')


    fig1 = plt.figure()
    plt.plot(range(len(lr_list)), lr_list)
    plt.savefig(save_dir + f'_{fold}learningRate.jpg')

    fig2 = plt.figure()
    x = range(1, len(loss_epoch) + 1)
    plt.plot(x, loss_epoch, 'b')
    plt.plot(x, val_loss_epoch, 'r')
    plt.legend(["train", "val"])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_dir + f'_{fold}loss.jpg')
    plt.close("all")
    torch.save(model_best, save_dir + f'model_best_{fold}.pth')
    
print("\n Loading the best model for final Test...")
model_best = torch.load(save_dir + f'model_best_{fold}.pth')

te_loss, te_output, te_affinity = evaluate(model_best, test_loader, criterion)

all_preds = te_output.cpu().numpy()
all_trues = te_affinity.cpu().numpy()

te_rp, _ = pearsonr(all_trues, all_preds)
te_rs, _ = spearmanr(all_trues, all_preds)
te_rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
te_mae = mean_absolute_error(all_trues, all_preds)

test_score_text = (
    "\n" + "="*40 + "\n"
    "Test Set Score (Graphomer)\n"
    + "="*40 + "\n"
    f"Pearson (Rp):  {te_rp:.4f}  \n"
    f"Spearman (Rs): {te_rs:.4f}  \n"
    f"RMSE:          {te_rmse:.4f}\n"
    f"MAE:           {te_mae:.4f} \n"
    + "="*40
)
print(test_score_text)
print(test_score_text, file=ff)    

ff.close()