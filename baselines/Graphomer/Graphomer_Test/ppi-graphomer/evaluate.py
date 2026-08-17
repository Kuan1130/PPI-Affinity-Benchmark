import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchinfo import summary
import esm
# import model_final
import tqdm
import pandas as pd
import argparse


# This code predicts the affinity of a batch of proteins and calculates the correlation coefficient with the standard values. The specific predicted values can be viewed in the evaluate.csv file.




# Check if CUDA is available

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# prefix="runs/result/cchar"


# Parameters
# epochs = 200
pro_len = 2000  # Encoder max sequence length
d_embed = 97  # Embedding Size
d_ff = 128  # FeedForward dimension
d_k = d_v = 32  # Dimension of K(=Q), V
n_layers_en = 2  # Number of Encoder layers
n_heads = 8  # Number of heads in Multi-Head Attention
batch_size = 2
n_fold = 5  # Number of folds for cross-validation

# Data Processing
print('数据处理...')
_, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()


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
    seq_features_pretrain = []
    mpnn_features = []
    batch_files = sorted([f for f in os.listdir(output_path) if f.startswith("batch_") and f.endswith(".pt")])
    # if len(batch_files)>80:
    #     batch_files =batch_files[0:90]

    # batch_files =batch_files[0:3]
    for batch_file in tqdm.tqdm(batch_files):
        batch_data = torch.load(os.path.join(output_path, batch_file))
        protein_names.extend(batch_data["protein_names"])
        seqs.extend(batch_data["seqs"])
        chain_id_res.extend(batch_data["chain_id_res"])
        enc_tokens.append(torch.stack(batch_data["enc_tokens"]))
        seq_features.append(torch.stack(batch_data["seq_features"]))
        coor_features.append(torch.stack(batch_data["coor_features"]))
        interface_atoms.append(torch.stack(batch_data["interface_atoms"]))
        # affinity.append(torch.stack(batch_data["affinity"]))
        affinity.append(torch.tensor(batch_data["affinity"]))

        interaction_type.append(torch.stack(batch_data["interaction_type"]))
        interaction_matrix.append(torch.stack(batch_data["interaction_matrix"]))
        res_mass_centor.append(torch.stack(batch_data["res_mass_centor"]))
        hetatm_features.append(torch.stack(batch_data["hetatm_features"]))
    # 合并所有批次的数据
    enc_tokens = torch.cat(enc_tokens, dim=0)
    seq_features = torch.cat(seq_features, dim=0)
    coor_features = torch.cat(coor_features, dim=0)
    interface_atoms = torch.cat(interface_atoms, dim=0)
    affinity = torch.cat(affinity, dim=0)
    interaction_type = torch.cat(interaction_type, dim=0)
    interaction_matrix = torch.cat(interaction_matrix, dim=0)
    res_mass_centor = torch.cat(res_mass_centor, dim=0)
    hetatm_features = torch.cat(hetatm_features, dim=0)

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

def load_if_add_dict(if_add_path):
    if_add_dict = {}
    if_add_path_list = os.listdir(if_add_path)
    for if_add_file in if_add_path_list:
        if_add_dict.update(np.load(os.path.join(if_add_path, if_add_file), allow_pickle=True).item())
    return if_add_dict

def evaluate(model, loader, criterion,save_dir):
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
            # print(protein_names_val)
            r=torch.corrcoef(torch.stack((val_outputs.view(-1), affinity_val)))[0,1]
            # if affinity_val.item()>5 and affinity_val.item()<15:
            output_list.append(val_outputs.view(-1))
            affinity_list.append(affinity_val)
            protein_names_val_list.extend(protein_names_val)
            
            loss = criterion(val_outputs.view(-1), affinity_val)
            epoch_loss += loss.item()
            print(it)
            print(f'Evaluate {it:4d} Loss: {loss:.4f} | Iter PPL: {math.exp(loss):7.4f}')
            print(f'Evaluate {it:4d} R: {r:.4f} ')
        output_all=torch.cat(output_list,dim=0)
        affinity_all=torch.cat(affinity_list,dim=0)
        df=pd.DataFrame(data=np.vstack((output_all.cpu().numpy(),affinity_all.cpu().numpy())).T,index=protein_names_val_list)
        df.columns=["predict","label"]
        df.to_csv(save_dir+"/evaluate.csv")

    return epoch_loss / len(loader),output_all,affinity_all


# config = model_final.Config(
#     pro_vocab_size=len(batch_converter.alphabet.all_toks), device=device, pro_len=pro_len,  
#     d_embed=d_embed, d_ff=d_ff, d_k=d_k, d_v=d_v, n_layers_en=n_layers_en, n_heads=n_heads
# )






if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_path", '-b', default="./data/batch/default/", type=str, help="batch data path")
    parser.add_argument("--result_path", '-r', default="./result/default/", type=str, help="batch data path")


    # assign your pdbs here
    # pdb_folder = "./data_final/pdb/Accepted"
    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)


    ff = open(args.result_path + 'evaluate.txt', 'w')

    for fold in range(n_fold):
        # print(f"第 {fold} 折：")
        if fold != 0:
            continue
        fold=0
        print("loading val data")
        val_output_path=args.batch_path
        # val_output_path="/public/mxp/xiejun/py_project/PPI_affinity/data_final/batchs/test_if
        val_data_dict = load_batches_from_disk(val_output_path)

        print(len(val_data_dict["protein_names"]))
        
        val_loader = Data.DataLoader(
            MyDataSet(val_data_dict["protein_names"], val_data_dict["chain_id_res"], val_data_dict["enc_tokens"], 
                    val_data_dict["seq_features"], val_data_dict["coor_features"], 
                    val_data_dict["interface_atoms"],val_data_dict["affinity"],val_data_dict["seqs"],
                    val_data_dict["interaction_type"],val_data_dict["interaction_matrix"],val_data_dict["res_mass_centor"],
                    val_data_dict["hetatm_features"]), 
                    
            batch_size=batch_size, shuffle=False,collate_fn=collate_fn
        )

        transformer_model = torch.load('./model/model_0.pth',map_location=device)
        
        summary(transformer_model)
        
        criterion = nn.L1Loss()

        valid_loss ,output_all,affinity_all= evaluate(transformer_model, val_loader, criterion,args.result_path)
        loss_all = criterion(output_all.view(-1), affinity_all)
        r_all=torch.corrcoef(torch.stack((output_all.view(-1), affinity_all)))[0,1]

        #设置绘图风格
        plt.style.use('ggplot')
        #处理中文乱码
        # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

        plt.rcParams['axes.unicode_minus']=False

        plt.scatter(x = output_all.view(-1).cpu(), # 指定散点图的x轴数据
                    y = affinity_all.cpu(), # 指定散点图的y轴数据
                    color = 'steelblue' # 指定散点图中点的颜色
                )

        plt.ylim(2, 23)
        plt.xlim(2, 23)
        plt.xlabel('predict')
        plt.ylabel('label')
        plt.plot([2, 23], [2, 23], color='red', linestyle='--', label='y=x')
        plt.legend(title=f'r= {r_all:.2f}')
        # 显示图形
        plt.show()
        result_path=args.result_path
        plt.savefig(result_path+"/relative_"+str(fold)+".png")
        
        print(f'Evaluate all Loss: {loss_all:.4f} | Iter PPL: {math.exp(loss_all):7.4f}')
        print(f'Evaluate all Loss: {loss_all:.4f} | Iter PPL: {math.exp(loss_all):7.4f}', file=ff)
        print(f'Evaluate all R: {r_all:.4f} ')
        print(f'Evaluate all R: {r_all:.4f} ', file=ff)
        torch.cuda.empty_cache()
