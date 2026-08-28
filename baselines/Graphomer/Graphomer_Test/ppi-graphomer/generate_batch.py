import os
import math
import time
import numpy as np

import torch
import torch.nn.functional as F
import random
import tqdm
import argparse

# This code will further organize the data to facilitate its use in subsequent models.

batch_size = 26
pro_len = 2000  # Encoder max sequence length
n_fold = 5  # Number of folds for cross-validation


import pandas as pd


def load_if_add_dict(if_add_path):
    if_add_dict = {}
    if_add_path_list = os.listdir(if_add_path)
    for if_add_file in if_add_path_list:
        if_add_list=np.load(os.path.join(if_add_path, if_add_file), allow_pickle=True)
        for if_add_dict_single in if_add_list:
            if_add_dict.update(if_add_dict_single)
    return if_add_dict

def process_train_data(train_data, pro_len, batch_size=1000):
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


    for idx, item in tqdm.tqdm(enumerate(train_data)):
        # pdbbind_name_list=np.load("./pdbbind_name_list.npy",allow_pickle=True)
        # if item["protein_name"].lower() not in pdbbind_name_list:
        #     continue
        protein_names.append(item["protein_name"])
        seq_temp=""
        for i in item["sequence"]:
            seq_temp+=i
        if len(seq_temp)>pro_len :
            continue
        # 对于pdbbind和skempi的不同处理
        seqs.append(seq_temp)
        chain_id_res.append(item["chain_id_res"])
        enc_tokens_temp=torch.cat(if_add_dict[item["protein_name"].replace(".pdb","")][2],dim=0).type(torch.int16)
        enc_tokens.append(F.pad(enc_tokens_temp,(0,pro_len-enc_tokens_temp.shape[0])))
        seq_feat_temp=torch.cat(if_add_dict[item["protein_name"].replace(".pdb","")][0],dim=1).squeeze()
        seq_features.append(F.pad(seq_feat_temp,(0,0,0,pro_len-seq_feat_temp.shape[0])))

        coor_feat_temp=torch.cat(if_add_dict[item["protein_name"].replace(".pdb","")][1],dim=0)
        coor_features.append(F.pad(coor_feat_temp,(0,0,0,pro_len-enc_tokens_temp.shape[0])))
        interface_res_matrix=torch.ones((pro_len,pro_len), dtype=torch.bool)
        # 将非界面氨基酸全部遮蔽
        for i in range(len(item["interface_res"])):
            for j in range(len(item["interface_res"][i])):
                if item["interface_res"][i][j]!=-1:
                    interface_res_matrix[i][item["interface_res"][i][j]]=False
        # 将自己同一条链的氨基酸全部遮蔽
        # chain_id_array = np.array(item["chain_id_res"])
        # i_grid, j_grid = np.meshgrid(np.arange(len(item["chain_id_res"])), np.arange(len(item["chain_id_res"])))
        # comparison_matrix = (chain_id_array[i_grid] == chain_id_array[j_grid])
        # interface_res_matrix[:len(item["chain_id_res"]), :len(item["chain_id_res"])] = torch.from_numpy(comparison_matrix)
        interface_atoms.append(interface_res_matrix)
        # 注意后来的系数
        # affinity.append(torch.tensor(iptm_dict[item["protein_name"].replace(".pdb","")[0:-7]]).float())
        affinity.append(torch.tensor(item["affinity"]))
        if_type=torch.tensor(item["interaction_type_matrix"]).type(torch.int16)
        interaction_type.append(F.pad(if_type,(0,pro_len-if_type.shape[0],0,pro_len-if_type.shape[0])))
        if_matrix=torch.tensor(item["interaction_matrix"]).type(torch.int16)
        interaction_matrix.append(F.pad(if_matrix,(0,0,0,pro_len-if_matrix.shape[0],0,pro_len-if_matrix.shape[0])))
        mass_centor=torch.tensor(item["res_mass_centor"])
        res_mass_centor.append(F.pad(mass_centor,(0,0,0,pro_len-mass_centor.shape[0])))
        hetatm_features_single=torch.tensor(np.stack(item["hetatm_features"])).type(torch.float32)
        hetatm_features.append(F.pad(hetatm_features_single,(0,0,0,pro_len-hetatm_features_single.shape[0])))

        if (idx + 1) % batch_size == 0:
            yield {
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
                "hetatm_features":hetatm_features
            }
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
    if protein_names:
        yield {
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
            "hetatm_features":hetatm_features
        }



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", '-d', default="./data/checked_data/default/", type=str, help="checked data path")
    parser.add_argument("--gpu_path", '-g', default="./data/preprocess/gpu/default/", type=str, help="gpu data path")
    parser.add_argument("--batch_path", '-b', default="./data/batchs/", type=str, help="batch data base path")
    
    parser.add_argument(
        "--csv_dir", "-c",
        required=True,
        type=str,
        help="Directory containing one seed's train_split.csv, val_split.csv, and test_split.csv",
    )
    args = parser.parse_args()

    # 1. load data
    data = np.load(args.data + "/checked_cpu_data.npy", allow_pickle=True)
    if_add_dict = load_if_add_dict(args.gpu_path)
    
    # 2. Read csv while define dic
    import pandas as pd
    def get_split_dict(csv_name):
        csv_path = os.path.join(args.csv_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"Cannot find {csv_path}, please ensure dir")
            return {}
        df = pd.read_csv(csv_path)
        # proaffinity_label 
        return {str(row['pdb_code']).lower(): float(row['proaffinity_label']) for _, row in df.iterrows()}

    print("Reading division csv...")
    train_dict = get_split_dict('train_split.csv')
    val_dict = get_split_dict('val_split.csv')
    test_dict = get_split_dict('test_split.csv')

    train_data, val_data, test_data = [], [], []

    # 3. Affinity label
    print("Distributing data...")
    for item in data:
        pdb = item["protein_name"].replace(".pdb", "").lower()
        
        if pdb in train_dict:
            item["affinity"] = train_dict[pdb] 
            train_data.append(item)
        elif pdb in val_dict:
            item["affinity"] = val_dict[pdb]
            val_data.append(item)
        elif pdb in test_dict:
            item["affinity"] = test_dict[pdb]
            test_data.append(item)

    print(f" No of data assigned：Train={len(train_data)} , Val={len(val_data)} , Test={len(test_data)} ")

    # 4. Batch
    splits = [("train", train_data), ("val", val_data), ("test", test_data)]

    for split_name, split_data in splits:
        if len(split_data) == 0: 
            continue
        print(f"\n Batching {split_name} dataset...")
        out_dir = os.path.join(args.batch_path, split_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # ONly shuffle train set
        if split_name == "train":
            random.seed(42)
            random.shuffle(split_data)

        for i, batch in enumerate(process_train_data(split_data, pro_len, batch_size)):
            try:
                torch.save(batch, os.path.join(out_dir, f"batch_{i}.pt"), _use_new_zipfile_serialization=False)
                print(f"Saved {split_name} batch {i}")
            except Exception as e:
                print(f"Error：Batch {i} in {split_name} write in unsuccessful, Skipped: {e}")
                continue

    print("\n Batch done! ")




    
