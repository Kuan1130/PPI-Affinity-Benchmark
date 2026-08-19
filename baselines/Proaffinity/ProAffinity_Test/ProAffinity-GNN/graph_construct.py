from transformers import AutoTokenizer, EsmModel
import torch
import copy
import math
import pickle
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes
from tqdm import tqdm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"current device: {device}")

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.to(device) 
model.eval()

def get_fasta_seq(pdb_tuple):
    fastaA = pdb_tuple[0]
    fastaB = pdb_tuple[1]
    return fastaA, fastaB

path = 'data/graph_construct/inter_graph/'
filenames = os.listdir(path)

def get_distance(x1, y1, z1, x2, y2, z2):
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return distance

def read_y(filename):
    y_dict = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pdb = line.split()[0].lower()
            y = float(line.split()[1])
            y = round(y, 2)
            y_dict.update({pdb:y})
    return y_dict

atom_pair = ['A_A', 'A_C', 'A_OA', 'A_N', 'A_NA', 'A_SA', 'A_HD', 
            'C_C', 'C_OA', 'C_N', 'C_NA', 'C_SA', 'C_HD',
            'OA_OA', 'OA_N', 'OA_NA', 'OA_SA', 'OA_HD',
            'N_N', 'N_NA', 'N_SA', 'N_HD', 
            'NA_NA', 'NA_SA', 'NA_HD',
            'SA_SA', 'SA_HD',
            'HD_HD', 'others']

bin_number = 10
type_number = len(atom_pair)
inter_distance = 15
y_dict = read_y('data/PPIdataindex.txt')

# 確保輸出目錄存在
os.makedirs('data/graph/inter_graph/', exist_ok=True)

for pdb in tqdm(filenames, desc="Constructing Graphs"):
    info = pickle.load(open('data/graph_construct/inter_graph/' + pdb, 'rb'))

    fasta1_list, fasta2_list = get_fasta_seq(info[2])
    output1_list = []
    output2_list = []

    with torch.no_grad(): 
        for fasta in fasta1_list:
            input1 = tokenizer(fasta, return_tensors="pt")
            # 把輸入資料搬到 GPU
            input1 = {k: v.to(device) for k, v in input1.items()} 
            output1 = model(**input1)
            last_hidden_state1 = output1.last_hidden_state
            last_hidden_state1 = torch.squeeze(last_hidden_state1)
            last_hidden_state1 = last_hidden_state1[1:-1]
            output1_list.append(last_hidden_state1.cpu()) 

        for fasta in fasta2_list:
            input2 = tokenizer(fasta, return_tensors="pt")
            # 把輸入資料搬到 GPU
            input2 = {k: v.to(device) for k, v in input2.items()}
            output2 = model(**input2)
            last_hidden_state2 = output2.last_hidden_state
            last_hidden_state2 = torch.squeeze(last_hidden_state2)
            last_hidden_state2 = last_hidden_state2[1:-1]
            output2_list.append(last_hidden_state2.cpu()) 

    x1 = torch.cat(output1_list, 0)
    x2 = torch.cat(output2_list, 0)
    x = torch.cat((x1, x2), 0)

    pairs = info[0]
    edge_index = info[1]

    try:
        edge_feature = []
        for pair in pairs:
            edge_encoding = np.zeros(type_number * bin_number)
            residueA = pair[0]
            residueB = pair[1]

            for atom1 in residueA['atoms']:
                x1, y1, z1 = float(atom1['x']), float(atom1['y']), float(atom1['z'])
                type1 = atom1['pdbqt_type']

                for atom2 in residueB['atoms']:
                    x2, y2, z2 = float(atom2['x']), float(atom2['y']), float(atom2['z'])
                    type2 = atom2['pdbqt_type']

                    dis = get_distance(x1, y1, z1, x2, y2, z2)
                    bin_n = math.ceil(dis / (inter_distance / bin_number))
                    if bin_n > 10: bin_n = 10

                    if type1 + '_' + type2 in atom_pair:
                        pair_type = type1 + '_' + type2       
                    elif type2 + '_' + type1 in atom_pair:
                        pair_type = type2 + '_' + type1
                    else:
                        pair_type = 'others'
                        
                    pair_type_index = atom_pair.index(pair_type)
                    encoding_index = (bin_n - 1) * type_number + pair_type_index
                    edge_encoding[encoding_index] += 1

            edge_feature.append(torch.from_numpy(edge_encoding))                

        edge_feature = torch.stack(edge_feature, 0) 
        edge_feature = torch.cat((edge_feature, edge_feature), 0)

    except Exception as e:
        print(f"\n{pdb} 發生錯誤: {e}")
        continue

    num_nodes = x.size(0)
    
    # Renumber edge_index, and send node_mask back
    clean_edge_index, clean_edge_attr, node_mask = remove_isolated_nodes(
        edge_index=edge_index, 
        edge_attr=edge_feature, 
        num_nodes=num_nodes
    )
    
    # remove useless features
    clean_x = x[node_mask]
    
    data = Data(x=clean_x, edge_index=clean_edge_index, edge_attr=clean_edge_attr, y=y_dict[pdb.split('.')[0]])
    save_path = 'data/graph/inter_graph/' + pdb  

    with open(save_path, 'wb') as f_save:
        pickle.dump(data, f_save)
    