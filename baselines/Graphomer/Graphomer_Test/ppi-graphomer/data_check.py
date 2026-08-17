import numpy as np    
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import argparse


# This code will verify the correctness of the data shape generated previously, ensuring that the features obtained from the CPU and GPU are aligned.

def load_if_add_dict(if_add_path):
    if_add_dict = {}
    if_add_path_list = os.listdir(if_add_path)
    for if_add_file in if_add_path_list:
        if_add_list=np.load(os.path.join(if_add_path, if_add_file), allow_pickle=True)
        for if_add_dict_single in if_add_list:
            if_add_dict.update(if_add_dict_single)
    return if_add_dict




if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    # parser.add_argument("--workers", '-n', default=1, type=int, help="The first number")
    # parser.add_argument("--save_dir", '-s', default="./data_final/preprocess/gpu/default/", type=str, help="Save directory")
    # parser.add_argument("--pdb_folder", '-p', default="./data_final/pdb/Accepted/", type=str, help="pdb directory")

    parser.add_argument("--cpu_path", '-c', default="./data/preprocess/cpu/default/", type=str, help="cpu data path")
    parser.add_argument("--gpu_path", '-g', default="./data/preprocess/gpu/default/", type=str, help="gpu data path")
    parser.add_argument("--save_folder", '-s', default="./data/checked_data/default/", type=str, help="checked data path")

    # assign your pdbs here
    # pdb_folder = "./data_final/pdb/Accepted"
    args = parser.parse_args()

    source_path=args.cpu_path
    if_add_path = args.gpu_path
    save_folder = args.save_folder

    if_add_dict = load_if_add_dict(if_add_path)
    data_all=[]
    res_length_list=[]
    atom_length_list=[]
    max_res_l=0
    max_atom_l=0
    min_res_l=100000
    min_atom_l=1000000
    name_list=[]
    source_dirs=os.listdir(source_path)
    seq_list=[]
    for data_dir in tqdm(os.listdir(source_path)):
        if not data_dir.endswith(".npy"):
            continue
        data=np.load(os.path.join(source_path,data_dir),allow_pickle=True)
        print("loading ",os.path.join(source_path,data_dir))
        for item in data:
            try:
                seq=""
                # 检查链数
                assert len(item["sequence"])==len(set(item["chain_id_res"]))
                name_list.append(item["protein_name"])
                seq_len=0
                gpu_data1_len=0
                gpu_data2_len=0
                gpu_data3_len=0
                continue_flag=0

                for i in range(len(set(item["chain_id_res"]))):
                    cpu_len_t=len(item["sequence"][i])
                    seq_len+=cpu_len_t
                    seq+=item["sequence"][i]

                    # 保持分割后的链长相等，确保链的顺序对应
                    gpu_len_t1=if_add_dict[item["protein_name"].replace(".pdb","")][0][i].shape[1]
                    gpu_len_t2=if_add_dict[item["protein_name"].replace(".pdb","")][1][i].shape[0]
                    gpu_len_t3=if_add_dict[item["protein_name"].replace(".pdb","")][2][i].shape[0]

                    gpu_data1_len+=gpu_len_t1
                    gpu_data2_len+=gpu_len_t2
                    gpu_data3_len+=gpu_len_t3
                    # 检查各个gpu特征长度和序列是否相同
                    if not (cpu_len_t==gpu_len_t1==gpu_len_t2==gpu_len_t3 ):
                        continue_flag=1
                    if not if_add_dict[item["protein_name"].replace(".pdb","")][5][i]==item["sequence"][i]:
                        continue_flag=1

                if continue_flag:
                    print("some data error,skip!")
                    continue
                seq_list.append(seq)
                chain_id_len=len(item["chain_id_res"])
                hetatm_len=len(item["hetatm_features"])
                ia_type_len=item["interaction_type_matrix"].shape[0]
                ia_matrix_len=item["interaction_matrix"].shape[0]
                if_len=len(item["interface_atoms"])
                res_mass_len=item["res_mass_centor"].shape[0]
                # 检查cpu各个特征的长度是否相同
                assert seq_len==chain_id_len==hetatm_len==ia_type_len==ia_matrix_len==if_len==res_mass_len==gpu_data1_len

                res_length_list.append(seq_len)
                data_all.append(item)
            except:
                print("skip ",item["protein_name"])

                
    os.makedirs(save_folder, exist_ok=True)
    # np.save("seq_list.npy",seq_list,allow_pickle=True)
    np.save(save_folder+"/checked_cpu_data.npy",data_all,allow_pickle=True)
    # np.save("./pdbbind_name_list.npy",name_list,allow_pickle=True)
    print("samples number:",len(data_all))
    dict_all=Counter(res_length_list)
    print("max res:",max(dict_all.keys()))
    print("min res:",min(dict_all.keys()))
    plt.bar(dict_all.keys(), height=dict_all.values())
    # plt.savefig(save_folder+"/res_length.png")

    print(len(res_length_list))
    print("completed!")
