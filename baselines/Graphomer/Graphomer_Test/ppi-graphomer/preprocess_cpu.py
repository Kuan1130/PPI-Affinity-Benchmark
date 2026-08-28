import os
import re
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
import torch
import pandas as pd
import argparse

import os



standard_res =[
        "GLY" , 'G',
        "ALA" , 'A',
        "VAL" , 'V',
        "LEU" , 'L',
        "ILE" , 'I',
        "PRO" , 'P',
        "PHE" , 'F',
        "TYR" , 'Y',
        "TRP" , 'W',
        "SER" , 'S',
        "THR" , 'T',
        "CYS" , 'C',
        "MET" , 'M',
        "ASN" , 'N',
        "GLN" , 'Q',
        "ASP" , 'D',
        "GLU" , 'E',
        "LYS" , 'K',
        "ARG" , 'R',
        "HIS" , 'H'
        ]
atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'else']
degrees = [0, 1, 2, 3, 4, 'else']
hybridizations = ['s', 'sp', 'sp2', 'sp3', 'sp3d', 'sp3d2', 'else']
charges = [-2, -1, 0, 1, 2, 3, 'else']
amino_acids = list("LAGVSETIRDPKQNFYMHW") + ["C", "others"]
num_elements = (20 * 21) // 2
upper_tri_values = np.arange(1, num_elements + 1)

symmetric_interaction_type_matrix = np.zeros((20, 20), dtype=int)

upper_tri_indices = np.triu_indices(20)
symmetric_interaction_type_matrix[upper_tri_indices] = upper_tri_values

symmetric_interaction_type_matrix = symmetric_interaction_type_matrix + symmetric_interaction_type_matrix.T - np.diag(symmetric_interaction_type_matrix.diagonal())


amino_acid_to_index = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


'''affinity_dict = {}

index_file = os.path.join("./data/INDEX_general_PP.2020")

# 获取亲和力信息
# PDBbind的获取方式，这里计算的是以10为底的log，在后面需要换底公式换掉
affinity_dict1 = {}
with open(index_file, 'r') as f:
    lines = f.readlines()
    for line in lines[6:]:
        tokens = line.split()
        protein_name = tokens[0].strip()
        # if protein_name=="5nvl":
        #     print("here")
        # 匹配 Kd, Ki, 或 IC50
        match = re.search(r'(Kd|Ki|IC50)([=<>~])([\d\.]+)([munpfM]+)', line)
        if match:
            measure_type = match.group(1)
            operator = match.group(2)
            value = float(match.group(3))
            unit = match.group(4)
            # 单位转换成 M（摩尔）
            unit_multiplier = {
                'mM': 1e-3,
                'uM': 1e-6,
                'nM': 1e-9,
                'pM': 1e-12,
                'fM': 1e-15
            }
            value_in_molar = value * unit_multiplier.get(unit, 1)  # 默认为 Mol

            # 计算以10为底的log值（pKa）
            if operator == '=' or operator == '~' or operator == '>':
                pKa_value = -np.log(value_in_molar)*0.592
            elif operator == '<':
                # 如果是 '<'，则取 "<" 值更保守的一种处理方式
                pKa_value = -np.log(value_in_molar)*0.592

            affinity_dict1[protein_name] = pKa_value

# benchmark79的亲和力数据
# 读取CSV文件
csv_file = './data/elife-07454-supp4-v4.csv'
data = pd.read_csv(csv_file)
# 创建包含 PDB 名称和亲和力数值的字典
affinity_dict3 = dict(zip(data.iloc[:, 0].apply(lambda x: x.replace(".pdb", "")), data.iloc[:, 1]))
# sorted(affinity_dict.items())
        
affinity_dict={}

affinity_dict.update(affinity_dict1)
# affinity_dict.update(affinity_dict2)
affinity_dict.update(affinity_dict3)



affinity_dict_add=np.load("./data/add_label.npy",allow_pickle=True).item()

affinity_dict.update(affinity_dict_add)'''


affinity_dict = {}

label_file_path = r"../PPIdataindex.txt" 

try:
    with open(label_file_path, 'r') as f:
        for line in f:
            if line.strip(): 
                parts = line.split() 
                pdb_id = parts[0].strip().lower() 
                
                affinity_val = float(parts[1].strip()) 
                affinity_dict[pdb_id] = affinity_val
                
    print(f" Total {len(affinity_dict)} data saved")
except Exception as e:
    print(f"Error: {e}")


from collections import OrderedDict



def list_to_ordered_set(lst):
    ordered_dict = OrderedDict.fromkeys(lst)
    ordered_set = list(ordered_dict.keys())
    return ordered_set


def one_hot_encoding(value, categories):
    vec = [0] * len(categories)
    if value in categories:
        vec[categories.index(value)] = 1
    else:
        vec[-1] = 1
    return vec


def compute_dist(coord1, coord2):
    return np.linalg.norm(coord1 - coord2)


def calculate_distance(atom1, atom2):
    return np.linalg.norm(atom1.coord - atom2.coord)

def is_hydrogen_bond(res1, res2):
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element in ['N', 'O', 'F'] and atom2.element in ['N', 'O', 'F']:
                distance = calculate_distance(atom1, atom2)
                if 2.7 <= distance <= 3.5:
                    count+=1
    return count

def is_halogen_bond(res1, res2):
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element in ['Cl', 'Br', 'I'] and atom2.element in ['N', 'O', 'F']:
                distance = calculate_distance(atom1, atom2)
                if 3.0 <= distance <= 4.0:
                    count+=1
    return count

def is_sulfur_bond(res1, res2):
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element == 'S' and atom2.element == 'S':
                distance = calculate_distance(atom1, atom2)
                if 3.5 <= distance <= 5.5:
                    count+=1
    return count

def is_pi_stack(res1, res2):
    count=0
    pi_residues = ['PHE', 'TYR', 'TRP']
    if res1.resname in pi_residues and res2.resname in pi_residues:
        for atom1 in res1.get_atoms():
            for atom2 in res2.get_atoms():
                distance = calculate_distance(atom1, atom2)
                if 3.3 <= distance <= 4.5:
                    count+=1
    return count

def is_salt_bridge(res1, res2):
    count = 0
    cationic_atoms = [('ARG', 'NH1'), ('ARG', 'NH2'), ('LYS', 'NZ')]
    anionic_atoms = [('ASP', 'OD1'), ('ASP', 'OD2'), ('GLU', 'OE1'), ('GLU', 'OE2')]

    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            res1_atom_pair = (res1.resname, atom1.name)
            res2_atom_pair = (res2.resname, atom2.name)
            if (res1_atom_pair in cationic_atoms and res2_atom_pair in anionic_atoms) or \
               (res1_atom_pair in anionic_atoms and res2_atom_pair in cationic_atoms):
                distance = calculate_distance(atom1, atom2)
                if 2.8 <= distance <= 4.0:
                    count += 1
    return count

def is_cation_pi(res1, res2):
    count = 0
    cationic_atoms = [('ARG', 'NH1'), ('ARG', 'NH2'), ('LYS', 'NZ')]
    pi_residues = ['PHE', 'TYR', 'TRP']
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            res1_atom_pair = (res1.resname, atom1.name)
            res2_resname = res2.resname
            res2_atom_pair = (res2.resname, atom2.name)
            res1_resname = res1.resname
            if (res1_atom_pair in cationic_atoms and res2_resname in pi_residues) or \
               (res2_atom_pair in cationic_atoms and res1_resname in pi_residues):
                distance = calculate_distance(atom1, atom2)
                if 4.0 <= distance <= 6.0:
                    count += 1
    return count


def distance(atom1, atom2):
    diff_vector = atom1.coord - atom2.coord
    return (diff_vector * diff_vector).sum() ** 0.5


def get_ca_positions(residues):
    positions = []
    for residue in residues:
        if 'CA' in residue:
            positions.append(residue['CA'].coord)
        else:
            positions.append(None)  
    return positions

def find_neighbors(query_positions, target_positions, radius=7.0):
    """
    Find neighbors within a given radius for each position in query_positions
    in relation to positions in target_positions.
    Returns a nested list where each sublist corresponds to the indices in
    target_positions that are within the radius of the respective query position.
    """
    neighbors_indices = []

    for query_pos in query_positions:
        if query_pos is None:
            neighbors_indices.append([])  # 跳过没有C-alpha原子的残基
            continue
        neighbors = []
        for i, target_pos in enumerate(target_positions):
            if target_pos is None:
                continue
            distance = np.linalg.norm(query_pos - target_pos)
            if distance <= radius:
                neighbors.append(i)
        neighbors_indices.append(neighbors)

    return neighbors_indices

# 对每个pdb提取信息
def extract_protein_data(pdb_file):
    parser = PDBParser(QUIET=True)
    pdbid=os.path.basename(pdb_file).split(".")
    structure = parser.get_structure(pdbid[0]+'.'+pdbid[1], pdb_file)
    protein_name = structure.id
    sequence = ""
    coord_matrix = []
    features = []
    interface_atoms = defaultdict(list)

    res_mass_centor=[]
    all_atom_coords = []
    all_atom_chains = []
    all_atoms = []
    res_index = -1
    all_res_chain = []
    # 全部氨基酸的坐标，包括水分子与不完整氨基酸等等
    residue_list = [res for res in structure.get_residues()]
    n_residues = len(residue_list)
    # 最后加入到序列中的氨基酸，不包括水分子、非标准氨基酸以及缺失骨架原子的氨基酸
    final_res_list=[]
    absolute_index_res=-1
    matrix_slice_list=[]
    hetatm_res_list=[]
    # 用来指示是否存在骨架原子缺失的问题
    is_fatal_atom=np.zeros(n_residues)
    for model in structure:
        for chain in model:
            for residue in chain:
                absolute_index_res+=1
                res_atoms=[]
                res_atom_coords=[]
                res_atom_chains=[]
                exist_flag=True
                # 先判断是否有三个骨架原子
                n_atom, ca_atom, c_atom = None, None, None
                for atom in residue:
                    coord = atom.get_vector().get_array()
                    if atom.get_id() == 'N':
                        n_atom = coord
                    elif atom.get_id() == 'CA':
                        ca_atom = coord
                    elif atom.get_id() == 'C':
                        c_atom = coord
                    res_atoms.append(atom)
                    res_atom_coords.append(coord)
                    res_atom_chains.append(chain.id)
                if n_atom is None or ca_atom is None or c_atom is None:
                    exist_flag=False
                    is_fatal_atom[absolute_index_res]=True
                # 这里有三种特殊情况，分别是水分子，非标准氨基酸与骨架原子不全的氨基酸
                # 对于骨架原子不全的氨基酸不加入原子列表，因为esmif对于没有骨架原子的氨基酸会出现none
                # 水分子加入原子列表，但不加入序列
                # 非标准氨基酸加入列表和序列，后面会根据seq中的X去掉，但是构建相互作用矩阵时不加入
                # if residue.get_resname() == 'HOH' or residue.get_resname() not in standard_res:
                if residue.get_resname() == 'HOH':
                    res_name = 'HOH'  # 处理水分子但不加入序列
                    all_atoms.extend(res_atoms)
                    all_atom_coords.extend(res_atom_coords)
                    all_atom_chains.extend(res_atom_chains)
                # 骨架原子都存在的情况下
                elif exist_flag:
                    res_name = seq1(residue.get_resname())
                    sequence += res_name
                    all_res_chain.append(chain.id)
                    coord_matrix.append([n_atom, ca_atom, c_atom])
                    res_index += 1
                    all_atoms.extend(res_atoms)
                    all_atom_coords.extend(res_atom_coords)
                    all_atom_chains.extend(res_atom_chains)
                    if residue.get_resname() in standard_res:
                        final_res_list.append(residue)
                        res_mass_centor.append(np.mean(res_atom_coords,axis=0))
                        matrix_res2_list=[]
                        for idx2, res2 in enumerate(residue_list):
                            # 只计算下三角，或者如果骨架原子缺失或者为水分子，则跳过
                            # 这里必须是下三角，因为只有res_index大于idx2，对应的is_fatal_atom才已经被验证过了
                            if absolute_index_res <= idx2  or residue.get_resname() == 'HOH':
                                continue  
                            if is_fatal_atom[idx2]:
                                continue
                            if res2.get_resname() not in standard_res:
                                continue
                            # 不同链则补0
                            matrix_res2_slice=np.zeros(6)
                            matrix_res2_slice[0]=is_hydrogen_bond(residue, res2)
                            matrix_res2_slice[1]=is_halogen_bond(residue, res2)
                            matrix_res2_slice[2]=is_sulfur_bond(residue, res2)
                            matrix_res2_slice[3]=is_pi_stack(residue, res2)
                            matrix_res2_slice[4]=is_salt_bridge(residue, res2)
                            matrix_res2_slice[5]=is_cation_pi(residue, res2)
                            matrix_res2_list.append(matrix_res2_slice)
                        matrix_slice_list.append(matrix_res2_list)

                    else:
                        hetatm_res_list.append(residue)
                else:
                    hetatm_res_list.append(residue)

    sum_array = np.zeros(6)

    # 遍历嵌套列表进行累加
    for inner_list in matrix_slice_list:
        for array in inner_list:
            sum_array += array


    # 找每个序列中氨基酸周围一定范围内的配体分子
    A_positions = get_ca_positions(final_res_list)
    B_positions = get_ca_positions(hetatm_res_list)
    # hetatm_feat_list=[]
    # 
    hetatm_list=np.load("./data/hetatm_list.npy")


    neighbors_hetatm_index = find_neighbors(A_positions, B_positions, radius=7.0)
    res_neighbors_hetatm=[]
    for res_record in neighbors_hetatm_index:
        feat=np.zeros(len(hetatm_list))
        for hetatm_record in res_record:
            feat+=one_hot_encoding(hetatm_res_list[hetatm_record].get_resname(),hetatm_list.tolist())
        res_neighbors_hetatm.append(feat)


    # 初始化 n*n*6 矩阵
    n_valid_residues=len(matrix_slice_list)
    interaction_matrix = np.zeros((n_valid_residues, n_valid_residues, 6))

    # 填充矩阵
    for idx1, matrix_slice in enumerate(matrix_slice_list):
        if idx1==0:
            continue
        for idx2,matrix_res2_slice in enumerate(matrix_slice):
            interaction_matrix[idx1, idx2, :] = matrix_res2_slice
            interaction_matrix[idx2, idx1, :] = matrix_res2_slice  # 对称填充
    seq=sequence.replace("X","").replace("Z","")
    n = len(seq)
    interaction_type = np.zeros((n, n), dtype=int)
    # Assume `sequence` is a string consisting of amino acids where each unique amino acid can be indexed
    for i in range(n):
        for j in range(n):
            aa1 = seq[i]
            aa2 = seq[j]
            idx1 = amino_acid_to_index[aa1]
            idx2 = amino_acid_to_index[aa2]
            interaction_value = symmetric_interaction_type_matrix[idx1, idx2]
            interaction_type[i, j] = interaction_value

    all_atom_coords = np.array(all_atom_coords)
    residue_to_index = {}
    current_index = 0

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() != 'HOH':
                    # 为每个残基分配一个唯一的序号
                    residue_to_index[residue] = current_index
                    current_index += 1


    coord_matrix = np.array(coord_matrix,dtype=object)

    # 计算界面原子
    # 获取所有原子的坐标
    all_atom_coords = np.array([atom.get_coord() for atom in all_atoms])
    distance_matrix = cdist(all_atom_coords, all_atom_coords)
    within_7A = distance_matrix <= 7.0
    # 构建界面原子列表
    for i in range(len(all_atoms)):
        interface_atoms[i] = np.where(
            (within_7A[i]) & 
            (np.arange(len(all_atoms)) != i) & 
            (np.array(all_atom_chains) != all_atom_chains[i])
        )[0].tolist()

    grouped_interface = defaultdict(list)
    for i,atom_if in enumerate(list(interface_atoms.values())):
        index=residue_to_index.get(all_atoms[i].get_parent(), -1)
        if index!=-1:
            grouped_interface[index].append(atom_if)

    # 将序列中的未知氨基酸去掉，同步去掉各个列表的对应氨基酸
    indices_to_delete = [i for i, char in enumerate(sequence) if char == 'X' or char == 'Z']
    coord_matrix = np.delete(coord_matrix, indices_to_delete, axis=0)

    interaction_type = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            aa1 = seq[i]
            aa2 = seq[j]
            idx1 = amino_acid_to_index[aa1]
            idx2 = amino_acid_to_index[aa2]
            interaction_value = symmetric_interaction_type_matrix[idx1, idx2]
            interaction_type[i, j] = interaction_value

    chain_id_res = [elem for i, elem in enumerate(all_res_chain) if i not in indices_to_delete]
    atom_interface_list=[elem for i, elem in enumerate(list(grouped_interface.values())) if i not in indices_to_delete]
    res_interface_list=[]
    for res_if in atom_interface_list:
        res_list=[]
        for atom_if in res_if:
            for atom_if_item in atom_if:
                res_list.append(residue_to_index.get(all_atoms[atom_if_item].get_parent(), -1))
        res_interface_list.append(list(set(res_list)))
    res_interface_list = [elem for i, elem in enumerate(res_interface_list) if i not in indices_to_delete]
    seq_single_chain = [''.join(seq[i] for i in range(len(seq)) if chain_id_res[i] == x) for x in list_to_ordered_set(chain_id_res)]


    if affinity_dict.get(protein_name.replace(".pdb",""), None) is None:
        print("affinity error!",protein_name)

    # 返回蛋白质信息字典
    protein_data = {
        "protein_name": protein_name,
        "sequence": seq_single_chain,
        "chain_id_res":chain_id_res,
        "hetatm_features": res_neighbors_hetatm,
        "interface_atoms": atom_interface_list,
        "interface_res":res_interface_list,
        "interaction_type_matrix":interaction_type.astype(np.int32),
        "interaction_matrix":interaction_matrix.astype(np.int32),
        "res_mass_centor":np.stack(res_mass_centor).astype(np.float16),
        "affinity": affinity_dict.get(protein_name.replace(".pdb",""), None)
    }
    return protein_data




def single_worker(pdb_sub_dir_list, p_number,save_dir,pdb_folder):
    # save_dir="/public/mxp/xiejun/py_project/PPI-Graphomer/data_final/preprocess/cpu/sequence_renamed1-4/"
    os.makedirs(save_dir, exist_ok=True)
    try:
        result_list=[]
        for pdb_file in tqdm(pdb_sub_dir_list):
            full_pdb_path = os.path.join(pdb_folder, pdb_file)
            result_list.append(extract_protein_data(full_pdb_path))
            torch.cuda.empty_cache()
            print("processed: ",pdb_file)
        np.save(save_dir+"/cpu"+str(p_number)+".npy", result_list,allow_pickle=True)
        print("saved!")
    except Exception as e:
        with open(save_dir+'error{}.txt'.format(p_number),'w+') as f:
            print("in No.{} process, exception occurred".format(p_number))
            print("in {}".format(pdb_file))
            print("line:{}".format(e.__traceback__.tb_lineno))
            print(str(e))
            f.write("in No.{} process, exception occurred:\n".format(p_number))
            f.write("in {}\n".format(pdb_file))
            f.write("line:{}".format(e.__traceback__.tb_lineno))
            f.write(str(e))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", '-n', default=3, type=int, help="The first number")
    parser.add_argument(
        "--save_dir", "-s",
        default="./preprocess_cpu_data",
        type=str,
        help="Directory for CPU preprocessing shards",
    )
    parser.add_argument(
        "--pdb_folder", "-p",
        default="../../../../data/local/pdbs",
        type=str,
        help="Directory containing shared PDB files",
    )

    # assign your pdbs here
    # pdb_folder = "./data_final/pdb/Accepted"
    args = parser.parse_args()


    # multiprocessing.set_start_method("spawn")
    processor=args.workers
    pdb_dir_list=os.listdir(args.pdb_folder)
    p = Pool(processor)
    num_pdb = len(pdb_dir_list)
    n = num_pdb // processor
    print(num_pdb)
    for i in range(processor):
        start = n * i
        end = num_pdb if i == processor - 1 else n * (i + 1)
        pdb_sub_dir_list = pdb_dir_list[start:end]
        print(pdb_sub_dir_list)
    print(num_pdb)
    for i in range(processor):
        start = n * i
        end = num_pdb if i == processor - 1 else n * (i + 1)
        pdb_sub_dir_list = pdb_dir_list[start:end]
        # pdb_sub_dir_list = ['nz']
        p.apply_async(single_worker, args=(pdb_sub_dir_list, i,args.save_dir,args.pdb_folder))
    p.close()
    p.join()
