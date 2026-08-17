# -*- coding: utf-8 -*-
import os
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from collections import defaultdict
from scipy.spatial.distance import cdist
import esm
import torch
import esm.inverse_folding
from collections import OrderedDict
import torch.nn.functional as F
import torch.utils.data as Data
import argparse
import warnings
warnings.filterwarnings("ignore")


# This code will predict the affinity for the specified PDB.


# 常量定义
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




def list_to_ordered_set(lst):
    # 使用字典来消除重复并保持顺序
    ordered_dict = OrderedDict.fromkeys(lst)
    # 将字典的键转换为集合（按照出现的顺序）
    ordered_set = list(ordered_dict.keys())
    return ordered_set


def extract_protein_data(pdb_file, model_esm, alphabet, model_esmif, alphabet_if, device):
    batch_converter = alphabet.get_batch_converter()
    parser = PDBParser(QUIET=True)
    base_name = os.path.basename(pdb_file).split(".")
    structure = parser.get_structure(base_name[0]+'.'+base_name[1], pdb_file)
    protein_name = os.path.basename(pdb_file).split("/")[-1][0:-4]


    sequence = ""
    sequence_true = ""
    coord_matrix = []
    all_atom_chains = []
    res_index = 0  # 氨基酸全局索引
    all_res_chain = []
    padG_index_list = [] # 所有用Gpad的索引（对应上面的全局索引）

    # standard_res = set("ACDEFGHIKLMNPQRSTVWY") # Assume you have set of standard residues

    for model in structure:
        for chain in model:
            previous_residue_id = None
            for residue in chain:
                res_atoms, res_atom_coords, res_atom_chains = [], [], []
                exist_flag = True   # 判断是否存在三个骨架原子

                # Check backbone atoms
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
                    exist_flag = False
                # 对于水分子和非标准氨基酸，不做处理
                if residue.get_resname() == 'HOH' or residue.get_resname() not in standard_res:
                    res_name = 'HOH'
                # 如果四个骨架原子都存在
                elif exist_flag:
                    res_name = seq1(residue.get_resname())
                    residue_id = residue.id[1]
                    # 判断两者相等是为了把插入残基也纳入
                    # if residue_id == previous_residue_id:
                    #     print("jere")
                    # 需要插入G的地方，氨基酸序号不连续且不相等，且前一个序号大于0（因为有些序号小于0的氨基酸作为补充实际上连续）
                    if previous_residue_id and residue_id != previous_residue_id + 1 and previous_residue_id > 0 and residue_id != previous_residue_id :
                        gap_size = residue_id - previous_residue_id - 1
                        gap_seq = 'G' * gap_size
                        sequence += gap_seq
                        padG_index_list.extend([res_index + i for i in range(gap_size)])
                        all_res_chain.extend([chain.id] * gap_size)
                        res_index += gap_size
                    sequence += res_name
                    # 真实序号，不加入G
                    sequence_true += res_name
                    all_res_chain.append(chain.id)
                    coord_matrix.append([n_atom, ca_atom, c_atom])
                    res_index += 1
                    previous_residue_id = residue_id
                else:
                    # 没有骨架原子的残缺氨基酸
                    res_name = 'canque'
    # 坐标信息
    coord_matrix = np.array(coord_matrix, dtype=object)

    # Filter out unwanted amino acids
    indices_to_delete = [i for i, char in enumerate(sequence) if char in 'XZ']
    sequence = sequence.replace("X", "").replace("Z", "")
    chain_id_res = [elem for i, elem in enumerate(all_res_chain) if i not in indices_to_delete]
    # 按链分割序列
    seq_single_chain = [
        ''.join(sequence[j] for j in range(len(sequence)) if chain_id_res[j] == x)
        for x in list_to_ordered_set(chain_id_res)
    ]
    # 链接不同链的linker
    separator = "GGGGGGGGGGGGGGGGGGGG"
    full_sequence = ""
    separator_indices = []
    length = 0

    current_length = 0  # 用于记录当前序列长度
    linker_offsets = []  # 用于记录每个链的开始位置，以供以后更新padG索引

    padG_index_list_delete=padG_index_list.copy()
    updated_padG_index_list = []
    for seq in seq_single_chain[:-1]:  
        full_sequence += seq
        current_length += len(seq)

        # 更新padG_index_list
        # 使用索引删除原来的index_list，如果小于curindex说明在当前序列，直接加入，否则索引加上linker长度，暂不处理
        count=0
        for it in range(len(padG_index_list_delete)):
            if padG_index_list_delete[it-count] >= current_length:
                padG_index_list_delete[it-count]+= len(separator)
            else:
                updated_padG_index_list.append(padG_index_list_delete[it-count])
                del padG_index_list_delete[it-count]
                count+=1

        # 插入分隔符linker
        full_sequence += separator

        separators_start = current_length
        separator_indices.extend(range(separators_start, separators_start + len(separator)))

        current_length += len(separator)

    full_sequence += seq_single_chain[-1]
    current_length += len(seq_single_chain[-1])
    # 确保最后一部分的索引也包含在updated_padG_index_list中
    count=0
    for it in range(len(padG_index_list_delete)):
        if padG_index_list_delete[it-count] >= current_length:
            padG_index_list_delete[it-count]+= len(separator)
        else:
            updated_padG_index_list.append(padG_index_list_delete[it-count])
            del padG_index_list_delete[it-count]
            count+=1

    # 理应处理完索引index
    assert padG_index_list_delete==[]
    # 名字无所谓
    concatenated_name = "_".join([f"{protein_name}" for i in range(len(seq_single_chain))])
    # 形成tokens，注意这里包含开始和结尾token
    enc_inputs_labels, enc_inputs_strs, enc_tokens = batch_converter([(concatenated_name, full_sequence)])
    # 获取特征，不包括开始和结尾token
    with torch.no_grad():
        results = model_esm(enc_tokens.to(device), repr_layers=[33], return_contacts=False)
        concatenated_representations = results["representations"][33][:, 1:-1, :].cpu()
    # 包括起始和结尾的token为all
    enc_tokens_all=enc_tokens
    # 不包括起始和结尾token
    enc_tokens = enc_tokens[:, 1:-1]

    # Combine padG_index_list and separator_indices
    combined_index_list = sorted(updated_padG_index_list + separator_indices)

    # Create mask
    mask = torch.ones(concatenated_representations.size(1), dtype=torch.bool)
    # 总的mask，包括两个pad——index
    mask[combined_index_list] = False

    # mask中有效值应该等于真实序列长度
    assert sum(mask).item()==len(sequence_true)

    # 获取真实token_representation
    token_representations = concatenated_representations[:, mask, :]
    masked_enc_tokens = enc_tokens[:, mask]

    # 更新 chain_id_res，padG_index_list是加入linker之前的，chain_id_res也是加入linker之前的,去掉对应索引，变成和sequence_true一样
    mask_padG = np.ones(len(chain_id_res), dtype=bool)
    mask_padG[padG_index_list] = False
    chain_id_res = [element for i, element in enumerate(chain_id_res) if mask_padG[i]]


    # indices_to_delete = [i for i, char in enumerate(sequence_true) if char in 'XZ']
    # sequence_true = sequence_true.replace("X", "").replace("Z", "")

    seq_single_chain = [
        ''.join(sequence_true[j] for j in range(len(sequence_true)) if chain_id_res[j] == x)
        for x in list_to_ordered_set(chain_id_res)
    ]

    coord_matrix = np.delete(coord_matrix, indices_to_delete, axis=0)
    # 拆分token_representation和tokens
    token_representations_list, enc_tokens_list = [], []
    curr_pos = 0
    for seq in seq_single_chain:
        seq_len = len(seq)
        single_representations = token_representations[:, curr_pos:curr_pos + seq_len, :]
        token_representations_list.append(single_representations)
        enc_tokens_list.append(masked_enc_tokens[0, curr_pos:curr_pos + seq_len])
        curr_pos += seq_len

    coor_dict = {
        chain: coord_matrix[np.array(chain_id_res) == chain].reshape(-1, 3, 3).astype('float32')
        for chain in list_to_ordered_set(chain_id_res)
    }

    torch.cuda.empty_cache()

    coor_feature_list = []

    for chain_id in list_to_ordered_set(chain_id_res):
        with torch.no_grad():
            coor_feature_list.append(
                esm.inverse_folding.multichain_util.get_encoder_output_for_complex(
                    model_esmif, alphabet_if, coor_dict, chain_id
                ).cpu()
            )
    # 每个链的长度应该相等
    for i in range(len(token_representations_list)):
        assert token_representations_list[i].shape[1] == coor_feature_list[i].shape[0] == enc_tokens_list[i].shape[0]

    
    addition = [token_representations_list, coor_feature_list, enc_tokens_list,enc_tokens_all,mask,seq_single_chain]

    protein_data = {
        protein_name: addition
    }

    return protein_data


def calculate_distance(atom1, atom2):
    """计算两个原子之间的距离"""
    return np.linalg.norm(atom1.coord - atom2.coord)

def is_hydrogen_bond(res1, res2):
    """判断两个氨基酸是否形成氢键"""
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element in ['N', 'O', 'F'] and atom2.element in ['N', 'O', 'F']:
                distance = calculate_distance(atom1, atom2)
                if 2.7 <= distance <= 3.5:
                    count+=1
    return count

def is_halogen_bond(res1, res2):
    """判断两个氨基酸是否形成卤键"""
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element in ['Cl', 'Br', 'I'] and atom2.element in ['N', 'O', 'F']:
                distance = calculate_distance(atom1, atom2)
                if 3.0 <= distance <= 4.0:
                    count+=1
    return count

def is_sulfur_bond(res1, res2):
    """判断两个氨基酸是否形成硫键"""
    count=0
    for atom1 in res1.get_atoms():
        for atom2 in res2.get_atoms():
            if atom1.element == 'S' and atom2.element == 'S':
                distance = calculate_distance(atom1, atom2)
                if 3.5 <= distance <= 5.5:
                    count+=1
    return count

def is_pi_stack(res1, res2):
    """判断两个氨基酸是否形成π-π堆积"""
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
    """判断两个氨基酸是否形成盐桥"""
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
    """判断两个氨基酸是否形成阳离子-π相互作用"""
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


def get_ca_positions(residues):
    """
    Get the C-alpha atom positions for the given list of residues.
    """
    positions = []
    for residue in residues:
        if 'CA' in residue:
            positions.append(residue['CA'].coord)
        else:
            positions.append(None)  # 无 C-alpha 原子时用 None 占位
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


def one_hot_encoding(value, categories):
    vec = [0] * len(categories)
    if value in categories:
        vec[categories.index(value)] = 1
    else:
        vec[-1] = 1
    return vec

amino_acid_to_index = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

affinity_dict = {}



# 创建一个包含连续整数的数组，而后形成上三角矩阵
num_elements = (20 * 21) // 2  # 计算上三角矩阵元素个数
upper_tri_values = np.arange(1, num_elements + 1)
# 初始化20x20矩阵
symmetric_interaction_type_matrix = np.zeros((20, 20), dtype=int)
# 填充上三角矩阵
upper_tri_indices = np.triu_indices(20)
symmetric_interaction_type_matrix[upper_tri_indices] = upper_tri_values
# 将矩阵对称化
symmetric_interaction_type_matrix = symmetric_interaction_type_matrix + symmetric_interaction_type_matrix.T - np.diag(symmetric_interaction_type_matrix.diagonal())


def extract_protein_cpu_data(pdb_file):
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


    metal_ions = ['CA', 'MG', 'ZN', 'FE', 'CU', 'K', 'NA']
    # 找每个序列中氨基酸周围一定范围内的配体分子
    A_positions = get_ca_positions(final_res_list)
    B_positions = get_ca_positions(hetatm_res_list)
    # hetatm_feat_list=[]
    hetatm_list=np.load("./data/hetatm_list.npy")

    # for h_res in hetatm_res_list:
    #     # if h_res.get_resname() in metal_ions:
    #     #     hetatm_feat_list.append(one_hot_encoding(h_res.get_resname(),metal_ions))
    #     # else:
    #     #     smiles_string = residue_to_smiles(h_res)
    #     #     # smiles=residue_to_smiles(h_res)
    #     #     hetatm_feat_list.append(extract_molecular_features(smiles_string))
    #     hetatm_feat_list.append(one_hot_encoding(h_res.get_resname(),hetatm_list))

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
    # Assume `sequence` is a string consisting of amino acids where each unique amino acid can be indexed
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


    # if affinity_dict.get(protein_name, None) is None:
    #    print("affinity error!",protein_name)

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
        "affinity": affinity_dict.get(protein_name, None)
    }
    return protein_data

def load_if_add_dict(if_add_path):
    if_add_dict = {}
    if_add_path_list = os.listdir(if_add_path)
    for if_add_file in if_add_path_list:
        if_add_list=np.load(os.path.join(if_add_path, if_add_file), allow_pickle=True)
        for if_add_dict_single in if_add_list:
            if_add_dict.update(if_add_dict_single)
    return if_add_dict

def process_train_data(train_data,if_add_dict, pro_len=2000):


    # if_add_path = "/public/mxp/xiejun/py_project/PPI_affinity/benchmark79_gpu"

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

    item = train_data
        # pdbbind_name_list=np.load("./pdbbind_name_list.npy",allow_pickle=True)
        # if item["protein_name"].lower() not in pdbbind_name_list:
        #     continue
    protein_names.append(item["protein_name"])
    seq_temp=""
    for i in item["sequence"]:
        seq_temp+=i
    if len(seq_temp)>pro_len:
        exit()
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
    affinity.append(item["affinity"])
    if_type=torch.tensor(item["interaction_type_matrix"]).type(torch.int16)
    interaction_type.append(F.pad(if_type,(0,pro_len-if_type.shape[0],0,pro_len-if_type.shape[0])))
    if_matrix=torch.tensor(item["interaction_matrix"]).type(torch.int16)
    interaction_matrix.append(F.pad(if_matrix,(0,0,0,pro_len-if_matrix.shape[0],0,pro_len-if_matrix.shape[0])))
    mass_centor=torch.tensor(item["res_mass_centor"])
    res_mass_centor.append(F.pad(mass_centor,(0,0,0,pro_len-mass_centor.shape[0])))
    hetatm_features_single=torch.tensor(np.stack(item["hetatm_features"])).type(torch.float32)
    hetatm_features.append(F.pad(hetatm_features_single,(0,0,0,pro_len-hetatm_features_single.shape[0])))

    batch_data={
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
    return batch_data

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
    affinity = None
    seqs = [item[7] for item in batch]
    interaction_type = torch.stack([item[8] for item in batch])
    interaction_matrix = torch.stack([item[9] for item in batch])
    res_mass_centor = torch.stack([item[10] for item in batch])
    hetatm_features = torch.stack([item[11] for item in batch])

    return protein_names,chain_id_res,enc_tokens,seq_features,coor_features,\
    interface_atoms,affinity,seqs,interaction_type,interaction_matrix,res_mass_centor,hetatm_features


def evaluate(model, loader,device):
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
            interface_atoms_val=interface_atoms_val.to(device)
            interaction_type_val,interaction_matrix_val,res_mass_centor_val=interaction_type_val.type(torch.int64).to(device),interaction_matrix_val.type(torch.int32).to(device),res_mass_centor_val.to(device)
            hetatm_features_val=hetatm_features_val.type(torch.float).to(device)

            val_outputs= model(enc_tokens_val, seq_features_val, coor_features_val, hetatm_features_val,interface_atoms_val,\
                               interaction_type_val,interaction_matrix_val,res_mass_centor_val,seqs_val,protein_names_val,chain_id_res_val)
            output_list.append(val_outputs.view(-1))
            protein_names_val_list.extend(protein_names_val)
        
        output_all=torch.cat(output_list,dim=0)


    return epoch_loss / len(loader),output_all


def inference(pdb_path,device="cuda"):
        
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cuda:0')
        model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # model_esm, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

        # model_esm=torch.load("/public/mxp/xiejun/py_project/esm_finetune/myresult/attempt1/model_esm_8.pth")
        model_esm=model_esm.eval().to(device)
        model_esmif, alphabet_if = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        model_esmif = model_esmif.eval()
        gpu_data=extract_protein_data(pdb_path,model_esm,alphabet,model_esmif,alphabet_if,device)
        torch.cuda.empty_cache()
        cpu_data=extract_protein_cpu_data(pdb_path)
        val_data_dict=process_train_data(cpu_data,gpu_data)
        batch_size=1
        val_loader = Data.DataLoader(
            MyDataSet(val_data_dict["protein_names"], val_data_dict["chain_id_res"], val_data_dict["enc_tokens"], 
                    val_data_dict["seq_features"], val_data_dict["coor_features"], 
                    val_data_dict["interface_atoms"],val_data_dict["affinity"],val_data_dict["seqs"],
                    val_data_dict["interaction_type"],val_data_dict["interaction_matrix"],val_data_dict["res_mass_centor"],
                    val_data_dict["hetatm_features"]), 
                    
            batch_size=batch_size, shuffle=False,collate_fn=collate_fn
        )
        # prefix="runs/run_11_final/attempt9_nompnn_enc_encpre4"
        transformer_model = torch.load('./model/model_0.pth',map_location=device)
        # transformer_model.save()
        valid_loss ,output_all= evaluate(transformer_model, val_loader,device)
        return output_all



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    # parser.add_argument("--workers", '-n', default=1, type=int, help="The first number")
    # parser.add_argument("--save_dir", '-s', default="./data_final/preprocess/gpu/default/", type=str, help="Save directory")
    # parser.add_argument("--pdb_folder", '-p', default="./data_final/pdb/Accepted/", type=str, help="pdb directory")

    parser.add_argument("--pdb", '-p', default="./data/1E96.pdb", type=str, help="pdb path")
    args = parser.parse_args()

    pdb_path=args.pdb
    aff=inference(pdb_path)
    print("completed!")
    print("predict affinity:",aff.item())



