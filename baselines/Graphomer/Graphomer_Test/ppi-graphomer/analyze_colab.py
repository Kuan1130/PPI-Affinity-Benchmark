import os
import re
import numpy as np
import esm.inverse_folding
import pandas as pd
from scipy.stats import pearsonr


# This code analyzes the correlation coefficient between IPAE scores and the standard values. We have provided the scoring results obtained using ColabFold.

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
symbol="test"
colab_result=np.load("./pae_result_"+symbol+".npy",allow_pickle=True)
colab_list=[]
true_list=[]
pdb_list_2chain = os.listdir("./data/pdb/2chain_"+symbol+"/")
for pdb in colab_result.item().keys():
    # if pdb+".pdb" not in pdb_list_2chain:
    #     continue
    colab_list.append(colab_result.item()[pdb])
    if symbol=="test":
        true_list.append(-affinity_dict1[pdb.replace(".ent","")])
    else:
        true_list.append(-affinity_dict3[pdb])
print(len(true_list))
# 计算皮尔逊相关系数
if len(colab_list) > 1:  # 至少需要两个数据点
    correlation, _ = pearsonr(true_list, colab_list)
    print(f"pcc: {correlation}")
else:
    print("there is not enough data to calculate pcc!")





