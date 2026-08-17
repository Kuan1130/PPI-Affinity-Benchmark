import os
import tempfile
from Bio import SeqIO
from subprocess import run, PIPE
import tqdm


# This script will use CD-HIT for deduplication. Please ensure that you have downloaded CD-HIT and set the correct PDB path.


def write_fasta(sequences, file_path):
    """将序列写入FASTA文件"""
    with open(file_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq{i}\n{seq}\n")

def read_fasta(file_path):
    """从FASTA文件读取序列"""
    with open(file_path, 'r') as f:
        return [str(rec.seq) for rec in SeqIO.parse(f, "fasta")]

def parse_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        pdb_name = None
        sequence_lines = []

        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if pdb_name is not None:
                    sequences[pdb_name] = ''.join(sequence_lines)
                pdb_name = line[1:]
                sequence_lines = []
            else:
                sequence_lines.append(line)

        # Add the last sequence to the dictionary
        if pdb_name is not None:
            sequences[pdb_name] = ''.join(sequence_lines)

    return sequences

def cdhit_deduplicate(source_path,  identity_threshold=0.7):
    """
    使用CD-HIT对序列去冗余

    :param sequences: 输入的序列列表
    :param cd_hit_executable: CD-HIT可执行文件的路径
    :param identity_threshold: 序列身份阈值 (默认0.6)
    :return: 去冗余后的序列列表
    """
    cd_hit_executable = "./your_cd-hit_path/cd-hit"
    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as input_file, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as output_file:

        input_file_path = input_file.name
        output_file_path = output_file.name

        # 将序列写入FASTA文件
        # write_fasta(sequences, input_file_path)
        input_file_path="dips.fasta"
        output_file_path="dips_drop1.fasta"
        # 运行CD-HIT
        cdhit_command = [
            cd_hit_executable,
            "-i", input_file_path,
            "-o", output_file_path,
            "-c", str(identity_threshold)
        ]
        run(cdhit_command, check=True, stdout=PIPE, stderr=PIPE)

        # 读取去冗余后的序列
        dedup_sequences = parse_fasta(output_file_path)
        delete_list=[]
        # 删除临时文件
        for name in os.listdir(all_path):
            if name not in dedup_sequences.keys():
                # print(source_path+name)
                delete_list.append(name)
            # else:
            #     print(source_path+name)
        print(len(delete_list))
        for it,name in tqdm.tqdm(enumerate(delete_list)):
            # print(source_path+name)
            if os.path.exists(source_path+name):
                # print(it)
                os.remove(source_path+name)



        return dedup_sequences

source_path="./data/pdb/benmark79/"
# all_path="/public/mxp/xiejun/py_project/PPI_affinity/data_final/pdb/dips_plus/"
deduplicated_sequences = cdhit_deduplicate(source_path)
print(len(deduplicated_sequences))