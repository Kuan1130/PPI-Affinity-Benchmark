import os
import subprocess

import os
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1
import tqdm
import subprocess
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# This code will use BLAST for deduplication. Please ensure that you have downloaded BLAST and set the correct PDB path.

def extract_sequences(pdb_folder, output_fasta):
    sequences = {}
    pdb_parser = PDBParser(QUIET=True)

    with open(output_fasta, 'w') as fasta_file:
            for filename in tqdm.tqdm(os.listdir(pdb_folder)):
                if filename.endswith(".pdb"):
                    filepath = os.path.join(pdb_folder, filename)
                    try:
                        structure = pdb_parser.get_structure(filename, filepath)
                        for model in structure:
                            seq_all = []
                            for chain in model:
                                chain_sequence = []
                                for residue in chain:
                                    if residue.id[0] == " " and residue.resname not in ["HOH"]:
                                        try:
                                            chain_sequence.append(seq1(residue.resname))  # Convert three-letter code to one-letter code
                                        except KeyError:
                                            pass  # Handle rare cases where the residue name is unknown
                                seq_all.append("".join(chain_sequence))
                            seq = "".join(seq_all)
                            seq_id = filename
                            sequences[seq_id] = seq
                            fasta_file.write(f">{seq_id}\n{seq}\n")
                    except:
                        print(filename)

    return sequences


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

def run_blast(fasta_file_source, fasta_file_target,blast_output):
    # 运行BLASTp比对并输出为tabular格式
    subprocess.run(['blastp', '-task', 'blastp-fast','-query', fasta_file_source, '-subject', fasta_file_target, '-out', blast_output,
                    '-outfmt', '6 qseqid sseqid pident length'])

def parse_blast_output(blast_output,sequences_source, threshold):
    # 使用pandas读取tabular格式的BLAST输出
    blast_df = pd.read_csv(blast_output, sep='\t', names=['query_id', 'subject_id', 'pident', 'length'])

    to_remove = set()

    for index, row in tqdm.tqdm(blast_df.iterrows()):
        query_id = row['query_id']
        subject_id = row['subject_id']
        similarity = row['pident'] / 100
        length = row['length']


        if query_id != subject_id and similarity >= threshold and length/len(sequences_source[query_id])>=0.9:
            to_remove.add(query_id)

    return to_remove

def find_and_remove_similar_sequences(source_folder,target_folder, threshold=0.7):
    output_fasta_source = "train.fasta"
    blast_output = "./data/train_test_bm79_skempi26.txt"
    output_fasta_target = "skempi26.fasta"

    print("loading sequence:")

    sequences_source = extract_sequences(source_folder, output_fasta_source)
    sequences_target = extract_sequences(target_folder, output_fasta_target)
    # exit()
    # sequences_source = parse_fasta(output_fasta_source)
    # sequences_target = parse_fasta(output_fasta_target)

    print("running blast:")

    run_blast(output_fasta_source,output_fasta_target, blast_output)
    print("removing duplicate:")

    to_remove = parse_blast_output(blast_output,sequences_source, threshold)
    print(len(to_remove))
    for filename in tqdm.tqdm(to_remove):
        pdb_path = os.path.join(source_folder, filename)
        if os.path.exists(pdb_path):
            os.remove(pdb_path)
            print(f"Removed {filename}.pdb")

train_folder = "your_train_pdb_folder"

val_folder = "your_test_pdb_folder"

find_and_remove_similar_sequences(train_folder,val_folder)
