import os
import pandas as pd
import torch
from torch.utils import data as torch_data
from torchdrug import data
from torchdrug.core import Registry as R
from tqdm import tqdm


# Use Monkey Patch (攔截 RDKit)


from rdkit import Chem
_original_read_pdb = Chem.MolFromPDBFile

def _hacked_read_pdb(*args, **kwargs):
    kwargs['sanitize'] = False
    kwargs['proximityBonding'] = False
    return _original_read_pdb(*args, **kwargs)


Chem.MolFromPDBFile = _hacked_read_pdb

@R.register("datasets.MCGLPPI")
class MCGLPPI(data.ProteinDataset):
    def __init__(self, train_csv, val_csv, test_csv, pdb_dir, transform=None, lazy=False):
        super().__init__()
        self.transform = transform
        self.lazy = lazy 

        self.data = []
        self.pdb_files = []
        self.targets = {'proaffinity_label': []} 
        self.num_samples = [] 

        for csv_file in [train_csv, val_csv, test_csv]:
            df = pd.read_csv(csv_file)
            valid_count = 0
            
            print(f"\n Loading file: {csv_file}")
            for _, row in tqdm(df.iterrows(), total=len(df)):
                pdb_code = row['pdb_code']
                pdb_file = os.path.join(pdb_dir, f"{pdb_code}.pdb")
                
                if not os.path.exists(pdb_file):
                    print(f"Error, cannot find {pdb_file}")
                    continue
                
                try:
                    protein = data.Protein.from_pdb(pdb_file)
                    
                    self.pdb_files.append(pdb_file)
                    self.targets['proaffinity_label'].append(row['proaffinity_label'])
                    
                    if not lazy:
                        self.data.append(protein)
                    valid_count += 1
                except Exception as e:
                    print(f"Skip {pdb_code}: {e}")
                    
            self.num_samples.append(valid_count)
            print(f" {valid_count} files are successful, from {csv_file} ")

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

    def get_item(self, index):
        if self.lazy:
            protein = data.Protein.from_pdb(self.pdb_files[index])
        else:
            protein = self.data[index].clone()
            
        item = {"graph": protein, "proaffinity_label": self.targets["proaffinity_label"][index]}
        if self.transform:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.pdb_files)