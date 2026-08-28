# Environment Setup

Use one Conda environment per neural baseline. Do not combine the three environments and do not install the old full `pip freeze` files directly.

Run all commands from the repository root.

## ProAffinity

```bash
conda env create -f environments/proaffinity.yml
conda activate proaffinity

python -m pip install \
  torch==2.2.1 \
  torchvision==0.17.1 \
  torchaudio==2.2.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements/proaffinity.txt

python -c "import torch, torch_geometric, transformers, pandas, scipy; print('ProAffinity environment PASS'); print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

## Graphomer

```bash
conda env create -f environments/graphomer.yml
conda activate ppi-graphomer

python -m pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp39-cp39-linux_x86_64.whl

python -m pip install -r requirements/graphomer.txt

python -c "import torch, torch_scatter, esm, Bio, numpy, pandas; print('Graphomer environment PASS'); print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

The Graphomer model itself is trained from scratch. `fair-esm` is required because preprocessing extracts ESM2 and ESM-IF1 features.

## GearNet

GearNet uses the oldest PyTorch stack and must remain isolated from the other environments.

```bash
conda env create -f environments/gearnet.yml
conda activate gearnet

python -m pip install \
  torch==1.8.1+cu111 \
  torchvision==0.9.1+cu111 \
  torchaudio==0.8.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install \
  torch-scatter==2.0.8 \
  torch-cluster==1.5.9 \
  -f https://data.pyg.org/whl/torch-1.8.1+cu111.html

python -m pip install -r requirements/gearnet.txt

python -c "import torch, torch_scatter, torch_cluster, torchdrug, rdkit, pandas; print('GearNet environment PASS'); print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

## ESM2

The tested environment is Python 3.10, PyTorch 2.1.2 with CUDA 11.8, and Transformers 4.37.2.

```bash
conda env create -f environments/esm2.yml
conda activate esm2

python -m pip install \
  torch==2.1.2 \
  torchvision==0.16.2 \
  torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r requirements/esm2.txt

python -c "import torch, transformers; print('ESM2 environment PASS'); print('Torch:', torch.__version__, 'Transformers:', transformers.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
```

## Shared preprocessing and MMseqs2

The Python preprocessing scripts use the standard library. Open Babel and MMseqs2 are external command-line programs.

```bash
conda env create -f environments/preprocess.yml
conda activate ppi-preprocess

obabel -V
mmseqs version
```

## FoldX and Rosetta

The batch and analysis scripts use the Python standard library, so FoldX and Rosetta share one lightweight environment. Their separately licensed binaries must still be installed in the paths documented in the root README.

```bash
conda env create -f environments/physics.yml
conda activate ppi-physics

python --version
test -x software/foldx/foldx5/foldx_20261231 && echo "FoldX executable PASS"
test -x software/rosetta/current/source/bin/InterfaceAnalyzer.static.linuxgccrelease && echo "Rosetta executable PASS"
test -d software/rosetta/current/database && echo "Rosetta database PASS"
```
