# PPI Affinity Prediction Benchmark on MCGLPPI

This repository provides a standardized benchmarking framework for protein-protein interaction (PPI) affinity prediction. We evaluate and compare three deep-learning baselines under a unified evaluation protocol using the **MCGLPPI** dataset (1,270 PPI complexes).

To assess sensitivity to data partitioning and training randomness, all models are evaluated on **five seeded train/validation/test splits** (nominally 8:1:1). This is a repeated holdout evaluation, not k-fold cross-validation. The exact split sizes and any grouping rule used to prevent homologous overlap should be documented in the split-generation script.

---

## I. Repository Structure

* `data/`: Contains the 1,270 raw PDB complexes and the 5 standardized random seed splits (seed 0, 1, 42, 142, 4242).
* `baselines/`: Contains the customized training and evaluation scripts for the three models.
* `results/`: Contains the raw console output logs showing the final Mean ± Std metrics for each model.

---

## II. Evaluated Baselines

We benchmarked the following three distinct architectural approaches:
1. **ProAffinity** (PLM + Graph Neural Network)
2. **Graphomer** (Pretrained PLM + Graph Transformer)
3. **GearNet-Res** (Residue-scale Geometry-Aware Relational GNN)

*(Note: The models are kept within their original structural frameworks, but their data loaders and evaluation metrics have been standardized to ensure a fair comparison on the exact same data splits.)*

---

## III. Environment Setup & Installation

Due to the fundamental architectural differences (and different PyTorch version requirements) among the three models, **we highly recommend using independent Conda environments for each model** on a Linux server (e.g., Ubuntu + AutoDL) with NVIDIA GPUs.

Please navigate to the respective folders in `baselines/` and follow the specific installation instructions below.


### 1. ProAffinity Environment

```bash
conda create -n proaffinity python=3.8
conda activate proaffinity
# Install base PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# Install PyG
pip install torch_geometric==2.3.0
# Install the rest of the dependencies
cd baselines/ProAffinity
pip install -r requirements.txt
```

### 2. Graphomer Environment

```bash
conda create -n graphomer python=3.9
conda activate graphomer
# Install base PyTorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# Install specific torch-scatter wheel (Crucial!)
wget https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.1.2+pt21cu121-cp39-cp39-linux_x86_64.whl
# Install the rest of the dependencies
cd baselines/Graphomer
pip install -r requirements.txt
```


### 3. GearNet-Res Environment

Requires an older PyTorch ecosystem

```bash
conda create -n gearnet python=3.8
conda activate gearnet
# Install base PyTorch (Modify CUDA version according to your machine)
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1 -c pytorch -c conda-forge
# Install PyG dependencies manually (Crucial!)
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
# Install the rest of the dependencies
cd baselines/GearNet-Res
pip install -r requirements.txt
```

## IV.  Benchmark Results

All models were evaluated based on their Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Pearson Correlation Coefficient ($R_p$), and Spearman Correlation Coefficient ($R_s$) across 5 random seeds.

---
  

|**Model**|**Pearson (Rp​)**|**Spearman (Rs​)**|**RMSE**|**MAE**|
|---|---|---|---|---|
|**ProAffinity**|0.5293 ± 0.0548|0.4915 ± 0.0495|1.7273 ± 0.1369|1.3539 ± 0.1309|
|**Graphomer**|0.5396 ± 0.0599|**0.5403 ± 0.0402**|**1.5681 ± 0.0492**|**1.2108 ± 0.0678**|
|**GearNet-Res**|**0.5725 ± 0.1015**|0.5125 ± 0.0809|1.5790 ± 0.0692|1.2231 ± 0.0270|

> _Detailed training logs and raw outputs can be found in the `results/` directory._

---

## V. How to Run

1. Generate the 5-seed splits by running the customized split generation script (already provided in `data/`).
    
2. Activate the corresponding Conda environment.
    
3. Run the training script in the model's directory. Each script is configured to automatically loop through the 5 seeds and print the aggregated results upon completion.

---

## VI. Important details

### For ProAffinity:

#### i. Procedures
- Run through all `.ipynb` in chronological order in `ProAffinity_Test/` (Ensure you have your raw pdbs from MCGLPPI).
- Then, go to `ProAffinity_Test/` for graph construction, by running `pdb2graph.py` and `pdb2graph_individual.py` FIRST, and run `graph_construct.py` and `graph_construct_indi.py` later. 
  - If you have other splits, you may change the route til `train_model` in **line 122**
  - If you met the `Error: Failed to establish a new connection: [Errno 101] Network is unreachable`, try `export HF_ENDPOINT=https://hf-mirror.com`
- Run `train_model.py`
- **Preprocessing fixes applied:** We identified and patched two issues in the original graph-construction code:
  1. **FASTA Truncation:** The original code (`lines[1].strip()`) truncated fasta sequences at 80 characters, causing misaligned ESM features.
  2. **Padding Node Corruption:** The original code included isolated padding nodes (missing 3D coordinates in PDB) into the graph. We applied PyG's `remove_isolated_nodes` to prevent these ghost nodes from corrupting the ESM features during the GNN Global Pooling phase.

#### ii. Model Optimization & Bottleneck Analysis

**1. Re-evaluating the baseline (potential homology leakage):**
The original ProAffinity paper reported **$R_p$ = 0.811** on a SKEMPI mutant subset. Random example-level splitting can place variants of the same wild-type complex, or closely homologous proteins, across training and test sets and may therefore inflate performance. In this benchmark, ProAffinity obtains **$R_p$ = 0.5293 ± 0.0548** across the five supplied splits. A zero-shot or non-homologous claim should be made only after the clustering criterion is documented and the absence of cluster overlap is verified.

**2. Hyperparameter Sensitivity (The "Armor Mismatch"):**
We initially applied GearNet-style hyperparameters (`BatchNorm`, `Batch Size 16`) to ProAffinity. This degraded performance in our runs (RMSE increased from approximately 1.7 to above 2.2), suggesting sensitivity to batch size and normalization. We therefore retained the model's native setup: `Batch Size 64`, no `BatchNorm`, and standard `MSELoss`.

**3. Conclusion (The Architectural Bottleneck):**
After correcting the ESM preprocessing and configuring the learning-rate scheduler to monitor validation $R_p$, ProAffinity reached **$R_p$ = 0.5293 ± 0.0548** under the current protocol.

The current results do not isolate architecture from tuning as the cause of the performance differences. They instead motivate controlled ablations of geometric inputs, model capacity, and optimization settings.

### For Graphomer:

#### i. Procedures

- Duplicate the `PPIdataindex.txt` generated in ProAffinity, then run `I. pdbs.ipynb` and `II. merge.ipynb` in order. (You might need to change the route in `I. pdbs.ipynb` targeting `MCGLPPI_RawData/`)

- Run `train_final.py` 

#### ii. Model Optimization & Bottleneck

In contrast to ProAffinity's lightweight GNN, Graphomer uses a Transformer-based architecture. Our optimization efforts focused on computational and memory bottlenecks (I/O and VRAM) caused by the $O(N^2)$ complexity of self-attention over long protein sequences (`pro_len = 2000`):

* **Data Pipeline & RAM Bottleneck Resolution:** Prevented catastrophic system SWAP lockups by abandoning full-dataset RAM loading. Implemented a strict memory management protocol (using Python `gc.collect()` and dynamic disk deletion) to isolate the memory footprint of each seed.
* **VRAM Tuning & Dimension Alignment:** Scaled down the Batch Size to 8 to fit within 24GB VRAM limits, and corrected the internal feature embedding projection (`d_embed = 64`) to perfectly align with the pre-processed ESM sequence features and 3D structural coordinates.
* **Architectural Enhancement (Spatial & Edge Encoding):** Graphomer extends the standard self-attention mechanism by incorporating graph-specific structural information (shortest path distances) and edge features as learned biases. This additive formulation ensures that both feature similarity and structural proximity are considered in the attention computation.



#### iii. Analysis

Classical message-passing neural networks can be affected by limited expressivity and oversmoothing. Under the current five-split protocol, ProAffinity obtains a mean $R_p$ of **0.5293 ± 0.0548**.

Graphomer partially addresses the locality limitation of message-passing GNNs through a global receptive field, allowing each node (residue) to attend to other nodes while incorporating structural biases into the attention score.

Across the five splits, Graphomer obtains a mean $R_p$ of **0.5396 ± 0.0599**, compared with **0.5293 ± 0.0548** for ProAffinity. The overlap in variability means that these results alone do not establish a statistically significant architectural advantage. They are consistent with the hypothesis that structural biases may help on some subsets, which should be tested using paired per-split results and confidence intervals or an appropriate paired test.



### For GearNet-Res:

#### Procedures

- Change your yaml in `config/downstream/`, and run `run.py` in `GearNet/`


#### Model Optimization & Bottleneck Analysis

GearNet-Res operates as a 3D-aware multi-relational heterogeneous GNN. To push its performance in this benchmark, we relied on its core architectural features while confronting its inherent computational limits:

* **Line-Graph Augmented Message Passing:** Unlike ProAffinity's standard node-aggregation, GearNet-Res employs an edge message passing mechanism to inject relative positional information between interactive edges, effectively capturing complex 3D geometric spatial structures.


* **The "Dense Edge" Memory Wall:** A critical bottleneck of this architecture is its graph construction method. By defining connections via geometric distance and sequential thresholds rather than concise chemical interactions, the resulting graphs become extremely dense with edges. This indiscriminately connected structure leads to massive memory consumption during neighboring message aggregation, frequently triggering Out-Of-Memory (OOM) errors and restricting the model to very small batch sizes.



## VII. Conclusion: The Residue-Scale Dilemma (High Peak, Extreme Instability)

The five-split benchmark suggests possible trade-offs among the evaluated representations.

**GearNet-Res** achieved the highest mean Pearson correlation in our evaluation ($R_p$ = **0.5725**), although its standard deviation (**± 0.1015**) was larger than those of ProAffinity and Graphomer. With only five splits, this should be described as higher split sensitivity rather than definitive evidence of instability or architectural superiority.

One possible contributor is the information loss introduced by **residue-scale modeling**: mapping amino acids to coarse residue-level nodes may omit side-chain orientations and atom-level interaction details. Other contributors—including dataset composition, optimization variance, and hyperparameter sensitivity—have not yet been ruled out.

Per-split error analysis is needed to determine whether the observed variation is associated with interface composition, structural quality, protein-family distribution, or fine-grained side-chain interactions.

Overall, the results motivate a more targeted analysis of when geometric representations help and whether atom-level chemical features can improve robustness.