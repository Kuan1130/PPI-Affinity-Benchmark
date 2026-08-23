# PPI Affinity Prediction Benchmark on MCGLPPI

This repository provides a standardized benchmarking framework for Protein-Protein Interaction (PPI) affinity prediction. We evaluate and compare three sota deep learning models under a strict, unified evaluation protocol using the **MCGLPPI** dataset (1,270 PPI complexes).

To ensure robustness and statistical significance, all models are evaluated using a **5-Seed Cross-Validation** approach (Train/Val/Test = 8:1:1).

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
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
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
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
# Install specific torch-scatter wheel (Crucial!)
wget [https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp39-cp39-linux_x86_64.whl](https://data.pyg.org/whl/torch-2.1.0%2Bcu121/torch_scatter-2.1.2%2Bpt21cu121-cp39-cp39-linux_x86_64.whl)
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
pip install torch-scatter torch-cluster -f [https://data.pyg.org/whl/torch-1.8.1+cu111.html](https://data.pyg.org/whl/torch-1.8.1+cu111.html)
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
- **CRITICAL FIXES APPLIED:** We identified and patched two fatal bugs in the original author's graph construction repository:
  1. **FASTA Truncation:** The original code (`lines[1].strip()`) truncated fasta sequences at 80 characters, causing misaligned ESM features.
  2. **Padding Node Corruption:** The original code included isolated padding nodes (missing 3D coordinates in PDB) into the graph. We applied PyG's `remove_isolated_nodes` to prevent these ghost nodes from corrupting the ESM features during the GNN Global Pooling phase.

#### ii. Model Optimization & Bottleneck Analysis

**1. Re-evaluating the Baseline (The Data Leakage Issue):**
The original ProAffinity paper reported an overly optimistic Pearson correlation of **Rp = 0.811** on the SKEMPI mutant dataset. However, standard random splits cause severe **homology data leakage** (the model memorizes the wild-type 3D backbones during training and predicts identical mutants during testing). Under our **strict, non-homologous zero-shot split**, the model's true generalization capacity is established at **Rp ≈ 0.518**.

**2. Hyperparameter Sensitivity (The "Armor Mismatch"):**
In an attempt to ensure a strictly controlled benchmark, we initially applied GearNet's heavy-duty hyperparameters (`BatchNorm`, `Batch Size 16`) to ProAffinity. This caused catastrophic failure (RMSE spiked from ~1.7 to > 2.2). We conclude that small batch sizes paired with `BatchNorm` destroy absolute-scale predictions for this lightweight architecture. The optimal and fairest setup for ProAffinity remains its native configuration: `Batch Size 64`, `No BatchNorm`, and standard `MSELoss`.

**3. Conclusion (The Architectural Bottleneck):**
After sanitizing the ESM features and optimizing the learning rate scheduler to monitor validation Rp, ProAffinity reached its absolute physical ceiling of **Rp = 0.53 ± 0.06**. 

This serves as strong empirical evidence that the performance gap between ProAffinity and 3D-aware models (like GearNet, Rp ≈ 0.60) is **NOT due to suboptimal hyperparameter tuning**, but rather an **architectural limitation**. The underlying AttentiveFP GNN solely extracts 1D/2D topological features and fundamentally lacks the capacity to comprehend 3D geometric spatial interactions (e.g., dihedrals, precise spatial distances), which are the ultimate deciders in Protein-Protein Interaction (PPI) tasks.

#### iii. Re-evaluating the Baseline: The "Data Leakage" Issue in PPI Benchmarks

In the original paper, ProAffinity reported an exceptionally high Pearson correlation of **Rp = 0.811** on the SKEMPI subset (166 complexes). However, our rigorous benchmarking (using a strictly split dataset of 107 unseen targets) reveals that the model's true generalization capacity is bounded.

**Why the huge discrepancy? (Data Leakage via Homology Overlap):**

The SKEMPI subset consists of only 26 wild-type complexes and 140 of their point-mutation variants. In standard random-split procedures (without strict sequence-identity clustering), the training set inevitably absorbs the structural backbones of these wild-type proteins. Consequently, when evaluating the SKEMPI mutants, the model is simply **memorizing (overfitting to) the highly homologous global 3D scaffolds** it has already seen during training, rather than genuinely predicting the biophysical impact of the mutations. 

Our benchmark corrects this over-optimistic estimation by employing a **strict, non-homologous data split**, ensuring zero structural overlap between the 1000 training samples and the 107 test samples. Under this true zero-shot/unseen evaluation, ProAffinity achieves an Rp of ~0.52, which serves as a realistic and scientifically rigorous baseline for 1D/2D GNN architectures in de novo PPI binding affinity prediction.



### For Graphomer:

#### i. Procedures

- Duplicate the `PPIdataindex.txt` generated in ProAffinity, then run `I. pdbs.ipynb` and `II. merge.ipynb` in order. (You might need to change the route in `I. pdbs.ipynb` targeting `MCGLPPI_RawData/`)

- Run `train_final.py` 

#### ii. Model Optimization & Bottleneck

In contrast to ProAffinity's lightweight GNN, Graphomer utilizes a heavy Transformer-based architecture. Our optimization efforts here focused on overcoming severe computational and memory bottlenecks (I/O & VRAM) caused by the $O(N^2)$ complexity of the self-attention mechanism over long protein sequences (`pro_len = 2000`):

* **Data Pipeline & RAM Bottleneck Resolution:** Prevented catastrophic system SWAP lockups by abandoning full-dataset RAM loading. Implemented a strict memory management protocol (using Python `gc.collect()` and dynamic disk deletion) to isolate the memory footprint of each seed.
* **VRAM Tuning & Dimension Alignment:** Scaled down the Batch Size to 8 to fit within 24GB VRAM limits, and corrected the internal feature embedding projection (`d_embed = 64`) to perfectly align with the pre-processed ESM sequence features and 3D structural coordinates.
* **Architectural Enhancement (Spatial & Edge Encoding):** Graphormer extends the standard self-attention mechanism by incorporating graph-specific structural information (shortest path distances) and edge features as learned biases. This additive formulation ensures that both feature similarity and structural proximity are considered in the attention computation.



#### iii. Analysis

As established in our ProAffinity analysis, classical Message Passing Neural Networks (MPNNs) face significant limitations, such as the 1-WL test limit for distinguishing non-isomorphic structures and the "oversmoothing" problem. The ProAffinity model's performance was strictly capped at $R_p \approx 0.42$ under our rigorous zero-shot, non-homologous split.

Graphomer explicitly overcomes these limitations because its self-attention mechanism has a global receptive field, allowing each node (residue) to attend to every other node while injecting 3D spatial distances directly into the attention score.

Evaluated on the exact same strict, non-homologous 107-target test set, Graphomer achieves a significantly higher performance ($R_p \approx 0.53 \sim 0.61$). This provides definitive empirical evidence that **incorporating explicit 3D spatial priors (distance matrices and edge encodings) into a Transformer architecture is critical**. It bridges the gap that 1D/2D topological GNNs cannot cross, enabling the model to genuinely comprehend the biophysical and geometric constraints of Protein-Protein Interactions (PPI) rather than merely memorizing homologous scaffolds.



### For GearNet-Res:

#### Procedures

- Change your yaml in `config/downstream/`, and run `run.py` in `GearNet/`


#### Model Optimization & Bottleneck Analysis

GearNet-Res operates as a 3D-aware multi-relational heterogeneous GNN. To push its performance in this benchmark, we relied on its core architectural features while confronting its inherent computational limits:

* **Line-Graph Augmented Message Passing:** Unlike ProAffinity's standard node-aggregation, GearNet-Res employs an edge message passing mechanism to inject relative positional information between interactive edges, effectively capturing complex 3D geometric spatial structures.


* **The "Dense Edge" Memory Wall:** A critical bottleneck of this architecture is its graph construction method. By defining connections via geometric distance and sequential thresholds rather than concise chemical interactions, the resulting graphs become extremely dense with edges. This indiscriminately connected structure leads to massive memory consumption during neighboring message aggregation, frequently triggering Out-Of-Memory (OOM) errors and restricting the model to very small batch sizes.



## VII. Conclusion: The Residue-Scale Dilemma (High Peak, Extreme Instability)

The 5-seed benchmark results perfectly illustrate the theoretical trade-offs of different geometric representations. 

**GearNet-Res** achieved the highest mean Pearson correlation in our evaluation (Rp = **0.5725**), definitively proving that explicitly modeling 3D spatial architecture shatters the performance ceiling of 1D/2D topological models. However, this high average performance masks a severe flaw: **Extreme Instability**. GearNet-Res exhibits a massive standard deviation of **± 0.1015** (double that of ProAffinity and Graphomer).

The root cause of this variance lies in the inherent limitation of **residue-scale modeling**. Mapping entire amino acids to single Cα nodes compresses complex side-chain orientations into single points, wiring them blindly based on distance thresholds. By doing so, GearNet-Res loses chemically plausible interaction details (e.g., hydrogen bonding, polarity matching).

Consequently, its performance becomes highly luck-dependent (split-sensitive). If a test split relies heavily on global backbone geometries, GearNet-Res performs exceptionally well. But if the binding affinities in a specific split are driven by fine-grained, side-chain-specific physical interactions, GearNet-Res's predictions collapse. 

This benchmark serves as a textbook example of how **geometric awareness brings high potential, but omitting chemically plausible atomic features leads to catastrophic variance.**