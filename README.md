# 🧬 Binding Pocket-Aware GNN Pipeline for Therapeutic Peptide Design

This repository contains a complete preprocessing and training pipeline for paper titled, "Enhancing Latent Diffusion Models with Graph Neural Networks for Binding Pocket-Aware Therapeutic Peptide Design".

---

## 🚀 Pipeline Overview

### 1. **Extract Binding Site Info**

```bash
python extract_pocket_loc.py
```
- Input: `peptide_data.csv` with PDB IDs.
- Output: `input_features.csv` with receptor/ligand chains, sequences, and pocket residue indices.

---

### 2. **Preprocess Features**

```bash
python edit_data_preprocessing.py
```
- Input: `input_features.csv` and PDB files.
- Output: `features.pt` — per-residue graph node features and labels.

---

### 3. **Build Edge Indices**

```bash
python edge_index.py
```
- Input: Same complexes as step 2.
- Output: `edge_indexes.pt` with intra-complex residue–residue proximity pairs.

```bash
python edge_index_train.py
```
- Input: BioLiP.txt metadata + BioLiP PDB files.
- Output: `edge_indexes_train.pt` for pretraining.

---

### 4. **Train GNN Models**

#### (a) GAT-based Model
```bash
python train_GAT.py
```
- Pretrains on BioLiP, then fine-tunes on BCL-xL data.
- Saves:
  - `model_pretrained_coordinate_abalation_test.pth`
  - `model_finetuned_coordinate_abalation_test.pth`
  - `resulting_embeddings_final_[biolip|acp]_test.pt`
  - `resulting_embeddings_final_pocket_[biolip|acp]_test.pt`

#### (b) GCN-based Model (Ablation Study)
```bash
python train_GCN.py
```
- Same setup as above, but using GCNConv layers.
- Output files follow the same structure with `GCN_` prefix.

---


## 🧪 Downloads

XXX

---

## 🛠️ Environments

XXX

---

## 📊 Logging & Evaluation

- Training logs and metrics (Eg. loss, accuracy, AUC-ROC) are tracked with [Weights & Biases](https://wandb.ai/)
- Classification reports and embedding extraction are saved automatically.

---

## 🧠 Docking

HADDOCK3 was used for docking the generated peptides onto BCL-XL (PDB ID: 4QVE).
Instructions on how to use HADDOCK3 can be found here: [HADDOCK3](https://www.bonvinlab.org/software/haddock3/)

---

## 📌 Citation

If this code contributes to your research, please consider citing our work (manuscript submitted to *Briefings in Bioinformatics*).
