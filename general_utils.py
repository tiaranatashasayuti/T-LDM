import argparse
import logging
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn
from collections import Counter
import torch.optim as optim
from transformers import AutoTokenizer, EsmForMaskedLM
import wandb
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
import torch.nn.functional as F



import re
from Bio.PDB import PDBParser

class PeptideDataset(Dataset):
    def __init__(self, embeddings, sequences):
        self.embeddings = embeddings
        self.sequences = sequences

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.sequences[idx])


class BioDataset_modified(Dataset):
    def __init__(self, sequences, features=None, raw_sequences=None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.sequences = self.sequences.permute(0, 2, 1)
        self.raw_sequences = raw_sequences  # Store raw sequences if provided

        if features is not None:
            self.features = torch.tensor(features, dtype=torch.float32)
            self.features = self.features.unsqueeze(1)
        else:
            self.features = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.features is not None and self.raw_sequences is not None:
            return self.sequences[idx].clone(), self.features[idx].clone(), self.raw_sequences[idx]
        elif self.features is not None:
            return self.sequences[idx].clone(), self.features[idx].clone()
        else:
            return self.sequences[idx].clone()


def initialize_wandb(run_name, config, group=None):
    #this initialization is catered for decoder training and diffusion model training
    wandb.init(
        project="ACP_DIFFUSION_2D_TEST",
        name=run_name,
        config=config,
        reinit=True,  # Ensures a new run is started
        group=group  # Optional: Group related runs together
    )
    wandb.define_metric("Decoder Train Loss", step_metric="epoch")
    wandb.define_metric("Decoder Validation Loss", step_metric="epoch")
    wandb.define_metric("Decoder Validation Sequence Accuracy", step_metric="epoch")
    wandb.define_metric("Decoder Validation Character Accuracy", step_metric="epoch")
    wandb.define_metric("Decoder Validation Average Levenshtein Distance", step_metric="epoch")
    wandb.define_metric("Decoder Validation Average BLEU Score", step_metric="epoch")
    wandb.define_metric("Validation Loss", step_metric="train_step")
    wandb.define_metric("Train Loss", step_metric="train_step")


def preprocess_peptide_data(peptide_df, esm_model, esm_tokenizer, args):
    encoded_sequences = []
    tokenized_sequences = []
    sequences = []
    for idx in range(len(peptide_df)):
        row = peptide_df.iloc[idx, :]
        sequence = row['Peptide Sequence'] if 'Peptide Sequence' in row else row['Sequence']
        encoded_input = esm_tokenizer(sequence, add_special_tokens=True, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(args.device)

        with torch.no_grad():
            output = esm_model(**encoded_input)
            hidden_states = output.hidden_states[-1]
            embedding = hidden_states.squeeze().cpu().numpy()

        encoded_sequences.append(embedding)
        sequences.append(sequence)

        tokenized_seq = esm_tokenizer(sequence, add_special_tokens=True, max_length=64, padding='max_length', truncation=True).input_ids
        tokenized_sequences.append(tokenized_seq)

    return np.array(encoded_sequences), np.array(tokenized_sequences), np.array(sequences)


def normalize_name(name):
    """
    Normalize peptide names consistently for both CSV and embeddings dictionary.
    """
    name = name.lower()  # Convert to lowercase
    name = re.sub(r'[^\w\s]', '_', name)  # Replace non-alphanumeric characters with underscores
    name = re.sub(r'\s+', '_', name)  # Replace spaces with underscores
    name = re.sub(r'_+', '_', name)  # Remove multiple consecutive underscores
    name = name.strip('_')  # Remove leading and trailing underscores
    name = name.replace("peptide_containing_the_bh3_regions_from_", "region_bh3_")  # Replace common phrase
    return name

def extract_peptide_sequence(pdb_id, receptor_chain, ligand_chain):
    """
    Given a pdb_id and chain identifiers, load the PDB file from your Biolip structures folder,
    and extract (for example) the receptor chain’s peptide sequence.
    """
    from Bio.PDB import PDBParser
    STRUCTURE_FOLDER = "/home2/s230112/BIB/GNN/train_data"  
    pdb_file_path = os.path.join(STRUCTURE_FOLDER, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_file_path):
        print(f"PDB file {pdb_file_path} not found.")
        return None
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("Complex", pdb_file_path)
    seq = ""
    # Here we extract the receptor chain’s sequence (as an example)
    mapping = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
               "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
               "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
               "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}
    for chain in structure.get_chains():
        if chain.get_id() == ligand_chain:
            for res in chain.get_residues():
                if res.id[0] == ' ' and res.get_resname() in mapping:
                    seq += mapping[res.get_resname()]
            break
    return seq if seq != "" else None

def load_features(file_path):
    return pd.read_csv(file_path)


def calculate_bleu(reference, generated):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], generated, smoothing_function=smoothie)

def calculate_perplexity_v2(seq, model, tokenizer, device):
    model.eval()
    tensor_input = tokenizer.encode(seq, return_tensors='pt')
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    
    # mask one by one except [CLS] and [SEP]
    mask = torch.ones(tensor_input.size(-1) -1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)

    labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)

    with torch.no_grad():
        loss = model(masked_input.to(device), labels=labels.to(device)).loss
    #print("Current loss:",loss.item())
    return np.exp(loss.item())

def calculate_entropy(sequence):
    counts = Counter(sequence)
    probabilities = [count / len(sequence) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities)

def calculate_jaccard_similarity(original, generated, k):
    def get_kmers(seq, k):
        return set(seq[i:i + k] for i in range(len(seq) - k + 1))
    
    def jaccard_similarity(set1, set2):
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0
    
    kmer_sets_ori = [get_kmers(seq, k) for seq in original]
    kmer_sets_gen = [get_kmers(seq, k) for seq in generated]
    
    similarities = []
    for ori_set, gen_set in zip(kmer_sets_ori, kmer_sets_gen):
        similarity = jaccard_similarity(ori_set, gen_set)
        similarities.append(similarity)
    
    average_similarity = sum(similarities) / len(similarities) if similarities else 0
    return average_similarity

def evaluate_sequences(generated_sequences, quality_threshold=0.5):
    total_score = 0
    count = 0
    
    for seq in generated_sequences:
        if seq is None:
            continue
        props = calculate_properties(seq)
        if props[0] is None:
            continue  # Skip sequences with non-standard amino acids
        
        good_charge = -5.0 <= props[0] <= 9.0
        good_isoelectric_point = 4.0 <= props[1] <= 13.0
        good_gravy = -2.0 <= props[2] <= 2.0
        good_molecular_weight = 2000 <= props[3] <= 4000

        score = sum([good_charge, good_isoelectric_point, good_gravy, good_molecular_weight]) / 4.0
        total_score += score
        count += 1
    
    avg_score = total_score / count if count > 0 else 0
    return avg_score >= quality_threshold


def calculate_properties(seq):
    if any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq):
        return None, None, None, None  # Skip sequences with non-standard amino acids
    analysis = ProteinAnalysis(seq)
    net_charge = analysis.charge_at_pH(7)
    isoelectric_point = analysis.isoelectric_point()
    gravy = analysis.gravy()
    molecular_weight = analysis.molecular_weight()
    return net_charge, isoelectric_point, gravy, molecular_weight

def calculate_reconstruction_loss(original, reconstructed,tokenizer):
    #input of both ori and recon is sequence in string, we will be tokenizing it
    if len(original)>len(reconstructed):
        padding_length = len(original)
    else:
        padding_length = len(reconstructed)

    encoded_ori = tokenizer(original, 
                                add_special_tokens=True,
                                max_length=padding_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='pt').input_ids
    encoded_generated = tokenizer(reconstructed, 
                            add_special_tokens=True,
                            max_length=padding_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt').input_ids
    #print(original,len(original), reconstructed,len(reconstructed))
    #print(encoded_ori.shape,encoded_ori)
    loss = F.mse_loss(encoded_generated.float(), encoded_ori.float())
    return loss.item()
