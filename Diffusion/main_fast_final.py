import argparse
import logging
import random
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, EsmForMaskedLM
from collections import Counter
import wandb
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from denoising_diffusion_pytorch_2d_cross_attn_film import Unet, GaussianDiffusion, Trainer, BioDataset_modified, decode_embeddings_to_sequence
import difflib
import re

#########################################
# Helper Functions
#########################################

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def normalize_name(name):
    """
    Normalize peptide names consistently for both CSV and embeddings dictionary.
    """
    name = name.lower()  
    name = re.sub(r'[^\w\s]', '_', name)  
    name = re.sub(r'\s+', '_', name)  
    name = re.sub(r'_+', '_', name) 
    name = name.strip('_')  
    name = name.replace("peptide_containing_the_bh3_regions_from_", "region_bh3_") 
    return name




# --- Peptide extraction helper for Biolip ---
def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)

def preprocess_peptide_data(peptide_df, esm_model, esm_tokenizer, args):
    encoded_sequences = []
    tokenized_sequences = []
    sequences = []
    for idx in range(len(peptide_df)):
        row = peptide_df.iloc[idx, :]
        sequence = row['Peptide Sequence'] if 'Peptide Sequence' in row else row['Sequence']
        encoded_input = esm_tokenizer(sequence, 
                                      add_special_tokens=True,
                                      max_length=64,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')
        encoded_input = encoded_input.to(args.device)
        with torch.no_grad():
            # Forward pass through the model
            output = esm_model(**encoded_input)
            # Use hidden_states to get the embeddings
            hidden_states = output.hidden_states[-1]  # Use the last layer's hidden states
            embedding = hidden_states.squeeze().cpu().numpy()
        encoded_sequences.append(embedding)

        sequences.append(sequence)
        
        # Tokenize the sequence
        tokenized_seq = esm_tokenizer(sequence, add_special_tokens=True, max_length=64, padding='max_length', truncation=True).input_ids
        tokenized_sequences.append(tokenized_seq)
        
    return np.array(encoded_sequences), np.array(tokenized_sequences), np.array(sequences)

def load_features(file_path):
    return pd.read_csv(file_path)

def decode_embeddings(embeddings, decoder, tokenizer, args):
    """
    Decode embeddings using the decoder and tokenizer.
    """
    decoded_sequences = []
    device = args.computation_device
    decoder.eval()
    with torch.no_grad():
        embedding = embeddings
        embedding = torch.tensor(embedding).to(args.device)
        decoded_output = decoder(embedding)
        decoded_tokens = torch.argmax(decoded_output, dim=-1)
        if decoded_tokens.ndim == 1:
            decoded_tokens = decoded_tokens.unsqueeze(0)
        decoded_tokens = decoded_tokens.tolist()
        for item in decoded_tokens:
            if not isinstance(item, list):
                item = [item]
            decoded_sequence = tokenizer.decode(
                [token for token in item if token != tokenizer.pad_token_id],
                skip_special_tokens=True,
            )
            decoded_sequence = decoded_sequence.replace(" ", "")  
            decoded_sequences.append(decoded_sequence)
    return decoded_sequences

def find_closest_match(generated_seq, original_sequences):
    """Find the closest matching original sequence to the generated sequence."""
    best_match = None
    highest_similarity = 0
    for original_seq in original_sequences:
        similarity = difflib.SequenceMatcher(None, generated_seq, original_seq).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = original_seq
    return best_match


def load_pt_file(path):
    """Load a Torch checkpoint with backwards-compatible flags."""
    if not path:
        return {}
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_embedding_path(root, dist, model, pooling, conditioning_mode, seed):
    if conditioning_mode not in {"full", "pocket"}:
        return None
    suffix = "full" if conditioning_mode == "full" else "pocket"
    candidate = os.path.join(root, dist, model, f"pool_{pooling}", f"emb_{suffix}_all_seed={seed}.pt")
    if not os.path.exists(candidate):
        raise FileNotFoundError(f"Expected embedding file not found: {candidate}")
    return candidate


def load_embedding_dict(path):
    if not path:
        return {}
    payload = load_pt_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding checkpoint at {path} is not a dict.")
    mapped = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Embedding entry for {key} is not a tensor.")
        mapped[normalize_name(str(key))] = value.detach().cpu().to(torch.float32)
    return mapped




def conditioning_label(mode: str) -> str:
    mapping = {"full": "full", "pocket": "pocket", "none": "unconstrained", "physchem": "physchem"}
    if mode is None:
        return "unknown"
    key = mode.lower()
    return mapping.get(key, key)


def load_physchem_dict(path, *, id_column=None):
    if not path:
        raise ValueError("Physicochemical CSV path must be provided for physchem conditioning.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Physicochemical CSV not found: {path}")

    df = pd.read_csv(path)

    candidate_ids = [id_column, "pdb_id", "PDB_ID", "Name", "name", "id"]
    id_col = next((col for col in candidate_ids if col and col in df.columns), None)
    if id_col is None:
        raise KeyError("Could not determine ID column in physchem CSV; provide a column named 'pdb_id' or 'Name'.")

    preferred_features = [
        "Net_Charge_at_pH_7",
        "Charge",
        "Isoelectric_Point",
        "Hydrophobicity_at_pH_2",
        "Hydrophobicity_at_pH_6.8",
        "GRAVY",
        "MW",
    ]
    feature_cols = [col for col in preferred_features if col in df.columns]
    if not feature_cols:
        raise KeyError(
            "Physchem CSV is missing expected feature columns (e.g., Net_Charge_at_pH_7, Isoelectric_Point, GRAVY, MW)."
        )

    subset = df[[id_col] + feature_cols].copy()
    subset = subset.dropna(subset=feature_cols, how="all")
    subset[id_col] = subset[id_col].astype(str)

    feature_array = subset[feature_cols].to_numpy(dtype=np.float32, copy=False)
    ids = subset[id_col].tolist()

    mapping = {}
    for raw_id, vector in zip(ids, feature_array):
        if not raw_id:
            continue
        key = normalize_name(raw_id)
        mapping[key] = torch.from_numpy(np.array(vector, dtype=np.float32))

    if not mapping:
        raise ValueError(f"No valid feature rows found in {path}.")
    return mapping

def records_from_dataframe(df, id_col, seq_col):
    records = []
    for row in df.itertuples():
        pdb_id = str(getattr(row, id_col)).strip()
        sequence = str(getattr(row, seq_col)).strip()
        if not pdb_id or not sequence:
            continue
        records.append({
            "pdb_id": pdb_id,
            "sequence": sequence,
            "lookup_key": normalize_name(pdb_id),
        })
    return records


def load_biolip_records(split_dir):
    splits = {}
    for split_name in ("train", "val", "test"):
        split_path = os.path.join(split_dir, f"biolip_split_{split_name}.csv")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing Biolip split CSV: {split_path}")
        df = pd.read_csv(split_path)
        seq_col = "peptide_seq" if "peptide_seq" in df.columns else "Peptide Sequence"
        splits[split_name] = records_from_dataframe(df, "pdb_id", seq_col)
    return splits


def load_acp_records(csv_dir, split_json=None):
    frames = []
    for split_name in ("train", "val", "test"):
        split_path = os.path.join(csv_dir, f"split_{split_name}.csv")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing ACP split CSV: {split_path}")
        df = pd.read_csv(split_path)
        df["__split__"] = split_name
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    seq_candidates = ["Peptide Sequence", "peptide_seq", "sequence", "Sequence"]
    seq_col = next((col for col in seq_candidates if col in meta.columns), None)
    if seq_col is None:
        raise KeyError("Unable to locate peptide sequence column in ACP metadata.")
    meta["__sequence__"] = meta[seq_col].astype(str)
    meta["__pdb_id__"] = meta["pdb_id"].astype(str)
    meta["__lookup__"] = meta["__pdb_id__"].map(normalize_name)
    lookup = {
        row["__lookup__"]: {"pdb_id": row["__pdb_id__"], "sequence": row["__sequence__"]}
        for _, row in meta.iterrows()
    }

    splits = {}
    if split_json and os.path.exists(split_json):
        with open(split_json, "r") as handle:
            payload = json.load(handle)
        for split_name in ("train", "val", "test"):
            recs = []
            for pid in payload.get(split_name, []):
                key = normalize_name(str(pid))
                if key not in lookup:
                    raise KeyError(f"Split {split_name} references unknown peptide id: {pid}")
                recs.append({
                    "pdb_id": lookup[key]["pdb_id"],
                    "sequence": lookup[key]["sequence"],
                    "lookup_key": key,
                })
            splits[split_name] = recs
    else:
        for split_name in ("train", "val", "test"):
            subset = meta[meta["__split__"] == split_name]
            splits[split_name] = [
                {
                    "pdb_id": row["__pdb_id__"],
                    "sequence": row["__sequence__"],
                    "lookup_key": row["__lookup__"],
                }
                for _, row in subset.iterrows()
            ]
    return splits


def encode_records(records, esm_model, esm_tokenizer, args):
    if not records:
        return (
            np.zeros((0, 64, 1280), dtype=np.float32),
            np.zeros((0, 64), dtype=np.int64),
            np.empty((0,), dtype="<U1"),
        )
    df = pd.DataFrame({"Sequence": [rec["sequence"] for rec in records]})
    encoded, tokenized, sequences = preprocess_peptide_data(df, esm_model, esm_tokenizer, args)
    return (
        encoded.astype(np.float32, copy=False),
        tokenized.astype(np.int64, copy=False),
        sequences.astype("<U64", copy=False),
    )


def build_feature_matrix(records, embedding_lookup, conditioning_mode):
    if conditioning_mode == "none" or not records:
        return np.zeros((len(records), 0), dtype=np.float32)
    features = []
    missing = []
    for rec in records:
        key = rec["lookup_key"]
        tensor = embedding_lookup.get(key)
        if tensor is None:
            missing.append(rec["pdb_id"])
            continue
        features.append(tensor.numpy())
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"Missing embeddings for: {preview}{'...' if len(missing) > 5 else ''}")
    if not features:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(features, axis=0).astype(np.float32, copy=False)


def normalize_feature_splits(train, val, test):
    if train.size == 0 or train.shape[1] == 0:
        zero = np.zeros((0,), dtype=np.float32)
        one = np.ones((0,), dtype=np.float32)
        return train, val, test, zero, one
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)

    def _standardize(array):
        if array.size == 0:
            return array
        return ((array - mean) / std).astype(np.float32, copy=False)

    return _standardize(train), _standardize(val), _standardize(test), mean.astype(np.float32), std.astype(np.float32)


def unpack_cached_dataset(npz_handle):
    payload = {}
    for split_name in ("train", "val", "test"):
        payload[split_name] = {
            "encoded": npz_handle[f"encoded_{split_name}"].astype(np.float32, copy=False),
            "tokens": npz_handle[f"token_{split_name}"].astype(np.int64, copy=False),
            "raw": npz_handle[f"raw_{split_name}"],
            "features": npz_handle[f"feat_{split_name}"].astype(np.float32, copy=False),
            "ids": npz_handle.get(f"ids_{split_name}", np.empty((0,), dtype="<U1")),
        }
    payload["feat_mean"] = npz_handle.get("feat_mean", np.zeros((0,), dtype=np.float32))
    payload["feat_std"] = npz_handle.get("feat_std", np.ones((0,), dtype=np.float32))
    context_entry = npz_handle.get("context_dim")
    if context_entry is not None:
        payload["context_dim"] = int(np.asarray(context_entry).flat[0])
    else:
        payload["context_dim"] = payload["train"]["features"].shape[1]
    payload["conditioning_mode"] = str(np.asarray(npz_handle.get("conditioning_mode", ["unknown"]))[0])
    return payload


def prepare_phase_dataset(args, phase, esm_model, esm_tokenizer):
    conditioning_mode = args.conditioning_mode.lower()
    cache_dir = os.path.abspath(args.dataset_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if phase == "pretrain":
        split_records = load_biolip_records(args.biolip_split_dir)
        seed_tag = args.pretrain_embedding_seed
        cache_name = f"biolip_{args.dist}_{conditioning_mode}_seed{seed_tag}.npz"
        if conditioning_mode == "physchem":
            phys_path = args.biolip_physchem_csv
            embedding_lookup = load_physchem_dict(phys_path)
        else:
            embedding_path = resolve_embedding_path(
                args.gnn_embedding_biolip_root,
                args.dist,
                args.gnn_model,
                args.gnn_pooling,
                conditioning_mode,
                seed_tag,
            )
            embedding_lookup = load_embedding_dict(embedding_path)
    else:
        split_records = load_acp_records(args.acp_split_csv_dir, args.acp_split_json)
        seed_tag = args.finetune_embedding_seed
        cache_name = f"{args.finetune_type}_{args.dist}_{conditioning_mode}_seed{seed_tag}_split{args.acp_split_seed}.npz"
        if conditioning_mode == "physchem":
            phys_path = args.acp_physchem_csv
            embedding_lookup = load_physchem_dict(phys_path)
        else:
            embedding_path = resolve_embedding_path(
                args.gnn_embedding_root,
                args.dist,
                args.gnn_model,
                args.gnn_pooling,
                conditioning_mode,
                seed_tag,
            )
            embedding_lookup = load_embedding_dict(embedding_path)

    cache_path = os.path.join(cache_dir, cache_name)

    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        try:
            return unpack_cached_dataset(cached)
        finally:
            cached.close()

    payload = {}
    flat_store = {}
    logger = logging.getLogger(__name__)
    for split_name in ("train", "val", "test"):
        records = split_records.get(split_name, [])
        original_count = len(records)
        if conditioning_mode != "none" and records:
            kept_records = []
            missing_records = []
            for rec in records:
                if rec["lookup_key"] in embedding_lookup:
                    kept_records.append(rec)
                else:
                    missing_records.append(rec)
            if missing_records:
                preview = ", ".join(rec["pdb_id"] for rec in missing_records[:5])
                logger.warning(
                    "Skipping %d/%d %s samples without conditioning embeddings (e.g., %s)",
                    len(missing_records), original_count, split_name, preview
                )
            records = kept_records
        encoded, tokenized, sequences = encode_records(records, esm_model, esm_tokenizer, args)
        features = build_feature_matrix(records, embedding_lookup, conditioning_mode)
        ids = np.array([rec["pdb_id"] for rec in records], dtype="<U64")

        payload[split_name] = {
            "encoded": encoded,
            "tokens": tokenized,
            "raw": sequences,
            "features": features,
            "ids": ids,
        }

    train_feats, val_feats, test_feats, feat_mean, feat_std = normalize_feature_splits(
        payload["train"]["features"],
        payload["val"]["features"],
        payload["test"]["features"],
    )

    payload["train"]["features"] = train_feats
    payload["val"]["features"] = val_feats
    payload["test"]["features"] = test_feats
    payload["feat_mean"] = feat_mean
    payload["feat_std"] = feat_std
    payload["context_dim"] = train_feats.shape[1] if train_feats.size else 0
    payload["conditioning_mode"] = conditioning_mode

    for split_name in ("train", "val", "test"):
        split_payload = payload[split_name]
        flat_store[f"encoded_{split_name}"] = split_payload["encoded"]
        flat_store[f"token_{split_name}"] = split_payload["tokens"]
        flat_store[f"raw_{split_name}"] = split_payload["raw"]
        flat_store[f"feat_{split_name}"] = split_payload["features"]
        flat_store[f"ids_{split_name}"] = split_payload["ids"]

    flat_store["feat_mean"] = feat_mean
    flat_store["feat_std"] = feat_std
    flat_store["context_dim"] = np.array([payload["context_dim"]], dtype=np.int64)
    flat_store["conditioning_mode"] = np.array([conditioning_mode], dtype="<U16")
    flat_store["phase_tag"] = np.array([phase], dtype="<U16")
    flat_store["dist_tag"] = np.array([args.dist], dtype="<U16")
    flat_store["embedding_seed"] = np.array([seed_tag], dtype=np.int64)

    np.savez_compressed(cache_path, **flat_store)
    return payload

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

def initialize_wandb(run_name, config, group=None):
    wandb.init(
        project="ACP_DIFFUSION_2D_TEST",
        name=run_name,
        config=config,
        reinit=True,  
        group=group  
    )
    wandb.define_metric("Decoder Train Loss", step_metric="epoch")
    wandb.define_metric("Decoder Validation Loss", step_metric="epoch")
    wandb.define_metric("Decoder Validation Sequence Accuracy", step_metric="epoch")
    wandb.define_metric("Decoder Validation Character Accuracy", step_metric="epoch")
    wandb.define_metric("Decoder Validation Average Levenshtein Distance", step_metric="epoch")
    wandb.define_metric("Decoder Validation Average BLEU Score", step_metric="epoch")
    wandb.define_metric("Validation Loss", step_metric="train_step")
    wandb.define_metric("Train Loss", step_metric="train_step")

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

    loss = F.mse_loss(encoded_generated.float(), encoded_ori.float())
    return loss.item()


def ensure_directory(path):
    """Create directory (and parents) if it does not already exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def get_split_file_paths(base_dir, split_name, sequences_filename, properties_filename):
    """Build output file paths for a given dataset split."""
    split_suffix = split_name.lower()
    generated_dir = os.path.join(base_dir, split_suffix, "generated")
    properties_dir = os.path.join(base_dir, split_suffix, "properties")
    ensure_directory(generated_dir)
    ensure_directory(properties_dir)
    generated_path = os.path.join(generated_dir, f"{split_suffix}_{sequences_filename}")
    properties_path = os.path.join(properties_dir, f"{split_suffix}_{properties_filename}")
    return generated_path, properties_path


def standardize_sequence_list(sequence_container):
    """Convert stored sequence containers (array/list/str) into a Python list of strings."""
    array_like = np.asarray(sequence_container)
    normalized = array_like.tolist()
    if isinstance(normalized, str):
        return [normalized]
    return [str(seq) for seq in normalized]


def sample_sequences_for_split(
    dataloader,
    diffusion_model,
    decoder,
    tokenizer,
    args,
    device,
    target_samples,
    initial_eval_size,
    quality_threshold,
):
    """Sample sequences conditioned on a dataloader until a target count is reached."""
    generated_sequences = []
    data_iter = iter(dataloader)
    passed_quality_gate = False

    while len(generated_sequences) < target_samples:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                seq, features, raw_seq = batch
            elif len(batch) == 2:
                seq, features = batch
                raw_seq = None
            else:
                seq = batch[0]
                features = batch[1] if len(batch) > 1 else None
                raw_seq = batch[-1] if len(batch) > 2 else None
        else:
            seq = batch
            features = None
            raw_seq = None

        seq = seq.to(device)
        context = None
        if isinstance(features, torch.Tensor):
            context = features.to(device)
            if context.ndim >= 3 and context.size(1) == 1:
                context = context.squeeze(1)
            if context.ndim >= 3 and context.size(-1) == 1:
                context = context.squeeze(-1)
            if context.numel() == 0:
                context = None

        sampled_seq = diffusion_model.sample(seq.shape[0], context=context).to(device)
        sampled_seq = sampled_seq.squeeze(1).permute(0, 2, 1)

        decoded_sequences = decode_embeddings(sampled_seq.cpu().numpy(), decoder.stage2, tokenizer, args)
        original_sequences = list(raw_seq) if raw_seq is not None else ["" for _ in decoded_sequences]

        for original, generated in zip(original_sequences, decoded_sequences):
            trimmed = generated[:len(original)] if original else generated
            generated_sequences.append(trimmed)
        print(f'Generated currently:{generated_sequences}')

    return generated_sequences[:target_samples]



def build_properties_dataframe(
    generated_sequences,
    reference_sequences,
    tokenizer,
    lm_model,
    device,
):
    """Compute physicochemical and language-model-based metrics for generated sequences."""
    peptide_properties = {
        'Sequence': [],
        'Reference_Sequence': [],
        'Net_Charge_at_pH_7': [],
        'Isoelectric_Point': [],
        'GRAVY': [],
        'Molecular_Weight': [],
        'BLEU': [],
        'Perplexity': [],
        'Reconstruction_Loss': []
    }

    if not generated_sequences:
        return pd.DataFrame(peptide_properties)

    for generated_seq in generated_sequences:
        closest_original = find_closest_match(generated_seq, reference_sequences)
        if not closest_original:
            continue

        props = calculate_properties(generated_seq)
        if props[0] is None:
            continue

        bleu_score = calculate_bleu(closest_original, generated_seq)
        perplexity = calculate_perplexity_v2(generated_seq, lm_model, tokenizer, device)
        reconstruction_loss = calculate_reconstruction_loss(closest_original, generated_seq, tokenizer)

        peptide_properties['Sequence'].append(generated_seq)
        peptide_properties['Reference_Sequence'].append(closest_original)
        peptide_properties['Net_Charge_at_pH_7'].append(props[0])
        peptide_properties['Isoelectric_Point'].append(props[1])
        peptide_properties['GRAVY'].append(props[2])
        peptide_properties['Molecular_Weight'].append(props[3])
        peptide_properties['BLEU'].append(bleu_score)
        peptide_properties['Perplexity'].append(perplexity)
        peptide_properties['Reconstruction_Loss'].append(reconstruction_loss)

    return pd.DataFrame(peptide_properties)


def filter_generated_sequences(generated_sequences, training_sequences):
    """Remove duplicates and sequences overlapping with the training set."""
    unique_sequences = []
    seen = set()
    training_set = set(training_sequences)

    for seq in generated_sequences:
        if seq is None:
            continue
        sequence = str(seq)
        if not sequence:
            continue
        if sequence in training_set or sequence in seen:
            continue
        seen.add(sequence)
        unique_sequences.append(sequence)

    return unique_sequences

#########################################
# Decoder and Diffusion Model Definitions
#########################################

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.embedding_bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.rnn_bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        output, _ = self.rnn(x)
        output = self.rnn_bn(output.transpose(1, 2)).transpose(1, 2)
        output = self.fc(output)
        output = self.fc_bn(output.transpose(1, 2)).transpose(1, 2)
        return output

class Decoder_2_stage(nn.Module):
    def __init__(self, stage1, stage2):
        super(Decoder_2_stage, self).__init__()
        self.stage1 = stage1
        self.stage2 = stage2


    def forward(self, x):
        x = self.stage1(x)
        output = self.stage2(x)
        return output

def train_decoder(decoder, train_loader, valid_loader, model, tokenizer, args):
    initialize_wandb(f"Decoder Training -{args.save_decoder_model}", args, group="Decoder")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    device = args.computation_device
    decoder.to(device)

    for epoch in range(args.decoder_epochs):
        decoder.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            embeddings, sequences = batch
            embeddings = embeddings.to(device, non_blocking=True)
            sequences = sequences.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                outputs = decoder(embeddings)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), sequences.reshape(-1))

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulate_every == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({"Decoder Train Loss": avg_train_loss}, step=epoch)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}')

        decoder.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                embeddings, sequences = batch
                embeddings = embeddings.to(device, non_blocking=True)
                sequences = sequences.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    outputs = decoder(embeddings)
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), sequences.reshape(-1))
                    valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        wandb.log({"Decoder Validation Loss": avg_valid_loss}, step=epoch)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_valid_loss}')

        all_orig_seqs = []
        all_gen_seqs = []
        with torch.no_grad():
            for batch in valid_loader:
                embeddings, sequences = batch
                embeddings = embeddings.to(device, non_blocking=True)
                sequences = sequences.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    outputs = decoder(embeddings)
                decoded_tokens = torch.argmax(outputs, dim=-1).cpu().tolist()
                for orig, gen in zip(sequences.cpu().tolist(), decoded_tokens):
                    orig_seq = tokenizer.decode([token for token in orig if token != tokenizer.pad_token_id], skip_special_tokens=True)
                    gen_seq = tokenizer.decode([token for token in gen if token != tokenizer.pad_token_id], skip_special_tokens=True)

                    orig_seq = orig_seq.replace(" ", "")
                    gen_seq = gen_seq.replace(" ", "")

                    min_len = len(orig_seq)
                    orig_seq = orig_seq[:min_len]
                    gen_seq = gen_seq[:min_len]

                    all_orig_seqs.append(orig_seq)
                    all_gen_seqs.append(gen_seq)

        sequence_accuracy, char_accuracy, avg_perplexity, avg_entropy, js_3, js_6 = evaluate_decoder(
            all_orig_seqs,
            all_gen_seqs,
            model,
            tokenizer,
            args.computation_device,
        )
        wandb.log({
            "Validation - Sequence Accuracy": sequence_accuracy,
            "Validation - Character Accuracy": char_accuracy,
            "Validation - Average Perplexity": avg_perplexity,
            "Validation - Average Entropy": avg_entropy,
            "Validation - JS-3": js_3,
            "Validation - JS-6": js_6
        }, step=epoch)

        print(f"Validation - Sequence Accuracy: {sequence_accuracy * 100:.2f}%")
        print(f"Validation - Character Accuracy: {char_accuracy * 100:.2f}%")
        print(f"Validation - Average Perplexity: {avg_perplexity:.2f}")
        print(f"Validation - Average Entropy: {avg_entropy:.2f}")
        print(f"Validation - JS-3: {js_3:.2f}")
        print(f"Validation - JS-6: {js_6:.2f}")


def test_decoder(decoder, test_loader, model, tokenizer, device):
    decoder.eval()
    all_orig_seqs = []
    all_gen_seqs = []
    with torch.no_grad():
        for batch in test_loader:
            embeddings, sequences = batch
            embeddings = embeddings.to(device, non_blocking=True)
            sequences = sequences.to(device, non_blocking=True)
            #print('test_decode embedding shape:', embeddings.shape)
            outputs = decoder(embeddings)
            decoded_tokens = torch.argmax(outputs, dim=-1).cpu().tolist()
            for orig, gen in zip(sequences.cpu().tolist(), decoded_tokens):
                orig_seq = tokenizer.decode([token for token in orig if token != tokenizer.pad_token_id], skip_special_tokens=True)
                gen_seq = tokenizer.decode([token for token in gen if token != tokenizer.pad_token_id], skip_special_tokens=True)

                orig_seq = orig_seq.replace(" ", "")
                gen_seq = gen_seq.replace(" ", "")

                min_len = len(orig_seq)
                orig_seq = orig_seq[:min_len]
                gen_seq = gen_seq[:min_len]

                all_orig_seqs.append(orig_seq)
                all_gen_seqs.append(gen_seq)
    print(f"test - all_orig_seqs: {all_orig_seqs}")
    print(f"test - all_gen_seqs: {all_gen_seqs}")

    sequence_accuracy, char_accuracy, avg_perplexity, avg_entropy, js_3, js_6 = evaluate_decoder(
        all_orig_seqs,
        all_gen_seqs,
        model,
        tokenizer,
        device,
    )
    print(f"Test - Sequence Accuracy: {sequence_accuracy * 100:.2f}%")
    print(f"Test - Character Accuracy: {char_accuracy * 100:.2f}%")
    print(f"Test - Average Perplexity: {avg_perplexity:.2f}")
    print(f"Test - Average Entropy: {avg_entropy:.2f}")
    print(f"Test - JS-3: {js_3:.2f}")
    print(f"Test - JS-6: {js_6:.2f}")

def evaluate_decoder(original_sequences, generated_sequences, model, tokenizer, device):
    """
    Evaluate the decoder using various metrics, including pseudo-perplexity.
    """
    # Sequence accuracy
    sequence_matches = [1 if orig == gen else 0 for orig, gen in zip(original_sequences, generated_sequences)]
    sequence_accuracy = np.mean(sequence_matches)
    
    # Character accuracy
    lengths = 0
    counter = 0
    for orig, gen in zip(original_sequences, generated_sequences):
        lengths = lengths + len(orig)
        for i, j in zip(orig, gen):
            if i == j:
                counter += 1
 
    char_accuracy = counter / lengths
    
    # Perplexity
    perplexities = [calculate_perplexity_v2(seq, model, tokenizer, device) for seq in generated_sequences]
    #print(perplexities)
    avg_perplexity = np.mean(perplexities)
    
    # Entropy
    entropies = [calculate_entropy(gen) for gen in generated_sequences]
    avg_entropy = np.mean(entropies)
    
    # Jaccard Similarity (JS-3 and JS-6)
    js_3 = calculate_jaccard_similarity(original_sequences, generated_sequences, 3)
    js_6 = calculate_jaccard_similarity(original_sequences, generated_sequences, 6)

    return sequence_accuracy, char_accuracy, avg_perplexity, avg_entropy, js_3, js_6
    #return sequence_accuracy, char_accuracy, avg_entropy, js_3, js_6

#########################################
# Main Diffusion Model Training Function
#########################################

def main(args):
    set_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
    device = args.device

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("medium")

    args.computation_device = device
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    precision_map = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    sequence_dtype = precision_map.get(args.dataset_dtype, torch.float32)
    feature_dtype = sequence_dtype

    conditioning_mode = (args.conditioning_mode or "").lower()
    if conditioning_mode == "physchem":
        esm_model = EsmForMaskedLM.from_pretrained(
            "facebook/esm2_t33_650M_UR50D",
            output_hidden_states=True,
        )
        esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    else:
        if args.phase == "pretrain":
            fine_tuned_model_path = "/home2/s230112/BIB/Diffusion/fine_tuned_esm2_biolip/epoch_5"
        else:
            fine_tuned_model_path = f"/home2/s230112/BIB/Diffusion/fine_tuned_esm2_{args.finetune_type}/epoch_5"
        esm_model = EsmForMaskedLM.from_pretrained(
            fine_tuned_model_path,
            output_hidden_states=True,
        )
        esm_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

    esm_model = esm_model.to(device)
    
    
    ######################################################
    # PHASE-SPECIFIC DATA PROCESSING (TRANSFER LEARNING) #
    ######################################################
    dataset_payload = prepare_phase_dataset(args, args.phase, esm_model, esm_tokenizer)

    train_split = dataset_payload["train"]
    val_split = dataset_payload["val"]
    test_split = dataset_payload["test"]

    encoded_peptides_train = train_split["encoded"]
    tokenized_sequences_train = train_split["tokens"]
    raw_sequences_train = train_split["raw"]
    feature_values_train = train_split["features"]

    encoded_peptides_val = val_split["encoded"]
    tokenized_sequences_val = val_split["tokens"]
    raw_sequences_val = val_split["raw"]
    feature_values_val = val_split["features"]

    encoded_peptides_test = test_split["encoded"]
    tokenized_sequences_test = test_split["tokens"]
    raw_sequences_test = test_split["raw"]
    feature_values_test = test_split["features"]

    context_dim = int(dataset_payload.get("context_dim", feature_values_train.shape[1] if feature_values_train.ndim == 2 else 0))
    conditioning_mode_effective = dataset_payload.get("conditioning_mode", args.conditioning_mode.lower())
    logger.info(
        "Loaded %s dataset with %d/%d/%d graphs (conditioning: %s, context_dim=%s)",
        args.phase,
        encoded_peptides_train.shape[0],
        encoded_peptides_val.shape[0],
        encoded_peptides_test.shape[0],
        conditioning_mode_effective,
        context_dim if context_dim else "None",
    )

    args.context_dim = context_dim

    conditioning_label_effective = conditioning_label(conditioning_mode_effective)
    phase_label = 'biolip' if args.phase == 'pretrain' else args.finetune_type
    if args.context_type == 'film':
        run_tag = f"fast_{conditioning_label_effective}_film_{phase_label}"
    else:
        run_tag = f"fast_{conditioning_label_effective}_{phase_label}"
    run_root = os.path.join('/home2/s230112/BIB/Diffusion', run_tag)
    results_folder = os.path.join(run_root, 'models')
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    base_output_dir = run_root
    if args.context_type == 'film':
        pretrain_root = os.path.join('/home2/s230112/BIB/Diffusion', f"fast_{conditioning_label_effective}_film_biolip")
    else:
        pretrain_root = os.path.join('/home2/s230112/BIB/Diffusion', f"fast_{conditioning_label_effective}_biolip")
    pretrain_models_dir = os.path.join(pretrain_root, 'models')
    args.run_root = run_root
    args.results_folder = results_folder
    args.pretrain_models_dir = pretrain_models_dir


    train_dataset = BioDataset_modified(
        encoded_peptides_train,
        feature_values_train,
        raw_sequences_train,
        sequence_dtype=sequence_dtype,
        feature_dtype=feature_dtype,
    )
    valid_dataset = BioDataset_modified(
        encoded_peptides_val,
        feature_values_val,
        raw_sequences_val,
        sequence_dtype=sequence_dtype,
        feature_dtype=feature_dtype,
    )
    test_dataset = BioDataset_modified(
        encoded_peptides_test,
        feature_values_test,
        raw_sequences_test,
        sequence_dtype=sequence_dtype,
        feature_dtype=feature_dtype,
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Valid dataset size: {len(valid_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    ####################
    # Decoder Training #
    ####################
    # Initialize the two-stage decoder 
    decoder_s1 = Decoder(input_dim=1280, hidden_dim=256, output_dim=128, num_layers=1).to(device)
    decoder_s2 = Decoder(input_dim=128, hidden_dim=256, output_dim=len(esm_tokenizer), num_layers=1).to(device)
    decoder = Decoder_2_stage(stage1=decoder_s1, stage2=decoder_s2)
    
    # Build decoder datasets (mapping embeddings to tokenized sequences)
    class PeptideDataset(Dataset):
        def __init__(self, embeddings, sequences, embedding_dtype=torch.float32):
            self.embeddings = embeddings
            self.sequences = sequences
            self.embedding_dtype = embedding_dtype

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            embedding = torch.as_tensor(self.embeddings[idx], dtype=self.embedding_dtype)
            sequence = torch.as_tensor(self.sequences[idx], dtype=torch.long)
            return embedding, sequence

    decoder_train_embeddings = encoded_peptides_train 
    decoder_train_sequences = tokenized_sequences_train 
    decoder_valid_embeddings = encoded_peptides_val
    decoder_valid_sequences = tokenized_sequences_val
    decoder_test_embeddings = encoded_peptides_test 
    decoder_test_sequences = tokenized_sequences_test
    
    decoder_train_dataset = PeptideDataset(decoder_train_embeddings, decoder_train_sequences, embedding_dtype=sequence_dtype)
    decoder_valid_dataset = PeptideDataset(decoder_valid_embeddings, decoder_valid_sequences, embedding_dtype=sequence_dtype)
    decoder_test_dataset = PeptideDataset(decoder_test_embeddings, decoder_test_sequences, embedding_dtype=sequence_dtype)
    
    def build_loader(dataset, *, batch_size, shuffle, num_workers, prefetch_factor=None):
        num_workers = max(0, int(num_workers))
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": args.pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
        return DataLoader(dataset, **loader_kwargs)

    decoder_train_loader = build_loader(
        decoder_train_dataset,
        batch_size=args.decoder_batch_size,
        shuffle=True,
        num_workers=args.decoder_num_workers,
        prefetch_factor=args.train_prefetch_factor,
    )
    decoder_valid_loader = build_loader(
        decoder_valid_dataset,
        batch_size=args.decoder_batch_size,
        shuffle=False,
        num_workers=args.decoder_num_workers,
        prefetch_factor=args.valid_prefetch_factor,
    )
    decoder_test_loader = build_loader(
        decoder_test_dataset,
        batch_size=args.decoder_batch_size,
        shuffle=False,
        num_workers=args.decoder_num_workers,
        prefetch_factor=args.valid_prefetch_factor,
    )
    
    if args.train_decoder.upper() == "Y":
        if args.phase == "pretrain":
            # Pretrain the decoder on BioLiP
            logger.info("Pretraining the decoder on Biolip dataset.")
            train_decoder(decoder, decoder_train_loader, decoder_valid_loader, esm_model, esm_tokenizer, args)

            # Save the pretrained decoder
            if args.save_decoder_model:
                save_path = os.path.join("/home2/s230112/BIB/Diffusion/decoder_models", f"{args.save_decoder_model}_biolip.pt")
                torch.save(decoder.state_dict(), save_path)
                logger.info(f"Pretrained decoder saved as {save_path}")


        if args.phase == "finetune":
            # Load the pretrained decoder from BioLiP
            pretrained_model_path = os.path.join("/home2/s230112/BIB/Diffusion/decoder_models", f"{args.save_decoder_model}_biolip.pt")
            if os.path.exists(pretrained_model_path):
                decoder.load_state_dict(torch.load(pretrained_model_path))
                logger.info(f"Pretrained Biolip decoder loaded from {pretrained_model_path} for fine-tuning.")
            else:
                logger.warning(f"Pretrained Biolip decoder not found. Training from scratch.")

            # Fine-tune on ACP–BCL-xL data
            logger.info(f"Fine-tuning the decoder on ACP–{args.finetune_type} dataset.")
            train_decoder(decoder, decoder_train_loader, decoder_valid_loader, esm_model, esm_tokenizer, args)

            # Save the fine-tuned decoder
            if args.save_decoder_model:
                save_path = os.path.join("/home2/s230112/BIB/Diffusion/decoder_models", f"{args.save_decoder_model}_{args.finetune_type}.pt")
                torch.save(decoder.state_dict(), save_path)
                logger.info(f"Fine-tuned decoder saved as {save_path}")

        test_decoder(decoder, decoder_test_loader, esm_model, esm_tokenizer, device)

        # Testing the decoder which was saved previously
        print("Running load_state_dict")
        decoder.load_state_dict(torch.load(save_path))
        decoder.to(device)
        decoder.eval()
        logger.info(f"Decoder model loaded from {save_path}")
        #test_decoder(decoder, decoder_test_loader, esm_model, esm_tokenizer, device)
        #print("test test decoder help sos on same file")
        exit()
        
    else:
        print("Loading the decoder for evaluation only")
        # Load the decoder for evaluation only
        model_path = os.path.join("/home2/s230112/BIB/Diffusion/decoder_models", f"{args.load_decoder_model}.pt")

        if os.path.exists(model_path):
            decoder.load_state_dict(torch.load(model_path))
            decoder.to(device)
            logger.info(f"Decoder model loaded from {model_path}")
            test_decoder(decoder, decoder_test_loader, esm_model, esm_tokenizer, device)
                       
        else:
            logger.error(f"Decoder model {model_path} not found! Cannot proceed with testing.")
            exit(1)

    
    #########################################
    # Diffusion Model Training
    #########################################
    # Use the first stage of your decoder as the constraint
    decoder_constraint = decoder.stage1
    context_dim_unet = context_dim if context_dim > 0 else None
    #print(context_dim_unet)
    #exit()
    model_unet = Unet(
        dim=args.model_dim,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        context_dim=context_dim_unet,
        context_type=args.context_type
    ).to(device)

    #print(model_unet)
    #exit()
    
    diffusion = GaussianDiffusion(
        model_unet,
        image_size=(128, 64),
        timesteps=args.noising_timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective='pred_noise'
    ).to(device)
    
    trainer = Trainer(
        diffusion_model=diffusion,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        train_lr=args.learning_rate,
        train_num_steps=args.train_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        amp=args.mixed_precision,
        save_and_sample_every=args.save_and_sample_every,
        specific_save_steps=[550, 4000, 9000],
        device=device,
        decoder=decoder_constraint,
        train_num_workers=args.train_num_workers,
        valid_num_workers=args.valid_num_workers,
        train_prefetch_factor=args.train_prefetch_factor,
        valid_prefetch_factor=args.valid_prefetch_factor,
        pin_memory=args.pin_memory,
        results_folder=results_folder
    )
    
    
    if args.train_steps > 0:
        if args.phase == "pretrain":
            initialize_wandb(f"CBM-full-cross-attn-{args.save_model}_{phase_label}", args, group="Diffusion")
        
        #Uncomment this if you want continue PRETRAIN from an existing model
            # if args.load_model:
            #     trainer.load(args.load_model)
        
            trainer.train()
            
            if args.save_model:
                trainer.save(f"{args.save_model}_{phase_label}")  # Save the pretrained model
                print(f"Saved pretrained diffusion model as {args.save_model}_{phase_label}")

                exit()

        if args.phase == "finetune":

            #load directory
            load_directory_pt = pretrain_models_dir  # biolip directory
            load_model_dir = os.path.join(load_directory_pt, f"model-{args.load_model}.pt")

            pretrained_diffusion_model_path = load_model_dir

            print(f"Loading model from path {pretrained_diffusion_model_path} path exist:{os.path.exists(pretrained_diffusion_model_path)}")
            if os.path.exists(pretrained_diffusion_model_path):
                trainer.load("",pretrained_diffusion_model_path)  # Load the pretrained model
                logger.info(f"Loaded pretrained diffusion model from {pretrained_diffusion_model_path} for fine-tuning.")
                initialize_wandb(f"CBM-Full-cross-attn-p2-{args.save_model}_{phase_label}", args, group="Diffusion")
                trainer.step = 0
                trainer.train()
                
                if args.save_model:
                    trainer.save(f"{args.save_model}_{phase_label}")  # Save the pretrained model
                    print(f"Saved pretrained diffusion model as {args.save_model}_{phase_label}")
            else:
                logger.warning("Pretrained diffusion model not found. Training from scratch.")
    
    else:
        trainer.load(args.load_model)
    torch.cuda.empty_cache()

    valid_dataloader = build_loader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.valid_num_workers,
        prefetch_factor=args.valid_prefetch_factor,
    )

    test_dataloader = build_loader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        prefetch_factor=args.valid_prefetch_factor,
    )

    base_output_dir = run_root
    training_sequences = set(standardize_sequence_list(raw_sequences_train))

    split_definitions = [
        ("validation", valid_dataloader, standardize_sequence_list(raw_sequences_val)),
        ("test", test_dataloader, standardize_sequence_list(raw_sequences_test)),
    ]

    for split_name, dataloader, reference_sequences in split_definitions:
        sequences_file, properties_file = get_split_file_paths(
            base_output_dir,
            split_name,
            args.gen_seqs_filename,
            args.gen_prop_filename,
        )

        if os.path.exists(sequences_file):
            generated_sequences = (
                pd.read_csv(sequences_file)['Generated_Sequence']
                .dropna()
                .astype(str)
                .tolist()
            )
            print(f"Generated sequences have been loaded from {sequences_file}")
        else:
            generated_sequences = []



        filtered_sequences = filter_generated_sequences(generated_sequences, training_sequences)
        properties_df = build_properties_dataframe(
            filtered_sequences,
            reference_sequences,
            esm_tokenizer,
            esm_model,
            device,
        )

        partial_interval = max(0, int(args.partial_save_interval))
        last_partial_save = (len(filtered_sequences) // partial_interval) if partial_interval > 0 else 0

        while (
            len(properties_df) < args.min_good_peptides
            and len(filtered_sequences) < args.num_samples
        ):
            new_sequences = sample_sequences_for_split(
                dataloader,
                diffusion,
                decoder,
                esm_tokenizer,
                args,
                device,
                args.sample_chunk_size,
                args.initial_eval_size,
                args.quality_threshold,
            )

            if not new_sequences:
                print(
                    f"Unable to reach {args.min_good_peptides} high-quality peptides for the {split_name} split."
                )
                break

            generated_sequences.extend(new_sequences)
            filtered_sequences = filter_generated_sequences(generated_sequences, training_sequences)
            properties_df = build_properties_dataframe(
                filtered_sequences,
                reference_sequences,
                esm_tokenizer,
                esm_model,
                device,
            )

            if partial_interval > 0:
                current_block = len(filtered_sequences) // partial_interval
                if current_block > last_partial_save:
                    pd.DataFrame({'Generated_Sequence': filtered_sequences}).to_csv(
                        sequences_file,
                        index=False,
                    )
                    print(
                        f"[Partial save] {len(filtered_sequences)} sequences written to {sequences_file}"
                    )
                    last_partial_save = current_block

        filtered_sequences = filter_generated_sequences(generated_sequences, training_sequences)
        properties_df = build_properties_dataframe(
            filtered_sequences,
            reference_sequences,
            esm_tokenizer,
            esm_model,
            device,
        )

        if len(properties_df) < args.min_good_peptides:
            print(
                f"Collected {len(properties_df)} peptides for the {split_name} split; "
                f"target was {args.min_good_peptides}."
            )

        pd.DataFrame({'Generated_Sequence': filtered_sequences}).to_csv(
            sequences_file,
            index=False,
        )
        print(f"Generated sequences have been saved to {sequences_file}")

        properties_df.to_csv(properties_file, index=False)
        print(
            "Generated sequences and their physicochemical properties "
            f"have been saved to {properties_file}"
        )

        if len(properties_df) >= args.best_top_k:
            best_df = (
                properties_df
                .sort_values(['BLEU', 'Perplexity'], ascending=[False, True])
                .head(args.best_top_k)
            )
            best_file = os.path.join(
                os.path.dirname(properties_file),
                f"{split_name.lower()}_best_{args.best_top_k}.csv"
            )
            best_df.to_csv(best_file, index=False)

            bleu_mean = best_df['BLEU'].mean()
            bleu_std = best_df['BLEU'].std()
            perplexity_mean = best_df['Perplexity'].mean()
            perplexity_std = best_df['Perplexity'].std()

            print(
                f"{split_name.title()} best {args.best_top_k} BLEU mean={bleu_mean:.4f}, std={bleu_std:.4f}; "
                f"Perplexity mean={perplexity_mean:.4f}, std={perplexity_std:.4f}"
            )
        else:
            print(
                f"Only {len(properties_df)} peptides available for the {split_name} split; "
                f"skipping best-{args.best_top_k} selection."
            )

    # Calculate properties for the train dataset
    train_properties = {
        'Sequence': [],
        'Net_Charge_at_pH_7': [],
        'Isoelectric_Point': [],
        'GRAVY': [],
        'Molecular_Weight': []
    }

    for seq in standardize_sequence_list(raw_sequences_train):
        props = calculate_properties(seq)
        if props[0] is None:
            continue  # Skip sequences with non-standard amino acids
        train_properties['Sequence'].append(seq)
        train_properties['Net_Charge_at_pH_7'].append(props[0])
        train_properties['Isoelectric_Point'].append(props[1])
        train_properties['GRAVY'].append(props[2])
        train_properties['Molecular_Weight'].append(props[3])

    train_peptide_df = pd.DataFrame(train_properties)
    train_peptide_df.to_csv("train_peptide_with_physicochemical_properties.csv", index=False)
    print("Training sequences and their physicochemical properties have been saved to train_peptide_with_physicochemical_properties.csv")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a 1D denoising diffusion model with transfer learning.")
    # Common arguments
    parser.add_argument("--phase", type=str, default="pretrain", choices=["pretrain", "finetune"],
                        help="Select phase: pretrain on Biolip or finetune on ACP–BCL-xL")
    parser.add_argument("--peptide-data", type=str, default="cleaned_peptide_data.csv", help="(Used in finetune phase)")
    parser.add_argument("--feature-data", type=str, default="cleaned_joined_data.csv", help="Feature data CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    parser.add_argument("--model-dim", type=int, default=64, help="Dimension of the UNet model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for diffusion training")
    parser.add_argument("--valid-batch-size", type=int, default=32, help="Validation batch size")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--train-steps", type=int, default=1, help="Number of training steps")
    parser.add_argument("--gradient-accumulate-every", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--ema-decay", type=float, default=0.995, help="EMA decay")
    parser.add_argument("--mixed-precision", action='store_true', help="Enable mixed precision training")
    parser.add_argument("--test-batch-size", type=int, default=4, help="Test batch size")
    parser.add_argument("--num-samples", type=int, default=6000, help="Maximum number of unique sequences retained per split during sampling")
    parser.add_argument("--sample-chunk-size", type=int, default=512, help="Number of sequences to draw per sampling round")
    parser.add_argument("--initial-eval-size", type=int, default=100, help="Generated sample count for the early quality gate")
    parser.add_argument("--quality-threshold", type=float, default=0.8, help="Quality threshold for the early stopping gate")
    parser.add_argument("--min-good-peptides", type=int, default=5000, help="Minimum number of peptides with valid properties required per split")
    parser.add_argument("--partial-save-interval", type=int, default=100,
                        help="Write intermediate generated-sequence CSV every N entries (set 0 to disable)")
    parser.add_argument("--best-top-k", type=int, default=1000, help="Number of top peptides to retain based on BLEU and Perplexity")
    parser.add_argument("--save-model", type=str, help="Filename to save the trained diffusion model")
    parser.add_argument("--load-model", type=str, help="Filename to load a diffusion model")
    parser.add_argument("--sampling-timesteps", type=int, default=200, help="Sampling timesteps")
    parser.add_argument("--noising-timesteps", type=int, default=200, help="Noising timesteps")
    parser.add_argument("--save-and-sample-every", type=int, default=1000, help="Frequency for saving and sampling")
    parser.add_argument("--train-num-workers", type=int, default=4, help="Number of workers for training dataloader")
    parser.add_argument("--valid-num-workers", type=int, default=2, help="Number of workers for validation dataloader")
    parser.add_argument("--test-num-workers", type=int, default=2, help="Number of workers for test dataloader")
    parser.add_argument("--decoder-num-workers", type=int, default=0, help="Number of workers for decoder dataloaders")
    parser.add_argument("--train-prefetch-factor", type=int, default=2, help="Prefetch factor for train dataloaders")
    parser.add_argument("--valid-prefetch-factor", type=int, default=2, help="Prefetch factor for validation/test dataloaders")
    parser.add_argument("--dist", type=str, default="8A", choices=["6A", "8A"], help="Distance cutoff associated with prepared splits.")
    parser.add_argument("--conditioning-mode", default="full",
                    choices=["full", "pocket", "none", "physchem"])
    parser.add_argument("--biolip-physchem-csv", type=str,
                        default="/home2/s230112/BIB/Ablation/ICASSP/biolip_physchem_all.csv",
                        help="CSV with Biolip physicochemical features")
    parser.add_argument("--acp-physchem-csv", type=str,
                        default="/home2/s230112/BIB/Ablation/ICASSP/acp_physchem_all.csv",
                        help="CSV with ACP physicochemical features")

    parser.add_argument("--dataset-cache-dir", type=str, default="/home2/s230112/BIB/Diffusion/cache", help="Directory to cache prepared diffusion datasets.")
    parser.add_argument("--biolip-split-dir", type=str, default=None, help="Biolip split CSV directory.")
    parser.add_argument("--acp-split-csv-dir", type=str, default=None, help="ACP split CSV directory.")
    parser.add_argument("--acp-split-json", type=str, default=None, help="Optional ACP split JSON path.")
    parser.add_argument("--acp-split-seed", type=int, default=42, help="Seed identifier encoded in ACP split JSON.")
    parser.add_argument("--pretrain-embedding-seed", type=int, default=42, help="Seed used for Biolip GNN embeddings.")
    parser.add_argument("--finetune-embedding-seed", type=int, default=42, help="Seed used for ACP GNN embeddings.")
    parser.add_argument("--gnn-model", type=str, default="GAT", help="GNN backbone used when exporting embeddings.")
    parser.add_argument("--gnn-pooling", type=str, default="attention", help="Pooling strategy used for exported embeddings.")
    parser.add_argument("--gnn-embedding-root", type=str, default="GNN/pooling_ablation/embeddings", help="Root path for ACP embedding checkpoints.")
    parser.add_argument("--gnn-embedding-biolip-root", type=str, default="GNN/pooling_ablation/embeddings_biolip", help="Root path for Biolip embedding checkpoints.")
    parser.add_argument("--dataset-dtype", type=str, default="float32", choices=["float32", "float16"], help="Storage precision for diffusion datasets")
    parser.add_argument("--no-pin-memory", action='store_true', help="Disable pinned memory for dataloaders")
    # Decoder arguments
    parser.add_argument("--decoder-epochs", type=int, default=10, help="Decoder training epochs")
    parser.add_argument("--decoder-batch-size", type=int, default=32, help="Decoder batch size")
    parser.add_argument("--train-decoder", type=str, default="N", help="Train the decoder? (Y/N)")
    parser.add_argument("--save-decoder-model", type=str, help="Filename to save the trained decoder model")
    parser.add_argument("--load-decoder-model", type=str, help="Filename to load a saved decoder model")
    
    args = parser.parse_args()
    args.dist = args.dist.upper()
    args.conditioning_mode = args.conditioning_mode.lower()
    if args.biolip_split_dir is None:
        args.biolip_split_dir = f"/home2/s230112/BIB/GNN/splits_biolip_{args.dist}"
    if args.acp_split_csv_dir is None:
        args.acp_split_csv_dir = f"/home2/s230112/BIB/GNN/splits_bclxl_{args.dist}"
    if args.acp_split_json is None:
        candidate_json = f"/home2/s230112/BIB/GNN/splits_generated/{args.dist}_seed{args.acp_split_seed}.json"
        args.acp_split_json = candidate_json if os.path.exists(candidate_json) else None
    args.dataset_cache_dir = os.path.abspath(args.dataset_cache_dir)
    args.gnn_embedding_root = os.path.abspath(args.gnn_embedding_root)
    args.gnn_embedding_biolip_root = os.path.abspath(args.gnn_embedding_biolip_root)
    if args.conditioning_mode == "physchem":
        if args.biolip_physchem_csv is None:
            raise ValueError("--biolip-physchem-csv must be provided when conditioning-mode=physchem")
        if args.acp_physchem_csv is None:
            raise ValueError("--acp-physchem-csv must be provided when conditioning-mode=physchem")
        args.biolip_physchem_csv = os.path.abspath(args.biolip_physchem_csv)
        args.acp_physchem_csv = os.path.abspath(args.acp_physchem_csv)
    # Optionally override some arguments:
    args.device_id = 0
    args.device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    args.pin_memory = not args.no_pin_memory
    args.gradient_accumulate_every = 1
    args.mixed_precision = True#False

    args.finetune_type = 'bcl2'# acp or bcl2 or mcl1

    ### DECODER MODEL ###
    args.decoder_batch_size = 4
    args.train_decoder = ("N").upper()
    if not args.load_decoder_model:
        args.load_decoder_model = "np_biolip" if args.phase == "pretrain" else f"np_{args.finetune_type}"
    #args.save_decoder_model = "np"
    #args.decoder_epochs = 5
    

    #### DIFFUSION MODEL ###
    # To train and save the model, put a model name to args.save_model and timesteps to args.train_steps
    # To load the model and skip Training, args.load_model the model name and set args.train_steps to 0
    # To load the model and continue training, load the current model name, save it to a new name, and set timesteps
    args.train_steps = 0
    args.batch_size = 4
    args.valid_batch_size = 4
    args.test_batch_size = 4
    # args.save_model = "550"
    args.load_model = "550"  
    args.context_type = 'cross-attention' #options are 'cross-attention' or 'film' or 'icassp'
    
    ###Sampling###
    args.num_samples = 6000
    args.sample_chunk_size = 2048
    args.initial_eval_size = 100
    args.quality_threshold = 0.8
    args.min_good_peptides = 5000
    args.best_top_k = 1000
    

    args.gen_seqs_filename = "generated_dataset_journal_550.csv"
    args.gen_prop_filename = "generated_properties_journal_550.csv"

    args.sampling_timesteps = 500
    args.noising_timesteps = 1000
    
    main(args)
    
    exit()

    # Initialize wandb before the loop if needed
    #wandb.init(project="ACP_DIFFUSION_2D_TEST", name="Sampling", reinit=True)

    # Define the list of model checkpoints you want to sample from
    model_checkpoints = [300]
    #model_checkpoints = range(53, 63)

    for model_number in model_checkpoints:
        args.load_model = str(model_number)
        args.gen_seqs_filename = f"generated_dataset_journal_{model_number}.csv"   
        args.gen_prop_filename = f"generated_properties_journal_{model_number}.csv"
        #wandb.log({"model_number": model_number})
        try:
            print(f"Processing model {model_number}")
            main(args)
        except Exception as e:
            print(f"Failed to process model {model_number}: {e}")
