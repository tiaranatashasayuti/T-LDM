import argparse
import logging
import random
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


import difflib
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from general_utils import preprocess_peptide_data,normalize_name,extract_peptide_sequence,PeptideDataset, initialize_wandb\
                            ,calculate_bleu,calculate_perplexity_v2,calculate_entropy,calculate_jaccard_similarity,evaluate_sequences,calculate_properties,calculate_reconstruction_loss

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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

    decoder.to(args.device)

    for epoch in range(args.decoder_epochs):
        decoder.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            embeddings, sequences = batch
            embeddings, sequences = embeddings.to(args.device), sequences.to(args.device)

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
                embeddings, sequences = embeddings.to(args.device), sequences.to(args.device)

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
                embeddings, sequences = embeddings.to(args.device), sequences.to(args.device)

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

        sequence_accuracy, char_accuracy, avg_perplexity, avg_entropy, js_3, js_6 = evaluate_decoder(all_orig_seqs, all_gen_seqs, model, tokenizer, args.device)
        #sequence_accuracy, char_accuracy, avg_entropy, js_3, js_6 = evaluate_decoder(all_orig_seqs, all_gen_seqs, model, tokenizer, args.device)
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



def test_decoder(decoder, test_loader, model, tokenizer, args):
    decoder.eval()
    all_orig_seqs = []
    all_gen_seqs = []
    with torch.no_grad():
        for batch in test_loader:
            embeddings, sequences = batch
            embeddings, sequences = embeddings.to(args.device), sequences.to(args.device)
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
    # print(f"test - all_orig_seqs: {all_orig_seqs}")
    # print(f"test - all_gen_seqs: {all_gen_seqs}")

    sequence_accuracy, char_accuracy, avg_perplexity, avg_entropy, js_3, js_6 = evaluate_decoder(all_orig_seqs, all_gen_seqs, model, tokenizer, args.device)
    #sequence_accuracy, char_accuracy, avg_entropy, js_3, js_6 = evaluate_decoder(all_orig_seqs, all_gen_seqs, model, tokenizer, args.device)
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
   



def main(args):
    set_seed(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")


    #Load finetuned ESM model and tokenizer 
    current_dir = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(current_dir, "..", "Diffusion", "fine_tuned_esm2_acp", "epoch_5"))
    fine_tuned_model_path = model_path

    esm_model = EsmForMaskedLM.from_pretrained(fine_tuned_model_path, output_hidden_states=True)
    esm_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

    esm_model = esm_model.to(device)

    ######################################################
    # PHASE-SPECIFIC DATA PROCESSING (TRANSFER LEARNING) #
    ######################################################

    if args.phase == "pretrain":
        logger.info("Phase: Pretraining on Biolip data")

        
        # Step 1: Load the biolip.pt file and extract the set of important PDB IDs.
        biolip_data = torch.load("/home2/s230112/BIB/GNN/biolip.pt")
        pt_pdb_ids = set(entry["structure_ids"]["pdb_id"].strip().lower() for entry in biolip_data)
        logger.info(f"Extracted {len(pt_pdb_ids)} PDB IDs from the PT file.")
        
        # Step 2: Load the full BioLiP metadata from the TXT file.
        BIOLIP_META_HEADER = [
            "pdb_id", "receptor_chain", "resolution", "binding_site", "ligand_ccd_id", "ligand_chain",
            "ligand_serial_num", "binding_site_pdb", "binding_site_reorder", "catalyst_site_pdb",
            "catalyst_site_reorder", "enzyme_class_id", "go_term_id", "binding_affinity_literature",
            "binding_affinity_binding_moad", "binding_affinity_pdbind_cn", "binding_affinity_binding_db",
            "uniprot_db", "pubmed_id", "ligand_res_num", "receptor_seq"
        ]
        BIOLIP_META_FILE = "/home2/s230112/BIB/GNN/BioLiP.txt"
        complexes = pd.read_csv(BIOLIP_META_FILE, sep="\t", names=BIOLIP_META_HEADER)
        complexes.drop_duplicates(subset="pdb_id", inplace=True)
        complexes.reset_index(drop=True, inplace=True)
        
        # Normalize the pdb_id column to lower case for matching.
        complexes["pdb_id"] = complexes["pdb_id"].str.lower()

        # Filter for only peptide ligands      
        complexes = complexes[complexes["ligand_ccd_id"] == "peptide"]

        # Remove any missing ligand chain entries
        complexes = complexes.dropna(subset=["ligand_chain"])
                
        # Filter complexes with resolution less than 5.
        complexes = complexes.loc[complexes.resolution < 5]
        
        # Filter complexes so that only those with pdb_id present in the PT file are kept.
        complexes = complexes[complexes["pdb_id"].isin(pt_pdb_ids)]
        logger.info(f"{len(complexes)} complexes remain after filtering by peptide ligands, PT PDB IDs and resolution.")
        
        BIOLIP_CSV_FILE = "biolip_pepseq_pdbid.csv"
        if os.path.exists(BIOLIP_CSV_FILE):
            # Load the existing CSV file
            print(f"Loading existing CSV file: {BIOLIP_CSV_FILE}")
            peptide_df = pd.read_csv(BIOLIP_CSV_FILE)

        else:
            # Step 3: Extract peptide sequences 
            print(BIOLIP_CSV_FILE," not found. Creating CSV file.")
            peptide_list = []
            for idx, row in complexes.iterrows():
                pdb_id = row['pdb_id'].strip()  # already lower case
                receptor_chain = row['receptor_chain']
                ligand_chain = row['ligand_chain']
                seq = extract_peptide_sequence(pdb_id, receptor_chain, ligand_chain)
                if seq is None:
                    continue
                peptide_list.append({"pdb_id": pdb_id, "Sequence": seq})
            
            peptide_df = pd.DataFrame(peptide_list)
            peptide_df.to_csv('biolip_pepseq_pdbid_2.csv', encoding='utf-8', index=False)
            logger.info(f"Extracted {len(peptide_df)} Biolip peptide sequences.")
        
        # Step 4: Load Biolip embeddings (precomputed) from file.
        biolip_embeddings = torch.load("/home2/s230112/BIB/GNN/resulting_embeddings_final_pocket_biolip.pt")
        peptide_df['Normalized_Name'] = peptide_df['pdb_id'].apply(lambda x: x.lower())
        normalized_dict = {k.lower(): v for k, v in biolip_embeddings.items()}
        peptide_df['Embedding'] = peptide_df['Normalized_Name'].map(lambda x: normalized_dict[x].cpu() if x in normalized_dict else None)
        valid_count = peptide_df['Embedding'].notnull().sum()
        logger.info(f"Number of peptides with valid embeddings: {valid_count}")
        peptide_df = peptide_df[peptide_df['Embedding'].notnull()].reset_index(drop=True)
        if len(peptide_df) > 0:
            feature_values = torch.stack(peptide_df['Embedding'].tolist())
            logger.info(f"Feature shape: {feature_values.shape}")
        else:
            logger.error("No valid embeddings found for Biolip!")
        
        
    else:
        # Otherwise (finetune phase), use original CSV and embeddings.
        logger.info("Phase: Finetuning on ACP–BCL-xL data")
        # peptide_df = pd.read_csv('ori_peptide_data.csv')
        peptide_df = pd.read_csv('/home2/s230112/BIB/GNN/input_features.csv')
        embeddings_dict = torch.load("/home2/s230112/BIB/GNN/resulting_embeddings_final_pocket_acp.pt")

        # Normalize names in CSV and embeddings dictionary.
        # peptide_df['Normalized_Name'] = peptide_df['Name'].apply(normalize_name)
        peptide_df['Normalized_Name'] = peptide_df['pdb_id'].apply(normalize_name)
        normalized_dict = {normalize_name(k): v for k, v in embeddings_dict.items()}
        
        peptide_df['Embedding'] = peptide_df['Normalized_Name'].map(lambda x: normalized_dict[x].cpu() if x in normalized_dict else None)
        valid_count = peptide_df['Embedding'].notnull().sum()
        logger.info(f"Number of peptides with valid embeddings: {valid_count}")
        peptide_df = peptide_df[peptide_df['Embedding'].notnull()].reset_index(drop=True)
        if len(peptide_df) > 0:
            feature_values = torch.stack(peptide_df['Embedding'].tolist())
            logger.info(f"Feature shape: {feature_values.shape}")
        else:
            logger.error("No valid embeddings found for ACP–BCL-xL data!")


    acp_filename ='ACP_processed_data.npz'
    biolip_filename = 'biolip_processed_data.npz'

    
    if os.path.exists(biolip_filename) and args.phase =='pretrain':
        data = np.load(biolip_filename)
        print("File found. Arrays loaded.")
        encoded_peptides = data['array1']
        tokenized_sequences = data['array2']
        raw_sequences = data['array3']

    elif os.path.exists(acp_filename) and args.phase =='finetune':
        data = np.load(acp_filename)
        print("File found. Arrays loaded.")
        encoded_peptides = data['array1']
        tokenized_sequences = data['array2']
        raw_sequences = data['array3']
    else:
        print("File not found. running shuffling manually")

        # Shuffle and preprocess the peptide data using the ESM model/tokenizer.
        shuffled_peptide_df = peptide_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        encoded_peptides, tokenized_sequences, raw_sequences = preprocess_peptide_data(shuffled_peptide_df, esm_model, esm_tokenizer, args)
        
        if args.phase == 'pretrain':
            np.savez(biolip_filename, array1=encoded_peptides, array2=tokenized_sequences, array3=raw_sequences)
        else:
            np.savez(acp_filename, array1=encoded_peptides, array2=tokenized_sequences, array3=raw_sequences)

    ##End of data processing

    total_size = len(encoded_peptides)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)
    ####################
    # Decoder Training #
    ####################
    # Initialize the two-stage decoder 
    decoder_s1 = Decoder(input_dim=1280, hidden_dim=256, output_dim=128, num_layers=1).to(device)
    decoder_s2 = Decoder(input_dim=128, hidden_dim=256, output_dim=len(esm_tokenizer), num_layers=1).to(device)
    decoder = Decoder_2_stage(stage1=decoder_s1, stage2=decoder_s2)


    decoder_train_embeddings = encoded_peptides[:train_size]
    decoder_train_sequences = tokenized_sequences[:train_size]
    decoder_valid_embeddings = encoded_peptides[train_size:train_size+valid_size]
    decoder_valid_sequences = tokenized_sequences[train_size:train_size+valid_size]
    decoder_test_embeddings = encoded_peptides[train_size+valid_size:]
    decoder_test_sequences = tokenized_sequences[train_size+valid_size:]
    
    decoder_train_dataset = PeptideDataset(decoder_train_embeddings, decoder_train_sequences)
    decoder_valid_dataset = PeptideDataset(decoder_valid_embeddings, decoder_valid_sequences)
    decoder_test_dataset  = PeptideDataset(decoder_test_embeddings, decoder_test_sequences)
    
    decoder_train_loader = DataLoader(decoder_train_dataset, batch_size=args.decoder_batch_size, shuffle=True, pin_memory=True)
    decoder_valid_loader = DataLoader(decoder_valid_dataset, batch_size=args.decoder_batch_size, shuffle=False, pin_memory=True)
    decoder_test_loader  = DataLoader(decoder_test_dataset, batch_size=args.decoder_batch_size, shuffle=False, pin_memory=True)
    


    if args.train_decoder.upper() == "Y":
        if args.phase == "pretrain":
            # Pretrain the decoder on BioLiP
            logger.info("Pretraining the decoder on Biolip dataset.")
            train_decoder(decoder, decoder_train_loader, decoder_valid_loader, esm_model, esm_tokenizer, args)

            # Save the pretrained decoder
            if args.save_decoder_model:
                save_path = os.path.join("Decoder/decoder_models", f"{args.save_decoder_model}_biolip.pt")
                torch.save(decoder.state_dict(), save_path)
                logger.info(f"Pretrained decoder saved as {save_path}")


        if args.phase == "finetune":
            # Load the pretrained decoder from BioLiP
            pretrained_model_path = os.path.join("Decoder/decoder_models", f"{args.save_decoder_model}_biolip.pt")
            if os.path.exists(pretrained_model_path):
                decoder.load_state_dict(torch.load(pretrained_model_path))
                logger.info(f"Pretrained Biolip decoder loaded from {pretrained_model_path} for fine-tuning.")
            else:
                logger.warning(f"Pretrained Biolip decoder not found. Training from scratch.")

            # Fine-tune on ACP–BCL-xL data
            logger.info("Fine-tuning the decoder on ACP–BCL-xL dataset.")
            train_decoder(decoder, decoder_train_loader, decoder_valid_loader, esm_model, esm_tokenizer, args)

            # Save the fine-tuned decoder
            if args.save_decoder_model:
                save_path = os.path.join("Decoder/decoder_models", f"{args.save_decoder_model}_acp.pt")
                torch.save(decoder.state_dict(), save_path)
                logger.info(f"Fine-tuned decoder saved as {save_path}")

        test_decoder(decoder, decoder_test_loader, esm_model, esm_tokenizer, args)

        
    else:
        print("Loading the decoder for evaluation only")
        # Load the decoder for evaluation only
        model_path = os.path.join("Decoder/decoder_models", f"{args.load_decoder_model}.pt")

        if os.path.exists(model_path):
            decoder.load_state_dict(torch.load(model_path))
            decoder.to(device)
            logger.info(f"Decoder model loaded from {model_path}")
            test_decoder(decoder, decoder_test_loader, esm_model, esm_tokenizer, args)
            
            
        else:
            logger.error(f"Decoder model {model_path} not found! Cannot proceed with testing.")
            exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a 1D denoising diffusion model with transfer learning.")
    # Common arguments
    parser.add_argument("--phase", type=str, default="pretrain", choices=["pretrain", "finetune"],
                        help="Select phase: pretrain on Biolip or finetune on ACP–BCL-xL")
    parser.add_argument("--peptide-data", type=str, default="cleaned_peptide_data.csv", help="(Used in finetune phase)")
    parser.add_argument("--feature-data", type=str, default="cleaned_joined_data.csv", help="Feature data CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=int, default=0, help="GPU device id")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--test-batch-size", type=int, default=4, help="Test batch size")

    # Decoder arguments
    parser.add_argument("--decoder-epochs", type=int, default=10, help="Decoder training epochs")
    parser.add_argument("--decoder-batch-size", type=int, default=32, help="Decoder batch size")
    parser.add_argument("--train-decoder", type=str, default="Y", help="Train the decoder? (Y/N)")
    parser.add_argument("--save-decoder-model", type=str, help="Filename to save the trained decoder model")
    parser.add_argument("--load-decoder-model", type=str, help="Filename to load a saved decoder model")
    
    args = parser.parse_args()
    # Optionally override some arguments:
    args.phase = "finetune"
    args.device = 0
    args.gradient_accumulate_every = 8
    args.mixed_precision = False

    ### DECODER MODEL ###
    args.decoder_batch_size = 8
    args.train_decoder = "Y" # Set "Y" to train the decoder, "N" to load a pretrained one
    #args.load_decoder_model = "np_acp" #only for evaluation
    args.save_decoder_model = "np"
    args.decoder_epochs = 1

    
    main(args)
    