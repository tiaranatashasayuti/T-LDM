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

from denoising_diffusion_pytorch_2d import Unet, GaussianDiffusion, Trainer, decode_embeddings_to_sequence
import difflib
import re

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from general_utils import preprocess_peptide_data,normalize_name,extract_peptide_sequence,PeptideDataset, initialize_wandb\
                            ,calculate_bleu,calculate_perplexity_v2,calculate_entropy,calculate_jaccard_similarity,evaluate_sequences,calculate_properties,calculate_reconstruction_loss\
                            ,BioDataset_modified

from Decoder.train_decoder import Decoder,Decoder_2_stage


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# --- Peptide extraction helper for Biolip ---

def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)

def decode_embeddings(embeddings, decoder, tokenizer, args):
    """
    Decode embeddings using the decoder and tokenizer.
    """
    decoded_sequences = []
    decoder.eval()
    with torch.no_grad():
        embedding = embeddings
        embedding = torch.tensor(embedding).to(args.device)
        decoded_output = decoder(embedding)
        decoded_tokens = torch.argmax(decoded_output, dim=-1).squeeze().tolist()
        for item in decoded_tokens:
            decoded_sequence = tokenizer.decode([token for token in item if token != tokenizer.pad_token_id], skip_special_tokens=True)
            decoded_sequence = decoded_sequence.replace(" ", "")  # Remove spaces from the decoded sequence
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



# Main Diffusion Model Training 


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


    # PHASE-SPECIFIC DATA PROCESSING (TRANSFER LEARNING) #
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

    ##########################################
    # Create datasets for diffusion training #
    ##########################################
    total_size = len(encoded_peptides)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    train_dataset = BioDataset_modified(encoded_peptides[:train_size], feature_values[:train_size], raw_sequences[:train_size])
    valid_dataset = BioDataset_modified(encoded_peptides[train_size:train_size+valid_size], feature_values[train_size:train_size+valid_size], raw_sequences[train_size:train_size+valid_size])
    test_dataset = BioDataset_modified(encoded_peptides[train_size+valid_size:], feature_values[train_size+valid_size:], raw_sequences[train_size+valid_size:])

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Valid dataset size: {len(valid_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    

    # Initialize the two-stage decoder 
    decoder_s1 = Decoder(input_dim=1280, hidden_dim=256, output_dim=128, num_layers=1).to(device)
    decoder_s2 = Decoder(input_dim=128, hidden_dim=256, output_dim=len(esm_tokenizer), num_layers=1).to(device)
    decoder = Decoder_2_stage(stage1=decoder_s1, stage2=decoder_s2)
    

    print("Loading the decoder for evaluation only")
    # Load the decoder for evaluation only
    model_path = os.path.join("Decoder/decoder_models", f"{args.load_decoder_model}.pt")

    if os.path.exists(model_path):
        decoder.load_state_dict(torch.load(model_path))
        decoder.to(device)
        logger.info(f"Decoder model loaded from {model_path}")
    else:
        raise Exception("Pretrained decoder not available! please load stage 1 decoder")


    
    #########################################
    # Diffusion Model Training
    #########################################
    # Use the first stage of decoder as the constraint
    decoder_constraint = decoder.stage1
    model_unet = Unet(
        dim=args.model_dim,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        context_dim=256  # constraint dimension
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model_unet,
        image_size=(128, 64),
        timesteps=args.noising_timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective='pred_noise'
    ).to(device)
    

    result_folder_directory = os.path.join(args.result_folder,'results_constrained_v2')
    if not os.path.exists(result_folder_directory):
        os.makedirs(result_folder_directory)

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
        device=device,
        decoder=decoder_constraint,
        results_folder=result_folder_directory
    )
    
    
    if args.train_steps > 0:
        if args.phase == "pretrain":
            initialize_wandb(f"Diffusion Model Training -{args.save_model}_biolip", args, group="Diffusion")
        
        #Uncomment this if you want continue PRETRAIN from an existing model
        # if args.load_model:
        #     trainer.load(args.load_model)
        
            trainer.train()
            if args.save_model:
                trainer.save(f"{args.save_model}_biolip")  # Save the pretrained model
                print(f"Saved pretrained diffusion model as {args.save_model}_biolip")
                exit()

        if args.phase == "finetune":

            #load directory
            
            load_directory_pt = result_folder_directory
            load_model_dir = os.path.join(load_directory_pt, f"model-{args.load_model}_biolip.pt")
            
            pretrained_diffusion_model_path = load_model_dir

            print(f"Loading model from path {pretrained_diffusion_model_path} path exist:{os.path.exists(pretrained_diffusion_model_path)}")
            if os.path.exists(pretrained_diffusion_model_path):
                trainer.load("",pretrained_diffusion_model_path)  # Load the pretrained model
                logger.info(f"Loaded pretrained diffusion model from {pretrained_diffusion_model_path} for fine-tuning.")
                initialize_wandb(f"Diffusion Model Training -{args.save_model}_acp", args, group="Diffusion")
                trainer.step = 0
                trainer.train()
                
                if args.save_model:
                    trainer.save(f"{args.save_model}_acp")  # Save the pretrained model
                    print(f"Saved pretrained diffusion model as {args.save_model}_acp")
            else:
                logger.warning("Pretrained diffusion model not found. Please Training from scratch.")
                raise Exception("Pretrained model not available. please train the model first")
    else:
        trainer.load(args.load_model)
    torch.cuda.empty_cache()
    
        
    
    save_directory = os.path.join(args.result_folder,'generated_constrained_v2')

    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Update the path to save the generated sequences file
    generated_sequences_file = os.path.join(save_directory, args.gen_seqs_filename)
    
    test_dataloader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, pin_memory=True)
    peptide = 5
    num_iterations = peptide // len(test_dataset) + 1

    if not os.path.exists(generated_sequences_file):
        train_sequences = []
        original_sequences = []
        generated_sequences = []
        gen_ori_sequences = []

        #num_iterations = 1

        # Early evaluation parameters
        initial_batch_size = 100
        quality_threshold = 0.8

        for _ in tqdm(range(num_iterations)):
            for i in test_dataloader:
                seq, features, raw_seq = i
                seq, features = seq.to(device), features.to(device)

                sampled_seq = diffusion.sample(seq.shape[0], context=features).to(device)
                sampled_seq = sampled_seq.squeeze(1)
                seq = seq.squeeze(1)
                sampled_seq = sampled_seq.permute(0, 2, 1)
                seq = seq.permute(0, 2, 1)

                gen_seqs = decode_embeddings(sampled_seq.cpu().numpy(), decoder.stage2, esm_tokenizer, args)
                raw_sequences = [seq for seq in raw_seq]

                for original, generated in zip(raw_sequences, gen_seqs):
                    length = len(original)
                    generated = generated[:length]
                    generated_sequences.append(generated)
                #print("generated currently:", generated_sequences)
                

                # Early evaluation after generating a small batch
                if len(generated_sequences) >= initial_batch_size:
                    print("initial_batch_size reached")
                    if not evaluate_sequences(generated_sequences[:initial_batch_size], quality_threshold):
                        print(f"Model {args.load_model} did not pass early evaluation. Skipping further generation.")
                        return  # Skip further generation for this model

            if len(generated_sequences) >= 5000:
                print("Sample reached, exiting")
                break

        # Save generated sequences to CSV
        generated_sequences_df = pd.DataFrame({'Generated_Sequence': generated_sequences})
        generated_sequences_df.to_csv(generated_sequences_file, index=False)
        print(f"Generated sequences have been saved to {generated_sequences_file}")

    else:
        # Load generated sequences from CSV
        generated_sequences_df = pd.read_csv(generated_sequences_file, header=None, names=['Generated_Seqs'])
        generated_sequences = generated_sequences_df['Generated_Seqs'].tolist()
        print(f"Generated sequences have been loaded from {generated_sequences_file}")

    # Define the directory where you want to save the generated sequences
    
    save_directory_prop = os.path.join(args.result_folder,'result_constrained_properties_v2')

    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(save_directory_prop):
        os.makedirs(save_directory_prop)

    # Update the path to save the generated sequences file
    generated_prop_file = os.path.join(save_directory_prop, args.gen_prop_filename)

    # Initialize test_raw_sequences before its first use
    test_raw_sequences = []
    train_raw_sequences = []
   
    #num_iterations = 1

    for _ in tqdm(range(num_iterations)):
        for i in test_dataloader:
            test_seq, test_features, test_raw_seq = i
            #print("printing raw seq")
            #print(test_raw_seq)

            test_raw_sequences=test_raw_sequences+test_raw_seq

    for i in train_dataset:
        train_seq, train_features, train_raw_seq = i
        train_raw_sequences.append(train_raw_seq)

    # Calculate properties for each generated peptide
    peptide_properties = {
        'Sequence': [],
        'Net_Charge_at_pH_7': [],
        'Isoelectric_Point': [],
        'GRAVY': [],
        'Molecular_Weight': [],
        'BLEU': [],
        'Perplexity': [],
        'Reconstruction_Loss': []
    }

    # Use train_raw_sequences to pair original and generated sequences
    #print(test_raw_sequences)
    #print(generated_sequences)

    #to remove the first index 'Generated_sequence'
    generated_sequences = generated_sequences[1:]
    for generated_seq in generated_sequences:
        closest_original = find_closest_match(generated_seq, test_raw_sequences)
    #for original, generated in zip(test_raw_sequences, generated_sequences):
        #if not generated:
        if not closest_original:
            continue

        props = calculate_properties(generated_seq)
        if props[0] is None:
            continue  # Skip sequences with non-standard amino acids
        peptide_properties['Sequence'].append(generated_seq)
        peptide_properties['Net_Charge_at_pH_7'].append(props[0])
        peptide_properties['Isoelectric_Point'].append(props[1])
        peptide_properties['GRAVY'].append(props[2])
        peptide_properties['Molecular_Weight'].append(props[3])

        # Calculate BLEU, Perplexity, and Reconstruction Loss
        bleu_score = calculate_bleu(closest_original, generated_seq)
        perplexity = calculate_perplexity_v2(generated_seq, esm_model, esm_tokenizer, device)
        reconstruction_loss = calculate_reconstruction_loss(closest_original, generated_seq, esm_tokenizer)

        peptide_properties['BLEU'].append(bleu_score)
        peptide_properties['Perplexity'].append(perplexity)
        peptide_properties['Reconstruction_Loss'].append(reconstruction_loss)

    generated_peptide_df = pd.DataFrame(peptide_properties)
    generated_peptide_df.to_csv(generated_prop_file, index=False)
    print(f"Generated sequences and their physicochemical properties have been saved to {generated_prop_file}")

    # Calculate properties for the train dataset
    train_properties = {
        'Sequence': [],
        'Net_Charge_at_pH_7': [],
        'Isoelectric_Point': [],
        'GRAVY': [],
        'Molecular_Weight': []
    }

    for seq in train_raw_sequences:
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
    parser.add_argument("--save-model", type=str, help="Filename to save the trained diffusion model")
    parser.add_argument("--load-model", type=str, help="Filename to load a diffusion model")
    parser.add_argument("--sampling-timesteps", type=int, default=200, help="Sampling timesteps")
    parser.add_argument("--noising-timesteps", type=int, default=200, help="Noising timesteps")
    parser.add_argument("--save-and-sample-every", type=int, default=1000, help="Frequency for saving and sampling")
    parser.add_argument("--result-folder", type=str, help="Result folder to store diffusion model and anaytics")
    # Decoder arguments
    #parser.add_argument("--decoder-epochs", type=int, default=10, help="Decoder training epochs")
    #parser.add_argument("--decoder-batch-size", type=int, default=32, help="Decoder batch size")
    #parser.add_argument("--train-decoder", type=str, default="Y", help="Train the decoder? (Y/N)")
    #parser.add_argument("--save-decoder-model", type=str, help="Filename to save the trained decoder model")
    parser.add_argument("--load-decoder-model", type=str, help="Filename to load a saved decoder model")
    
    args = parser.parse_args()
    # Optionally override some arguments:
    args.phase = "finetune"
    args.device = 0
    args.gradient_accumulate_every = 8
    args.mixed_precision = False

    ### DECODER MODEL ###
    #args.decoder_batch_size = 8
    #args.train_decoder = "N" # Set "Y" to train the decoder, "N" to load a pretrained one
    args.load_decoder_model = "np_acp"
    #args.save_decoder_model = "np"
    #args.decoder_epochs = 5

    #### DIFFUSION MODEL ###
    # To train and save the model, put a model name to args.save_model and timesteps to args.train_steps
    # To load the model and skip Training, args.load_model the model name and set args.train_steps to 0
    # To load the model and continue training, load the current model name, save it to a new name, and set timesteps
    args.train_steps = 1
    args.batch_size = 8
    args.valid_batch_size = 8
    args.test_batch_size = 8
    args.save_model = "1"
    args.result_folder= "/home2/s230112/BIB_FINAL/Diffusion/test"
    args.load_model = "1"

    args.gen_seqs_filename = "generated_dataset_journal_2.csv"
    args.gen_prop_filename = "generated_properties_journal_2.csv"

    args.sampling_timesteps = 5
    args.noising_timesteps = 1000
    args.save_and_sample_every = 10
    
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
