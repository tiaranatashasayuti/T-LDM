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
from wae_misc.model_VAE import RNN_VAE
import wae_misc.losses as losses

from wae_data_processing.utils import  anneal


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Decoder.train_decoder import Decoder,Decoder_2_stage,test_decoder

from general_utils import preprocess_peptide_data,normalize_name,extract_peptide_sequence,PeptideDataset, initialize_wandb\
                            ,calculate_bleu,calculate_perplexity_v2,calculate_entropy,calculate_jaccard_similarity,evaluate_sequences,calculate_properties,calculate_reconstruction_loss\
                            ,BioDataset_modified


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


def decode_embeddings(embeddings, decoder, tokenizer, args):
    decoded_sequences = []
    decoder.eval()
    with torch.no_grad():
        embedding = torch.tensor(embeddings).to(args.device)
        decoded_output = decoder(embedding)
        decoded_tokens = torch.argmax(decoded_output, dim=-1).squeeze().tolist()

        if isinstance(decoded_tokens[0], list):
            for item in decoded_tokens:
                decoded_sequence = tokenizer.decode([token for token in item if token != tokenizer.pad_token_id], skip_special_tokens=True)
                decoded_sequences.append(decoded_sequence.replace(" ", ""))
        else:
            decoded_sequence = tokenizer.decode([token for token in decoded_tokens if token != tokenizer.pad_token_id], skip_special_tokens=True)
            decoded_sequences.append(decoded_sequence.replace(" ", ""))

    return decoded_sequences


def decode_data(decoder, data):
    data = data.squeeze(1).permute(0, 2, 1)
    data = decoder(data).permute(0, 2, 1).unsqueeze(1)
    return data

def train_ae(model_name, model, loss_fun, train_batches, optimizer, device, it, decoder):
    model.train()
    loss_all = 0.0
    recon_mean = 0.0
    kl_mean = 0.0
    mmd_mean = 0.0
    
    for batch in train_batches:
        data, context, raw_sequences = batch if len(batch) == 3 else (batch[0], batch[1], None)
        data = data.to(device)
        context = context.to(device)

        with torch.no_grad():
            data = decode_data(decoder, data)
            stage1_seq = data.squeeze(1).permute(0, 2, 1)
            #print("Stage 1 seq:",stage1_seq.shape)
        data = data.squeeze(1).permute(0, 2, 1)

        (z_mu, z_logvar), (z, c), dec_logits = model(data,context)
            
        recon_loss = losses.recon_dec(data, dec_logits)
        kl_loss = losses.kl_gaussianprior(z_mu, z_logvar)
        wae_mmdrf_loss = losses.wae_mmd_gaussianprior(z, method='rf', sigma=7)
        z_logvar_KL_penalty = losses.kl_gaussian_sharedmu(z_mu, z_logvar)
        


        if model_name == 'WAE':
            beta = 3
            loss = recon_loss + beta*wae_mmdrf_loss + 0.001*z_logvar_KL_penalty
            
        else:
            beta = 0.03
            loss = recon_loss + beta*kl_loss


        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        
        it += 1
        loss_all += loss.item()
        recon_mean += recon_loss.item()
        kl_mean += kl_loss.item()
        mmd_mean += wae_mmdrf_loss.item()
    
        
    
    return loss_all / len(train_batches), recon_mean / len(train_batches), kl_mean / len(train_batches), mmd_mean / len(train_batches), it, dec_logits,stage1_seq,raw_sequences

def valid_ae(model_name, model, loss_fun, valid_batches,  device, it, decoder):
    model.eval()
    loss_all = 0.0
    recon_mean = 0.0
    kl_mean = 0.0
    mmd_mean = 0.0
    
    for batch in valid_batches:
        data, context, raw_sequences = batch if len(batch) == 3 else (batch[0], batch[1], None)
        data = data.to(device)
        context = context.to(device)

        with torch.no_grad():
            data = decode_data(decoder, data)
            data = data.squeeze(1).permute(0, 2, 1)
            (z_mu, z_logvar), (z, c), dec_logits = model(data,context)
            
        recon_loss = losses.recon_dec(data, dec_logits)
        kl_loss = losses.kl_gaussianprior(z_mu, z_logvar)
        wae_mmdrf_loss = losses.wae_mmd_gaussianprior(z, method='rf', sigma=7)
        z_logvar_KL_penalty = losses.kl_gaussian_sharedmu(z_mu, z_logvar)
        


        if model_name == 'WAE' :
            beta = anneal(start_val=1.0, end_val=2.0, start_iter=0, end_iter=40000, it=it)
            loss = recon_loss + beta*wae_mmdrf_loss + 0.001*z_logvar_KL_penalty
        else:
            beta = 0.03
            loss = recon_loss + beta*kl_loss


   
        it += 1
        loss_all += loss.item()
        recon_mean += recon_loss.item()
        kl_mean += kl_loss.item()
        mmd_mean += wae_mmdrf_loss.item()
        
        
    return loss_all / len(valid_batches), recon_mean / len(valid_batches), kl_mean / len(valid_batches), mmd_mean / len(valid_batches), it, dec_logits,data,raw_sequences




def initialize_wandb(model_name,run_name, config, group=None):
    wandb.init(
        project=f"ACP_{model_name}_TEST",
        name=run_name,
        config=config,
        reinit=True,  # Ensures a new run is started
        group=group  # Optional: Group related runs together
    )
    wandb.define_metric(f"{model_name} Train Loss", step_metric="epoch")
    wandb.define_metric(f"{model_name} Validation Loss", step_metric="epoch")
    wandb.define_metric(f"{model_name} Train recon Loss", step_metric="epoch")
    wandb.define_metric(f"{model_name} Validation recon Loss", step_metric="epoch")
    wandb.define_metric(f"{model_name} Train mmd Loss", step_metric="epoch")
    wandb.define_metric(f"{model_name} Validation mmd Loss", step_metric="epoch")


def main(args):
    set_seed(args.seed)
    
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.set_device(args.device)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")



    #Load finetuned ESM model and tokenizer 
    fine_tuned_model_path = "/home2/s230112/BIB/Diffusion/fine_tuned_esm2_acp/epoch_5"
    esm_model = EsmForMaskedLM.from_pretrained(fine_tuned_model_path, output_hidden_states=True)
    esm_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)

    esm_model = esm_model.to(device)

    # If using the ESM2 as it is without finetuning
    #esm_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    #esm_model.to(device)
    
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
        
        # Step 4: Load Biolip embeddings (precomputed) .
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

    decoder_test_embeddings = encoded_peptides[train_size + valid_size:]
    decoder_test_sequences = tokenized_sequences[train_size + valid_size:]
    
    decoder_s1 = Decoder(input_dim=1280, hidden_dim=256, output_dim=128, num_layers=1).to(device)
    decoder_s2 = Decoder(input_dim=128, hidden_dim=256, output_dim=len(esm_tokenizer), num_layers=1).to(device)
    decoder = Decoder_2_stage(stage1=decoder_s1, stage2=decoder_s2)
    
    decoder_test_dataset = PeptideDataset(decoder_test_embeddings, decoder_test_sequences)
    decoder_test_loader = DataLoader(decoder_test_dataset, batch_size=args.decoder_batch_size, shuffle=False, pin_memory=True)


    #Load pretrained decoder from Diffusion directory
    decoder.load_state_dict(torch.load("/home2/s230112/BIB_FINAL/Decoder/decoder_models/" + args.load_decoder_model + ".pt"))
    decoder.to(device)
    print(f"Decoder model loaded from {args.load_decoder_model}.pt")
    
    #Running a quick test for the preloaded decoder
    test_decoder(decoder, decoder_test_loader, esm_model, esm_tokenizer, args)


    model_name = args.model_name
    lr = args.learning_rate
    epochs = args.epochs
    save_path = os.path.join(args.save_path,f'{model_name}')

   
    if not os.path.exists(save_path):

        os.makedirs(save_path)

    embedding = torch.randn((64,1280))

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    valid_dl = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    model_details = {
        'z_dim': 100,
        'c_dim': 2,
        'emb_dim': 128,
        'Encoder_args': {'h_dim': 80, 'biGRU': True, 'layers': 1, 'p_dropout': 0.0},
        'Decoder_args': {'p_word_dropout': 0.3, 'biGRU': False, 'layers': 1}
    }

    model = RNN_VAE(vocab_size=128, max_seq_len=64, context_dim = 256,device=device, **model_details).to(device)
    loss_fun = losses
    optimizer = optim.Adam(model.parameters(), lr=lr)
    

    print("Initialise wandDB")
    initialize_wandb(model_name,f"{model_name} Model Training -{args.save_ae_model}", args, group=model_name)

    print(f'Training base {model_name}...')
    decoder_for_ae = decoder.stage1
    
    ##########################################
        # Begin training for VAE/WAE #
    ##########################################
    
    if args.train_ae == "Y":
        start_epoch = args.start_epoch
        total_epochs = args.total_epochs
        pbar = tqdm(range(start_epoch, total_epochs + 1))
        #pbar = tqdm(range(1, epochs + 1))

        if args.phase != "finetune" and args.phase != "pretrain":
            raise Exception("Please choose finetune or pretrain")

        if args.phase == "pretrain":
            logger.info(f"{args.phase} selected. Model will load Biolip dataset")
            if args.load_ae_model is not None:
                load_path = os.path.join(save_path, f'{args.load_ae_model}.pt')
                if os.path.exists(load_path):
                    logger.info("Loading model checkpoint from: %s", load_path)
                    model.load_state_dict(torch.load(load_path))
                    model.to(device)
                    logger.info(f"{model_name} model loaded from {load_path}")
                else:
                    logger.info("Checkpoint %s not found. Training from scratch.", load_path)

            pass
            
        if args.phase == "finetune":
            print(f"{args.phase} selected. Model will load ACP Bcl-xL dataset")
            print("Loading model")
            model.load_state_dict(torch.load(os.path.join(save_path, f'{model_name}_models/{args.load_ae_model}.pt')))
            model.to(device)
            print(f"{model_name} model loaded from {os.path.join(save_path, f'{model_name}_models/{args.load_ae_model}.pt')}")
            
        

        for epoch in pbar:
            it=0 #variable used for monitoring iterations, deprecated
            loss, loss_recon, loss_kl, loss_mmd, it,dec_logits,ori_data,raw = train_ae(model_name, model, loss_fun, train_dl, optimizer, device, it, decoder_for_ae)
            vloss, vloss_recon, vloss_kl, vloss_mmd, vit,vdec_logits,vori_data,vraw = valid_ae(model_name, model, loss_fun, valid_dl, device, it, decoder_for_ae)

            wandb.log({f"{model_name} Train Loss": loss}, step=epoch)
            wandb.log({f"{model_name} Validaton Loss": vloss}, step=epoch)

            wandb.log({f"{model_name} Train recon Loss": loss_recon}, step=epoch)
            wandb.log({f"{model_name} Validaton recon Loss": vloss_recon}, step=epoch)

            wandb.log({f"{model_name} Train mmd Loss": loss_mmd}, step=epoch)
            wandb.log({f"{model_name} Validaton mmd Loss": vloss_mmd}, step=epoch)



            output_seq =decode_embeddings(dec_logits.detach().cpu().numpy(), decoder.stage2, esm_tokenizer, args)
            ori_seq =decode_embeddings(ori_data.detach().cpu().numpy(), decoder.stage2, esm_tokenizer, args)

            raw_sequences = raw

            for i in range(len(output_seq)):   
                min_len = len(raw_sequences[i])
                #min_len = len(ori_seq[i])
                output_seq[i] = output_seq[i][:min_len]
                ori_seq[i] = ori_seq[i][:min_len]
            
            pbar.set_description(f'Epoch {epoch}. train_loss_{model_name}: {loss:.4f}; valid_loss_{model_name}: {vloss:.4f}; ')



        torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}_models/{args.save_ae_model}.pt'))
        print(f"{model_name} model saved to {os.path.join(save_path, f'{model_name}_models/{args.save_ae_model}.pt')}")
        
    else:
        model.load_state_dict(torch.load(os.path.join(save_path, f'{model_name}_models/{args.load_ae_model}.pt')))
        model.to(device)
        print(f"{model_name} model loaded from {os.path.join(save_path, f'{model_name}_models/{args.load_ae_model}.pt')}")

    # Define the directory where you want to save the generated sequences
    
    save_directory = os.path.join(save_path,f'{model_name}_generated_constrained')

    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    print("Proceeding to sampling mode")
    # Update the path to save the generated sequences file
    generated_sequences_file = os.path.join(save_directory, args.gen_seqs_filename)
    if not os.path.exists(generated_sequences_file):
        
        # Generate 2000 peptides
        peptide = 2000
        test_dataloader = DataLoader(test_dataset, args.test_batch_size, shuffle=True, pin_memory=True,  num_workers=8)
        generated_sequences = []
        num_iterations = peptide // len(test_dataset) + 1

        print(f"Generating sequences with {num_iterations} iterations...")

        for _ in tqdm(range(num_iterations)):
            for i in test_dataloader:
                seq, features, raw_seq = i
                seq, features = seq.to(device), features.to(device)

                #get size of batch as the final batch size is different

                batch_sample = seq.shape[0]
               
                embedding = torch.randn((batch_sample,64,1280))
                with torch.no_grad():
                    embedding = embedding.to(device).unsqueeze(1).permute(0,1,3,2)
                    data = decode_data(decoder.stage1, embedding)
                empty_embedding = data.squeeze(1).permute(0,2,1)
        
                sampled_seq = model.sample(mbsize=batch_sample,embedding=empty_embedding, z=None, c=None, context=features).to(device)
                gen_seqs = decode_embeddings(sampled_seq.detach().cpu().numpy(), decoder.stage2, esm_tokenizer, args)

                for original, generated in zip(raw_sequences, gen_seqs):
                    length = len(original)
                    generated = generated[:length]
                    generated_sequences.append(generated)
                    
            if len(generated_sequences) >= peptide:
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
    
     # Define the directory to save the generated sequences and properties
    
    save_directory_prop = os.path.join(save_path,f'{model_name}_results_constrained_properties')
    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(save_directory_prop):
        os.makedirs(save_directory_prop)

    # Update the path to save the generated sequences file
    generated_prop_file = os.path.join(save_directory_prop, args.gen_prop_filename)
    # Calculate properties for each generated peptide
    peptide_properties = {
        'Sequence': [],
        'Net_Charge_at_pH_7': [],
        'Isoelectric_Point': [],
        'GRAVY': [],
        'Molecular_Weight': []
    }

    for seq in generated_sequences:
        props = calculate_properties(seq)
        if props[0] is None:
            continue  # Skip sequences with non-standard amino acids
        peptide_properties['Sequence'].append(seq)
        peptide_properties['Net_Charge_at_pH_7'].append(props[0])
        peptide_properties['Isoelectric_Point'].append(props[1])
        peptide_properties['GRAVY'].append(props[2])
        peptide_properties['Molecular_Weight'].append(props[3])
    generated_peptide_df = pd.DataFrame(peptide_properties)
    generated_peptide_df.to_csv(generated_prop_file, index=False)
    print(f"Generated sequences and its physicochemical properties have been saved to {generated_prop_file}")



    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a 1D VAE or WAE model.")
    parser.add_argument("--peptide-data", type=str, default="cleaned_peptide_data.csv", help="Path to the peptide data CSV file.")
    parser.add_argument("--feature-data", type=str, default="cleaned_joined_data.csv", help="Path to the feature data CSV file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=int, default=0, help="GPU device number to use.")
    parser.add_argument("--model-dim", type=int, default=64, help="Dimension of the model.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--test-batch-size", type=int, default=8, help="Batch size for testing.")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Learning rate for training.")
    parser.add_argument("--save-model", type=str, help="Path to save the trained model.")
    parser.add_argument("--load-model", type=str, help="Path to load a pre-trained model.")
    
    parser.add_argument("--decoder-batch-size", type=int, default=32, help="Batch size for decoder training.")
    
    
    parser.add_argument("--load-decoder-model", type=str, help="Path to load a saved decoder model.")
    parser.add_argument("--train-ae", type=str, default="Y", help="Whether to train the WAE/VAE model.")
    parser.add_argument("--save-ae-model", type=str, default="wae_model", help="Path to save the trained WAE/VAE model.")
    parser.add_argument("--load-ae-model", type=str, help="Path to load a saved WAE/VAE model.")
    parser.add_argument("--gen_seqs_filename", type=str, help="Path to save the generated peptide sequences.")
    parser.add_argument("--gen_prop_filename", type=str, help="Path to save the generated peptide sequences.")
    
    parser.add_argument("--model-name", type=str, default="VAE", help="Name of the model to use.")
    
    parser.add_argument("--save-path", type=str, default="./VAE_models/", help="Directory to save the trained model.")
    
    parser.add_argument("--start-epoch", type=int, default=1, help="Number of start epochs to train the model.")
    parser.add_argument("--total-epochs", type=int, default=100, help="Number of total epochs to train the model.")
    args = parser.parse_args()

    args.device = 2
    args.gradient_accumulate_every = 8
    args.mixed_precision = True

    args.decoder_batch_size = 8
    
    
    args.load_decoder_model = "np_acp"
    

    args.train_ae = "Y"
    args.model_name = "VAE" #VAE or WAE
    args.start_epoch = 1
    args.total_epochs = 15
    #args.epochs = 50
    args.save_path = "/home2/s230112/BIB_FINAL/Ablation"
    args.save_ae_model = "vae_15_acp"
    args.load_ae_model = "vae_25_biolip"

    args.phase = 'finetune'

    args.gen_seqs_filename = "generated_dataset_vae_journal_15_acp.csv"
    args.gen_prop_filename = "generated_properties_vae_np_15_acp.csv"

    
    main(args)
