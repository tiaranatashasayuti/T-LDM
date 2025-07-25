import argparse
import os
import torch
import wandb
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmForMaskedLM, AdamW, get_linear_schedule_with_warmup

class BioMLMDataset(Dataset):
    """
    Reads a CSV file with a 'Sequence' column,
    tokenizes each sequence, and performs masked language modeling (MLM).
    """
    def __init__(self, csv_file, tokenizer, max_length=64, mask_prob=0.15):
        self.df = pd.read_csv(csv_file)
        self.sequences = self.df["Sequence"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Create MLM labels (randomly mask some tokens)
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)

        # Do not mask special tokens ([CLS], [SEP], [PAD])
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix = torch.where(
            torch.tensor(special_tokens_mask, dtype=torch.bool),
            torch.tensor(0.0),
            probability_matrix
        )

        # Randomly select tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only masked tokens contribute to loss

        # Replace masked input tokens with tokenizer.mask_token_id
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train_esm_mlm(model, tokenizer, train_csv, device, 
                  output_dir, epochs=5, batch_size=8, 
                  lr=2e-5, max_length=64, mask_prob=0.15, 
                  run_name="ESM2_FT"):
    # Initialize wandb run
    wandb.init(project="ESM2_FineTuning_2Stage", name=run_name)
    wandb.config.update({
        "train_csv": train_csv,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "mask_prob": mask_prob,
        "run_name": run_name
    })

    # Prepare dataset and dataloader
    train_dataset = BioMLMDataset(
        csv_file=train_csv,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_prob=mask_prob
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"[{run_name}] Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        wandb.log({"Epoch Loss": avg_loss}, step=epoch)

        # Save model and tokenizer after every epoch 
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)
        print(f"[{run_name}] Model checkpoint saved to {epoch_output_dir}")

    # Save final checkpoint to the output directory root
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{run_name}] Final model saved to {output_dir}")
    
    wandb.finish()
    return model  # Return the updated model

def freeze_all_except_last_layer(model):
    # Freeze all parameters first.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last transformer block.
    for param in model.esm.encoder.layer[-1].parameters():
        param.requires_grad = True

    # Unfreeze the LM head as well.
    for param in model.lm_head.parameters():
        param.requires_grad = True

    print("Unfroze model.esm.encoder.layer[-1] and model.lm_head")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D",
                        help="Pretrained ESM-2 model name from HuggingFace or path to a local ESM checkpoint.")
    parser.add_argument("--biolip_csv", type=str, default="biolip_pepseq_pdbid.csv",
                        help="Path to the BioLiP CSV with 'Sequence' column.")
    parser.add_argument("--acp_csv", type=str, default="ori_peptide_data.csv",
                        help="Path to the ACP CSV with 'Sequence' column.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'.")
    parser.add_argument("--biolip_output", type=str, default="fine_tuned_esm2_biolip",
                        help="Output directory for BioLiP-tuned ESM-2.")
    parser.add_argument("--acp_output", type=str, default="fine_tuned_esm2_acp",
                        help="Output directory for ACP-tuned ESM-2.")
    parser.add_argument("--biolip_epochs", type=int, default=5,
                        help="Number of training epochs for the BioLiP stage.")
    parser.add_argument("--acp_epochs", type=int, default=5,
                        help="Number of training epochs for the ACP stage.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for both stages.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for both stages.")
    parser.add_argument("--max_length", type=int, default=64,
                        help="Max sequence length.")
    parser.add_argument("--mask_prob", type=float, default=0.15,
                        help="Probability of masking each token.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ----------------------
    # Stage 1: Fine-tune on BioLiP
    # ----------------------
    print(f"=== Stage 1: Fine-tuning ESM-2 on BioLiP ({args.biolip_csv}) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EsmForMaskedLM.from_pretrained(args.model_name)

    model.to(device)

    # Freeze all layers except the last transformer block and LM head
    freeze_all_except_last_layer(model)

    model = train_esm_mlm(
        model=model,
        tokenizer=tokenizer,
        train_csv=args.biolip_csv,
        device=device,
        output_dir=args.biolip_output,
        epochs=args.biolip_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        mask_prob=args.mask_prob,
        run_name="ESM2_BioLiP_FT"
    )

    # ----------------------
    # Stage 2: Fine-tune on ACP
    # ----------------------
    print(f"=== Stage 2: Fine-tuning from BioLiP checkpoint on ACP ({args.acp_csv}) ===")
    # Reload the final checkpoint from Stage 1.
    tokenizer = AutoTokenizer.from_pretrained(args.biolip_output)
    model = EsmForMaskedLM.from_pretrained(args.biolip_output)
    model.to(device)

    # Freeze all layers except the last transformer block and LM head again
    freeze_all_except_last_layer(model)

    model = train_esm_mlm(
        model=model,
        tokenizer=tokenizer,
        train_csv=args.acp_csv,
        device=device,
        output_dir=args.acp_output,
        epochs=args.acp_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        mask_prob=args.mask_prob,
        run_name="ESM2_ACP_FT"
    )

    print("=== Two-stage fine-tuning complete! ===")
    print(f"Final ACP-tuned model is saved in: {args.acp_output}")

if __name__ == "__main__":
    main()
