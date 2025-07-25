import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import numpy as np
from collections import Counter
import subprocess

# Function to calculate instability index
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def calculate_instability_index(seq):
    try:
        return ProteinAnalysis(seq).instability_index()
    except:
        return np.nan

def run_anticp2_on_sequences(df, label, temp_dir="anticp2"):
    os.makedirs(temp_dir, exist_ok=True)
    fasta_path = os.path.join(temp_dir, f"{label}.fasta")
    default_output = os.path.join(os.getcwd(), "outfile.csv")

    # Write FASTA file
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(df["Sequence"], 1):
            f.write(f">seq_{i}\n{seq}\n")

    # Run AntiCP2
    conda_path = "/data/s230112/anaconda3/etc/profile.d/conda.sh"
    cmd = f"source {conda_path} && conda activate anticp2 && anticp2 -i {fasta_path} && conda deactivate"
    result = subprocess.run(cmd, shell=True, executable="/bin/bash")

    # Check if it ran successfully and output exists
    if result.returncode == 0 and os.path.exists(default_output):
        df_scores = pd.read_csv(default_output, comment='#', header=None)
        df_scores.columns = ['Sequence_ID', 'Sequence', 'Prediction', 'ACP_Scores']
        merged_df = pd.merge(df, df_scores[['Sequence', 'ACP_Scores']], on="Sequence", how="left")

        os.remove(default_output)

        return merged_df
    else:
        print(f"Warning: AntiCP2 failed or output not found for label {label}")
        df["ACP_Scores"] = np.nan
        return df


# Load data
# Load data
train_df = pd.read_csv("/home2/s230112/BIB_FINAL/train_peptide_with_physicochemical_properties.csv")
ldm_550_df = pd.read_csv("/home2/s230112/BIB_FINAL/Diffusion/V2_TL_acp_nopocket_dec_b5_a5/results_properties_filtered_v2/generated_properties_journal_55.csv")
icassp_df = pd.read_csv("/home2/s230112/BIB_FINAL/Diffusion/final_generated_properties_ldm_ICASSP_198.csv")

# Add group labels
train_df["Group"] = "Training Peptides"
ldm_550_df["Group"] = "LDM (Ours)"
icassp_df["Group"] = "Sequence-Only Model"

# Apply instability index
for df in [train_df, ldm_550_df, icassp_df]:
    df["Instability_Index"] = df["Sequence"].apply(calculate_instability_index)

# Run anticp2 scoring and merge scores
train_df = run_anticp2_on_sequences(train_df, "train")
ldm_550_df = run_anticp2_on_sequences(ldm_550_df, "ldm550")
icassp_df = run_anticp2_on_sequences(icassp_df, "icassp")

output_dir = "/home2/s230112/BIB_FINAL/Diffusion/output"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train_peptides_with_acp_scores.csv"), index=False)
ldm_550_df.to_csv(os.path.join(output_dir, "ldm550_peptides_with_acp_scores.csv"), index=False)
icassp_df.to_csv(os.path.join(output_dir, "icassp_peptides_with_acp_scores.csv"), index=False)

