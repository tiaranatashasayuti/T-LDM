import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, entropy
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
from sklearn.preprocessing import StandardScaler

# === Output folder ===
density_output_folder = 'density_curves'
os.makedirs(density_output_folder, exist_ok=True)

# === Helper Functions ===

def calculate_entropy(sequence):
    counts = Counter(sequence)
    probabilities = [count / len(sequence) for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities)

def calculate_instability_index(sequence):
    try:
        analysis = ProteinAnalysis(sequence)
        return analysis.instability_index()
    except:
        return np.nan

def normalize_data(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def compute_kl(train_data, model_data):
    train_data = train_data[~np.isnan(train_data)]
    model_data = model_data[~np.isnan(model_data)]

    if len(train_data) == 0 or len(model_data) == 0:
        return np.nan

    kde_train = gaussian_kde(train_data)
    kde_model = gaussian_kde(model_data)
    xs = np.linspace(min(min(train_data), min(model_data)), max(max(train_data), max(model_data)), 1000)
    return entropy(kde_train(xs), kde_model(xs))


def plot_all_models(train_df, model_dfs, model_labels, output_file,
                    legend_fontsize=17, box_fontsize=17):
    properties = [
        ('Molecular_Weight', 'Molecular Weight'),
        ('Isoelectric_Point', 'Isoelectric Point'),
        ('Net_Charge_at_pH_7', 'Net Charge at pH 7'),
        ('GRAVY', 'GRAVY'),
        ('Instability_Index', 'Instability Index'),
        ('ACP_Scores', 'ACP Scores')
    ]

    model_colors = ['red', 'green']  # For LDM and ICASSP
    plt.figure(figsize=(35, 12))

    for i, (prop, title) in enumerate(properties, start=1):
        ax = plt.subplot(2, 3, i)

        data_train = train_df[prop]
        data_models = [df[prop] for df in model_dfs]

        sns.kdeplot(data_train.dropna(), fill=True, color='tan', linewidth=0, label='Training Peptides')

        for data, label, color in zip(data_models, model_labels, model_colors):
            sns.kdeplot(data.dropna(), label=label, color=color, linewidth=2)


        # KL Divergence annotation
        annotation_text = f'KL Divergence\n'
        for data, label in zip(data_models, model_labels):
            kl = compute_kl(data_train, data)
            annotation_text += f'{label}: {kl:.4f}\n'

        ax.annotate(annotation_text.strip(),
                    xy=(0.640, 0.97), xycoords='axes fraction',
                    va='top', ha='left',
                    fontsize=box_fontsize,
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black',
                              linewidth=1.0, pad=0.3))

        # Legend styling
        ax.legend(loc='upper right', bbox_to_anchor=(1.005, 0.83),
                  prop={'size': legend_fontsize}, frameon=True,
                  fancybox=True, edgecolor='black', facecolor='white',
                  borderpad=0.3, handlelength=2.0, framealpha=1.0)

        plt.xlabel(title, fontsize=20)
        plt.ylabel('Probability Density', fontsize=20)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# === Main Script ===

def main():
    model_files = [
        ('output/ldm550_peptides_with_acp_scores.csv', 'LDM (Ours)'),
        ('output/icassp_peptides_with_acp_scores.csv', 'Sequence-Only Model')
    ]
    train_physchem_file = 'output/train_peptides_with_acp_scores.csv'

    # Load and normalize training data
    df_train = pd.read_csv(train_physchem_file)
    df_train['Instability_Index'] = df_train['Sequence'].apply(calculate_instability_index)
    df_train['Entropy'] = df_train['Sequence'].apply(calculate_entropy)

    df_train = normalize_data(df_train, [
        'Molecular_Weight', 'Isoelectric_Point', 'Net_Charge_at_pH_7',
        'GRAVY', 'Instability_Index', 'ACP_Scores'
    ])

    model_dfs = []
    model_labels = []

    for path, label in model_files:
        df = pd.read_csv(path)
        df['Instability_Index'] = df['Sequence'].apply(calculate_instability_index)
        df['Entropy'] = df['Sequence'].apply(calculate_entropy)

        df = normalize_data(df, [
            'Molecular_Weight', 'Isoelectric_Point', 'Net_Charge_at_pH_7',
            'GRAVY', 'Instability_Index', 'ACP_Scores'
        ])

        model_dfs.append(df)
        model_labels.append(label)

    output_file = os.path.join(density_output_folder, 'combined_density_plot_all_models_final.png')
    plot_all_models(df_train, model_dfs, model_labels, output_file)

if __name__ == "__main__":
    main()
