# from train_GAT_no_coords import set_seed, BindingPocketGNN, test_loop1, get_test_env
from train_GCN import set_seed, BindingPocketGNN, test_loop1, get_test_env
import torch
import pandas as pd

# 🔧 Get evaluation environment
device, loss_fn, finetune_test_loader = get_test_env()

seeds = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
# model_paths = [f"model_finetuned_seed_{seed}.pth" for seed in seeds]
model_paths = [f"model_GCN_finetuned_seed_{seed}.pth" for seed in seeds]

all_metrics = []

for seed, model_path in zip(seeds, model_paths):
    set_seed(seed)

    model = BindingPocketGNN(input_dim=15, hidden_dim=256, output_dim=1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    results = test_loop1(model, finetune_test_loader, loss_fn, device)
    results['seed'] = seed
    all_metrics.append(results)

df = pd.DataFrame(all_metrics)
print(df.describe())

for col in df.columns:
    if col != "seed":
        mean = df[col].mean()
        std = df[col].std()
        print(f"{col}: {mean:.4f} ± {std:.4f}")

from scipy.stats import ttest_rel, wilcoxon

# Save current model results 
df.to_csv("results_GCN.csv", index=False)

# To compare, you must also have GAT results stored in "results_GAT.csv"
# If this script is being run for GAT, change the save path accordingly

# Try loading both result sets
try:
    df_gat = pd.read_csv("results_GAT.csv")
    df_gcn = pd.read_csv("results_GCN.csv")

    print("\n📊 Statistical Significance Tests (GCN vs. GAT):")
    metrics_to_compare = ["accuracy","precision_0","recall_0","f1_0","precision_1","recall_1","f1_1","auc"]

    for metric in metrics_to_compare:
        x = df_gat[metric]
        y = df_gcn[metric]

        t_stat, t_p = ttest_rel(x, y)
        try:
            w_stat, w_p = wilcoxon(x, y)
        except ValueError:
            w_stat, w_p = (None, None)  # fallback for constant arrays

        print(f"\n🔹 {metric.upper()}:")
        print(f"  Paired t-test:      t = {t_stat:.4f}, p = {t_p:.4e}")
        if w_stat is not None:
            print(f"  Wilcoxon signed-rank: W = {w_stat:.4f}, p = {w_p:.4e}")
        else:
            print(f"  Wilcoxon test not valid (constant inputs).")

except FileNotFoundError:
    print("Either results_GAT.csv or results_GCN.csv not found. Run and save both models first.")
