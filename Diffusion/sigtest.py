import pandas as pd
from scipy.stats import ttest_ind, f_oneway

# Load CSVs
ldm_np = pd.read_csv("/home2/s230112/BIB_FINAL/Diffusion/V2_TL_acp_nopocket_dec_b5_a5/results_properties_filtered_v2/generated_properties_journal_55.csv")
ldm_p = pd.read_csv("/home2/s230112/BIB_FINAL/Diffusion/V2_TL_acp_pocket_dec_b5_a5/results_properties_filtered_v2/generated_properties_journal_55.csv")
vae = pd.read_csv("/home2/s230112/BIB_FINAL/Ablation/VAE/VAE_results_constrained_properties/generated_properties_vae_np_15.csv")
wae = pd.read_csv("/home2/s230112/BIB_FINAL/Ablation/WAE/WAE_results_constrained_properties/generated_properties_wae_np_15.csv")

# T-test between LDM (non-pocket) and LDM (pocket)
ttest_bleu = ttest_ind(ldm_np['BLEU'], ldm_p['BLEU'], equal_var=False)
ttest_perplexity = ttest_ind(ldm_np['Perplexity'], ldm_p['Perplexity'], equal_var=False)

# ANOVA between LDM (non-pocket), VAE, WAE
anova_bleu = f_oneway(ldm_np['BLEU'], vae['BLEU'], wae['BLEU'])
anova_perplexity = f_oneway(ldm_np['Perplexity'], vae['Perplexity'], wae['Perplexity'])

# Means and standard deviations
metrics = ['BLEU', 'Perplexity']
models = {
    'LDM_NonPocket': ldm_np,
    'LDM_Pocket': ldm_p,
    'VAE': vae,
    'WAE': wae
}

for metric in metrics:
    print(f"\n=== {metric} ===")
    for name, df in models.items():
        mean = df[metric].mean()
        std = df[metric].std()
        print(f"{name}: {mean:.3f} ± {std:.3f}")

# Print p-values
print("\nT-test LDM (non-pocket vs pocket):")
print(f"BLEU p = {ttest_bleu.pvalue:.5e}")
print(f"Perplexity p = {ttest_perplexity.pvalue:.5e}")

print("\nANOVA LDM vs VAE vs WAE:")
print(f"BLEU p = {anova_bleu.pvalue:.5e}")
print(f"Perplexity p = {anova_perplexity.pvalue:.5e}")
