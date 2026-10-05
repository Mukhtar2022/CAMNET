import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from statsmodels.stats.multitest import multipletests

BASE_DIR = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"

models_to_compare = [
    ("Noisy Input", None),
    ("NLM", "results_nlm.csv"),
    ("BM3D", "results_bm3d.csv"),
    ("DnCNN", "results_dncnn.csv"),
    ("RED-CNN", "results_red_cnn_final_journal_run.csv"),
    ("Improved RED-CNN-CSA", "results_improved_red_cnn_csa.csv"),
    ("EDCNN", "results_edcnn.csv"),
    ("CT-Former", "results_ctformer.csv"),
    ("U-Former", "results_uformer.csv"),
    ("LEDA", "results_leda.csv"),
]

proposed_file = os.path.join(BASE_DIR, "results_proposed_model_epo50.csv")
proposed_df = pd.read_csv(proposed_file)

def compute_noisy_metrics():
    test_clean_dir = os.path.join(BASE_DIR, "Testing")
    test_noisy_dir = os.path.join(BASE_DIR, "Testing_Noisy")
    psnrs, ssims, rmses = [], [], []
    files = sorted([f for f in os.listdir(test_clean_dir) if f.endswith('.png')])
    for f in files:
        clean = cv2.imread(os.path.join(test_clean_dir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        noisy = cv2.imread(os.path.join(test_noisy_dir, f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        p = psnr_fn(clean, noisy, data_range=1.0)
        s = ssim_fn(clean, noisy, data_range=1.0)
        r = float(np.sqrt(np.mean((clean - noisy)**2)))
        psnrs.append(p); ssims.append(s); rmses.append(r)
    return pd.DataFrame({'psnr': psnrs, 'ssim': ssims, 'rmse': rmses})

print("="*80)
print("WILCOXON SIGNED-RANK TEST (vs Proposed Model)")
print("="*80)
print(f"{'Model':<15} | {'PSNR p-value':<15} | {'SSIM p-value':<15} | {'RMSE p-value':<15} | {'PSNR adj':<12} | {'SSIM adj':<12} | {'RMSE adj':<12}")
print("-" * 80)

results = []
raw_psnr = []
raw_ssim = []
raw_rmse = []

for model_name, csv_file in models_to_compare:
    if csv_file is None:
        df = compute_noisy_metrics()
    else:
        path = os.path.join(BASE_DIR, csv_file)
        if not os.path.exists(path):
            print(f"{model_name:<15} | {'FILE NOT FOUND':<15} | {'-':<15} | {'-':<15}")
            continue
        df = pd.read_csv(path)
    
    # Wilcoxon test
    _, p_psnr = wilcoxon(proposed_df['psnr'], df['psnr'], alternative='greater')
    _, p_ssim = wilcoxon(proposed_df['ssim'], df['ssim'], alternative='greater')
    _, p_rmse = wilcoxon(proposed_df['rmse'], df['rmse'], alternative='less')
    
    raw_psnr.append(p_psnr)
    raw_ssim.append(p_ssim)
    raw_rmse.append(p_rmse)
    
    results.append({
        "Model": model_name,
        "PSNR p-value": p_psnr,
        "SSIM p-value": p_ssim,
        "RMSE p-value": p_rmse
    })
    
    print(f"{model_name:<15} | {p_psnr:<15.4e} | {p_ssim:<15.4e} | {p_rmse:<15.4e}")

# Apply Holm-Bonferroni correction separately for each metric
_, adj_psnr, _, _ = multipletests(raw_psnr, method='holm')
_, adj_ssim, _, _ = multipletests(raw_ssim, method='holm')
_, adj_rmse, _, _ = multipletests(raw_rmse, method='holm')

print("="*80)
print("Corrected p-values (Holm-Bonferroni):")
print(f"{'Model':<15} | {'PSNR adj':<12} | {'SSIM adj':<12} | {'RMSE adj':<12}")
print("-" * 80)
for i, res in enumerate(results):
    print(f"{res['Model']:<15} | {adj_psnr[i]:<12.4e} | {adj_ssim[i]:<12.4e} | {adj_rmse[i]:<12.4e}")

print("="*80)

md_path = os.path.join(BASE_DIR, "Wilcoxon_Holm_Bonferroni.md")
with open(md_path, 'w') as f:
    f.write("# Wilcoxon Signed-Rank Test with Holm-Bonferroni Correction\n\n")
    f.write("| Model | PSNR p-value | SSIM p-value | RMSE p-value | PSNR adj | SSIM adj | RMSE adj |\n")
    f.write("|-------|--------------|--------------|--------------|----------|----------|----------|\n")
    for i, res in enumerate(results):
        f.write(f"| {res['Model']} | {res['PSNR p-value']:.4e} | {res['SSIM p-value']:.4e} | {res['RMSE p-value']:.4e} | {adj_psnr[i]:.4e} | {adj_ssim[i]:.4e} | {adj_rmse[i]:.4e} |\n")

print(f"\nSaved Markdown table to: {md_path}")
