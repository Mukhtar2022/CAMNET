import os
import torch
import numpy as np
import cv2
from dataset import CTDenoisingDataset
from torch.utils.data import DataLoader
from pdlm_ldctid_new import compute_psnr, compute_ssim, compute_rmse, set_seed, Proposed_RED_CNN, DEVICE

# Paths
PROJECT_ROOT = r"C:\\Users\\HP\\Documents\\COVID-19-EXPIREMENT"
PIGLET_DIR = os.path.join(PROJECT_ROOT, "piglet-dataset")
NOISY_DIR = os.path.join(PIGLET_DIR, "FD_850")
GT_DIR = os.path.join(PIGLET_DIR, "LD25_850")
CKPT_PATH = os.path.join(PROJECT_ROOT, "proposed_model_epo50_best.pth")

def load_model():
    model = Proposed_RED_CNN().to(DEVICE).eval()
    if os.path.exists(CKPT_PATH):
        try:
            ckpt_state = torch.load(CKPT_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt_state, strict=False)
            print(f"[Info] Loaded checkpoint: {CKPT_PATH}")
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}")
    else:
        print("[Info] Checkpoint not found, using random weights.")
    return model

def build_dataloader():
    # Build list of matching pairs (assumes same filenames in both dirs)
    pairs = []
    for fname in sorted(os.listdir(NOISY_DIR)):
        if not fname.lower().endswith('.png'):
            continue
        noisy_path = os.path.join(NOISY_DIR, fname)
        gt_path = os.path.join(GT_DIR, fname)
        if os.path.exists(gt_path):
            pairs.append((noisy_path, gt_path))
    if not pairs:
        raise RuntimeError("No matching image pairs found in piglet dataset.")
    dataset = CTDenoisingDataset(pairs_override=pairs, patch_size=0, augment=False, image_size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return loader

def evaluate():
    set_seed(42)
    loader = build_dataloader()
    model = load_model()
    psnrs, ssims, rmses = [], [], []
    with torch.no_grad():
        for noisy, gt in loader:
            noisy = noisy.to(DEVICE)
            gt = gt.to(DEVICE)
            out = model(noisy).clamp(0, 1)
            gn = gt[0,0].cpu().numpy().astype(np.float32)
            pn = out[0,0].cpu().numpy().astype(np.float32)
            psnrs.append(compute_psnr(gn, pn))
            ssims.append(compute_ssim(gn, pn))
            rmses.append(compute_rmse(gn, pn))
    # Aggregate results
    metrics_path = os.path.join(PIGLET_DIR, "global_metrics_proposed_model_piglet.txt")
    with open(metrics_path, "w") as f:
        f.write(f"PSNR  mean: {np.mean(psnrs):.4f} ± {np.std(psnrs):.4f}\n")
        f.write(f"SSIM  mean: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}\n")
        f.write(f"RMSE  mean: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}\n")
    print(f"[Info] Metrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate()
