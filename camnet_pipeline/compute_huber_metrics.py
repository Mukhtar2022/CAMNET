"""compute_huber_metrics.py
Compute average PSNR, SSIM, RMSE across all test slices for the Huber‑loss model.
No ROI visualisation – numeric CSV output.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from pdlm_ldctid_new import DEVICE, CKPT_DIR, Proposed_RED_CNN, compute_psnr, compute_ssim, compute_rmse

# -------------------------------------------------------------------
# Paths & constants
# -------------------------------------------------------------------
BASE_DIR = r"C:\\Users\\HP\\Documents\\COVID-19-EXPIREMENT"
TEST_CLEAN = os.path.join(BASE_DIR, "Testing")
TEST_NOISY = os.path.join(BASE_DIR, "Testing_Noisy")

# Huber‑loss checkpoint name (must exist in the checkpoints folder)
HUBER_CKPT = "proposed_model_loss_huber_best.pth"

def load_checkpoint(model, ckpt_path):
    state = torch.load(ckpt_path, map_location=DEVICE)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model

def run_model(noisy_img, ckpt_name):
    """Run the Huber‑loss model on a pre‑processed noisy image.
    Returns a NumPy array (H, W) with values in [0, 1].
    """
    noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
    model = Proposed_RED_CNN().to(DEVICE)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] checkpoint {ckpt_path} missing – returning noisy input.")
        return noisy_img
    model = load_checkpoint(model, ckpt_path)
    with torch.no_grad():
        out = model(noisy_tensor).clamp(0, 1)
    return out[0, 0].cpu().numpy()

def process_dataset():
    slice_files = sorted([f for f in os.listdir(TEST_CLEAN) if f.lower().endswith('.png')])
    psnr_vals, ssim_vals, rmse_vals = [], [], []
    for fname in slice_files:
        noisy_path = os.path.join(TEST_NOISY, fname)
        gt_path = os.path.join(TEST_CLEAN, fname)
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        # resize to match training dimensions
        noisy_img = cv2.resize(noisy_img, (256, 256))
        gt_img = cv2.resize(gt_img, (256, 256))
        pred = run_model(noisy_img, HUBER_CKPT)
        psnr_vals.append(compute_psnr(gt_img, pred))
        ssim_vals.append(compute_ssim(gt_img, pred))
        rmse_vals.append(compute_rmse(gt_img, pred))
    out_path = os.path.join(BASE_DIR, "huber_metrics.csv")
    df = pd.DataFrame({
        "psnr_mean": [np.mean(psnr_vals)],
        "ssim_mean": [np.mean(ssim_vals)],
        "rmse_mean": [np.mean(rmse_vals)]
    })
    df.to_csv(out_path, index=False)
    print(f"Huber metrics saved to {out_path}")

if __name__ == "__main__":
    process_dataset()
