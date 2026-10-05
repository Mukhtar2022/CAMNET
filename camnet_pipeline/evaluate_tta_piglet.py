import os
import glob
import csv
import argparse

import numpy as np
import torch
import pydicom
import cv2
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

BASE_DIR = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"
PIGLET_DIR = os.path.join(BASE_DIR, "piglet-dataset")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = np.clip(img, -160, 240)
    img = (img + 160) / (240 + 160)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    return img


def get_piglet_test_pairs():
    gt_dir = os.path.join(PIGLET_DIR, "Piglet_Testing_85")
    noisy_dir = os.path.join(PIGLET_DIR, "Piglet_Testing_Noisy_85")
    gt_files = sorted(os.listdir(gt_dir))
    noisy_files = sorted(os.listdir(noisy_dir))
    if gt_files != noisy_files:
        raise RuntimeError("Mismatched Piglet test filenames between GT and Noisy dirs")
    return [(os.path.join(noisy_dir, f), os.path.join(gt_dir, f)) for f in gt_files]


def build_tta_transforms():
    """8-way dihedral TTA: {no-flip, hflip} x {rot0, rot90, rot180, rot270}."""
    transforms = []
    for flip in [False, True]:
        for k in [0, 1, 2, 3]:
            def fwd(x, flip=flip, k=k):
                if flip:
                    x = torch.flip(x, dims=[-1])
                x = torch.rot90(x, k, dims=[-2, -1])
                return x

            def inv(x, flip=flip, k=k):
                x = torch.rot90(x, -k, dims=[-2, -1])
                if flip:
                    x = torch.flip(x, dims=[-1])
                return x

            transforms.append((fwd, inv))
    return transforms


def load_model(ckpt_path):
    from models.proposed_red_cnn import Proposed_RED_CNN
    model = Proposed_RED_CNN()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def run(args):
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "proposed_model_piglet_best.pth")
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(ckpt_path)

    pairs = get_piglet_test_pairs()
    if args.limit:
        pairs = pairs[: args.limit]

    transforms = build_tta_transforms()
    print(f"Running {len(transforms)}-way TTA on {len(pairs)} Piglet held-out test slices...")

    records_tta, records_base = [], []
    with torch.no_grad():
        for noisy_path, gt_path in tqdm(pairs):
            fname = os.path.basename(noisy_path)
            noisy_np = read_dicom(noisy_path)
            gt_np = read_dicom(gt_path)
            noisy = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

            # baseline (no TTA), for direct side-by-side comparison
            base_out = model(noisy).clamp(0, 1)
            base_np = base_out[0, 0].cpu().numpy()

            # 8-way TTA
            outs = []
            for fwd, inv in transforms:
                aug_in = fwd(noisy)
                aug_out = model(aug_in).clamp(0, 1)
                outs.append(inv(aug_out))
            tta_out = torch.stack(outs).mean(dim=0).clamp(0, 1)
            tta_np = tta_out[0, 0].cpu().numpy()

            for rec_list, out_np in [(records_base, base_np), (records_tta, tta_np)]:
                p = float(psnr_fn(gt_np, out_np, data_range=1.0))
                s = float(ssim_fn(gt_np, out_np, data_range=1.0))
                r = float(np.sqrt(np.mean((gt_np - out_np) ** 2)))
                rec_list.append({"filename": fname, "psnr": p, "ssim": s, "rmse": r})

    def summarize(records, label, csv_name):
        psnrs = [r["psnr"] for r in records]
        ssims = [r["ssim"] for r in records]
        rmses = [r["rmse"] for r in records]
        m_p, s_p = np.mean(psnrs), np.std(psnrs)
        m_s, s_s = np.mean(ssims), np.std(ssims)
        m_r, s_r = np.mean(rmses), np.std(rmses)
        print(f"\n{'='*65}")
        print(f"{label}")
        print(f"Global PSNR: {m_p:.4f} +/- {s_p:.4f} dB")
        print(f"Global SSIM: {m_s:.4f} +/- {s_s:.4f}")
        print(f"Global RMSE: {m_r:.4f} +/- {s_r:.4f}")
        print(f"{'='*65}")
        csv_path = os.path.join(BASE_DIR, csv_name)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "rmse"])
            w.writeheader()
            w.writerows(records)
        return m_p, s_p, m_s, s_s, m_r, s_r

    b = summarize(records_base, "proposed_model (Piglet, NO TTA, this run)", "results_proposed_model_piglet_notta_check.csv")
    t = summarize(records_tta, "proposed_model (Piglet, 8-WAY TTA)", "results_proposed_model_piglet_tta.csv")

    with open(os.path.join(BASE_DIR, "global_metrics_proposed_model_piglet_tta.txt"), "w") as f:
        f.write(f"Global PSNR: {t[0]:.4f} +/- {t[1]:.4f}\n")
        f.write(f"Global SSIM: {t[2]:.4f} +/- {t[3]:.4f}\n")
        f.write(f"Global RMSE: {t[4]:.4f} +/- {t[5]:.4f}\n")

    print("\n" + "=" * 65)
    print("DELTA (TTA - no-TTA, this run):")
    print(f"  PSNR: {t[0]-b[0]:+.4f} dB")
    print(f"  SSIM: {t[2]-b[2]:+.4f}")
    print(f"  RMSE: {t[4]-b[4]:+.4f}")
    print("=" * 65)
    print("\nRED-CNN (piglet, no TTA) for reference: PSNR 31.5381, SSIM 0.9395, RMSE 0.0267")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    run(parser.parse_args())
