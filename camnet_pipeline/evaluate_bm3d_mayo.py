import os
import glob
import csv
import time
import argparse

import numpy as np
import pydicom
import cv2
from tqdm import tqdm
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import bm3d

BASE_DIR = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"
MAYO_DIR = os.path.join(BASE_DIR, "Mayo-folder")


def read_dicom(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = np.clip(img, -160, 240)
    img = (img + 160) / (240 + 160)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    return img


def get_mayo_test_pairs():
    fd_dir = os.path.join(MAYO_DIR, "L506_Testing_211")
    qd_dir = os.path.join(MAYO_DIR, "L506_Testing_Noisy_211")
    fd_files = sorted(glob.glob(os.path.join(fd_dir, "*.IMA")))
    qd_files = sorted(glob.glob(os.path.join(qd_dir, "*.IMA")))
    if len(fd_files) != len(qd_files):
        raise RuntimeError(f"Mismatched L506 test pair counts: {len(fd_files)} full-dose vs {len(qd_files)} quarter-dose")
    return list(zip(qd_files, fd_files))


def evaluate_bm3d_mayo(args):
    pairs = get_mayo_test_pairs()
    if args.limit:
        pairs = pairs[: args.limit]

    csv_path = os.path.join(BASE_DIR, "results_bm3d_mayo.csv")
    fieldnames = ["filename", "psnr", "ssim", "rmse", "time"]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    processed = set()
    existing_results = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            existing_results.append({k: (row[k] if k == "filename" else float(row[k])) for k in fieldnames})
            processed.add(row["filename"])

    print(f"Starting BM3D (fast, pip package) evaluation on {len(pairs)} Mayo L506 test pairs...")
    print(f"Already processed: {len(processed)}")

    for noisy_path, gt_path in tqdm(pairs):
        fname = os.path.basename(noisy_path)
        if fname in processed:
            continue

        noisy_img = read_dicom(noisy_path)   # [0,1] float32, 256x256
        gt_img = read_dicom(gt_path)

        t0 = time.time()
        sigma_est = float(np.mean(estimate_sigma(noisy_img, channel_axis=None)))
        denoised = bm3d.bm3d(noisy_img, sigma_psd=sigma_est)
        denoised = np.clip(denoised, 0.0, 1.0).astype(np.float32)
        runtime = time.time() - t0

        p = float(psnr_fn(gt_img, denoised, data_range=1.0))
        s = float(ssim_fn(gt_img, denoised, data_range=1.0))
        r = float(np.sqrt(np.mean((gt_img - denoised) ** 2)))

        result = {"filename": fname, "psnr": p, "ssim": s, "rmse": r, "time": runtime}
        existing_results.append(result)
        processed.add(fname)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(result)

    if existing_results:
        psnrs = [r["psnr"] for r in existing_results]
        ssims = [r["ssim"] for r in existing_results]
        rmses = [r["rmse"] for r in existing_results]
        m_p, s_p = np.mean(psnrs), np.std(psnrs)
        m_s, s_s = np.mean(ssims), np.std(ssims)
        m_r, s_r = np.mean(rmses), np.std(rmses)
        print(f"\n{'='*65}")
        print(f"Model: BM3D (Mayo L506, sigma auto-estimated per-slice)")
        print(f"Global PSNR: {m_p:.4f} +/- {s_p:.4f} dB")
        print(f"Global SSIM: {m_s:.4f} +/- {s_s:.4f}")
        print(f"Global RMSE: {m_r:.4f} +/- {s_r:.4f}")
        print(f"{'='*65}")
        with open(os.path.join(BASE_DIR, "global_metrics_bm3d_mayo.txt"), "w") as f:
            f.write(f"Global PSNR: {m_p:.4f} +/- {s_p:.4f}\n")
            f.write(f"Global SSIM: {m_s:.4f} +/- {s_s:.4f}\n")
            f.write(f"Global RMSE: {m_r:.4f} +/- {s_r:.4f}\n")
    else:
        print("No images were processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test slices")
    evaluate_bm3d_mayo(parser.parse_args())
