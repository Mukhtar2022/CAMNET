import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
from BM3D.BM3D import BM3D_Step1, BM3D_Step2
import argparse
import time

np.random.seed(42)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # Simplified SSIM for speed if needed, but using cv2's quality if available
    # For now, using a simple structural similarity index
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1, img2, data_range=255)

def evaluate_bm3d(args):
    base_dir = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"
    noisy_dir = os.path.join(base_dir, "Testing_Noisy")
    gt_dir = os.path.join(base_dir, "Testing")
    save_dir = os.path.join(base_dir, "results_images")
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, "results_bm3d.csv")
    fieldnames = ['filename', 'psnr', 'ssim', 'rmse', 'time']
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Load existing results to skip and for final averages
    existing_results = []
    processed_files = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_results.append({
                    'filename': row['filename'],
                    'psnr': float(row['psnr']),
                    'ssim': float(row['ssim']),
                    'rmse': float(row['rmse']),
                    'time': float(row['time'])
                })
                processed_files.add(row['filename'])

    filenames = [f for f in os.listdir(noisy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if args.limit:
        filenames = filenames[:args.limit]

    print(f"Starting BM3D Evaluation on {len(filenames)} images...")
    print(f"Already processed (in CSV): {len(processed_files)} images.")
    print(f"Using sigma={args.sigma}. Note: This is slow (approx 10-20 mins per 256x256 image).")

    for fname in tqdm(filenames):
        if fname in processed_files:
            continue
            
        # Check if visual comparison already exists in results_images
        comp_path = os.path.join(save_dir, f"bm3d_{fname}")
        if os.path.exists(comp_path):
            # Try to recover metrics from the existing image to avoid re-processing
            comp_img = cv2.imread(comp_path, cv2.IMREAD_GRAYSCALE)
            if comp_img is not None:
                h, w = comp_img.shape
                if w % 3 == 0:
                    unit_w = w // 3
                    # The combined image is [Noisy, Denoised, GT]
                    final_img = comp_img[:, unit_w:2*unit_w]
                    
                    gt_path = os.path.join(gt_dir, fname)
                    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    
                    if gt_img is not None:
                        psnr_val = calculate_psnr(gt_img, final_img)
                        ssim_val = calculate_ssim(gt_img, final_img)
                        rmse_val = np.sqrt(np.mean((gt_img.astype(float) - final_img.astype(float))**2))

                        result = {
                            'filename': fname,
                            'psnr': psnr_val,
                            'ssim': ssim_val,
                            'rmse': rmse_val,
                            'time': 0.0 # Unknown time
                        }
                        
                        existing_results.append(result)
                        processed_files.add(fname)
                        
                        # Append to CSV
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow(result)
                        continue

        noisy_path = os.path.join(noisy_dir, fname)
        gt_path = os.path.join(gt_dir, fname)
        
        if not os.path.exists(gt_path):
            continue

        # Load images as grayscale
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if noisy_img is None or gt_img is None:
            continue

        # Step 1: Basic Estimate
        t0 = time.time()
        basic_img = BM3D_Step1(noisy_img, sigma=args.sigma)
        
        # Step 2: Final Estimate
        final_img = BM3D_Step2(basic_img, noisy_img, sigma=args.sigma)
        runtime = time.time() - t0

        # Post-process (clip and normalize)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        # Metrics
        psnr_val = calculate_psnr(gt_img, final_img)
        ssim_val = calculate_ssim(gt_img, final_img)
        rmse_val = np.sqrt(np.mean((gt_img.astype(float) - final_img.astype(float))**2))

        result = {
            'filename': fname,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'rmse': rmse_val,
            'time': runtime
        }
        
        existing_results.append(result)
        processed_files.add(fname)

        # Append to CSV immediately
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(result)

        # Save visual comparison
        combined = np.hstack([noisy_img, final_img, gt_img])
        cv2.imwrite(os.path.join(save_dir, f"bm3d_{fname}"), combined)

    # Calculate averages from all results (existing + new)
    if existing_results:
        psnrs = [r['psnr'] for r in existing_results]
        ssims = [r['ssim'] for r in existing_results]
        rmses = [r['rmse'] for r in existing_results]

        m_psnr, s_psnr = np.mean(psnrs), np.std(psnrs)
        m_ssim, s_ssim = np.mean(ssims), np.std(ssims)
        m_rmse, s_rmse = np.mean(rmses), np.std(rmses)

        # Balanced Best-Result Selection (PSNR + SSIM - RMSE) using Z-scores
        z_psnr = (np.array(psnrs) - m_psnr) / (s_psnr + 1e-8)
        z_ssim = (np.array(ssims) - m_ssim) / (s_ssim + 1e-8)
        z_rmse = (np.array(rmses) - m_rmse) / (s_rmse + 1e-8)
        
        combined_scores = z_psnr + z_ssim - z_rmse
        best_idx = np.argmax(combined_scores)
        best_img = existing_results[best_idx]

        print(f"\n{'='*65}")
        print(f"Model: BM3D (sigma={args.sigma})")
        print(f"{'Metric':<15} | {'Mean +/- SD':<20}")
        print(f"{'-'*15}-|-{'-'*20}")
        print(f"{'Global PSNR':<15} | {m_psnr:>8.4f} +/- {s_psnr:.4f} dB")
        print(f"{'Global SSIM':<15} | {m_ssim:>8.4f} +/- {s_ssim:.4f}")
        print(f"{'Global RMSE':<15} | {m_rmse:>8.4f} +/- {s_rmse:.4f}")
        print(f"{'='*65}")
        print(f"BALANCED BEST RESULT: {best_idx+1} ({best_img['filename']})")
        print(f"   PSNR: {best_img['psnr']:.4f} | SSIM: {best_img['ssim']:.4f} | RMSE: {best_img['rmse']:.4f}")
        print(f"{'='*65}\n")
        print(f"Results saved and appended to results_bm3d.csv")
    else:
        print("No images were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=int, default=25, help='Noise standard deviation')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to test')
    evaluate_bm3d(parser.parse_args())
