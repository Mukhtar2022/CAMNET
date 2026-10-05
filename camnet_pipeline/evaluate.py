import os
import cv2
import csv
import torch
import torch.nn as nn
import numpy as np
import argparse
import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import torch.nn.functional as F
# Added imports for BM3D and Non-Local Means
try:
    from bm3d import bm3d
except ImportError:
    bm3d = None
from skimage.restoration import denoise_nl_means, estimate_sigma


from dataset import CTDenoisingDataset
from torch.utils.data import DataLoader
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

BASE_DIR = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"
TEST_IMG_DIR = os.path.join(BASE_DIR, "Testing")
TEST_NOISY_DIR = os.path.join(BASE_DIR, "Testing_Noisy")
IMG_DIR = os.path.join(BASE_DIR, "results_images")
SAMPLE_DIR = os.path.join(BASE_DIR, "train_samples")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(gt, out, noisy):
    """
    Computes PSNR, SSIM, RMSE, and GMSD between gt and out tensors [B, 1, H, W] in [0, 1].
    GMSD is computed via Sobel gradient magnitude similarity deviation.
    """
    psnrs, ssims, rmses, gmsds = [], [], [], []
    for i in range(gt.size(0)):
        gt_np = gt[i, 0].cpu().numpy().astype(np.float32)
        out_np = out[i, 0].cpu().numpy().astype(np.float32)
        # PSNR, SSIM, RMSE
        p = psnr_fn(gt_np, out_np, data_range=1.0)
        s = ssim_fn(gt_np, out_np, data_range=1.0)
        r = float(np.sqrt(np.mean((gt_np - out_np) ** 2)))
        # GMSD
        # Sobel gradients
        gx_gt = cv2.Sobel(gt_np, cv2.CV_32F, 1, 0, ksize=3)
        gy_gt = cv2.Sobel(gt_np, cv2.CV_32F, 0, 1, ksize=3)
        gx_out = cv2.Sobel(out_np, cv2.CV_32F, 1, 0, ksize=3)
        gy_out = cv2.Sobel(out_np, cv2.CV_32F, 0, 1, ksize=3)
        grad_gt = np.sqrt(gx_gt ** 2 + gy_gt ** 2)
        grad_out = np.sqrt(gx_out ** 2 + gy_out ** 2)
        C = 0.01
        s_map = (2 * grad_gt * grad_out + C) / (grad_gt ** 2 + grad_out ** 2 + C)
        gmsd = np.sqrt(np.mean((s_map - 1) ** 2))
        psnrs.append(p)
        ssims.append(s)
        rmses.append(r)
        gmsds.append(gmsd)
    return psnrs, ssims, rmses, gmsds

def save_best_sample(model_name, noisy, out, gt, epoch):
    """
    Saves a horizontal grid of (Noisy, Predicted, GroundTruth) for visual tracking.
    """
    noisy_np = (noisy[0, 0].cpu().numpy() * 255).astype(np.uint8)
    out_np   = (out[0, 0].cpu().numpy() * 255).astype(np.uint8)
    gt_np    = (gt[0, 0].cpu().numpy() * 255).astype(np.uint8)
    
    grid = np.hstack([noisy_np, out_np, gt_np])
    path = os.path.join(SAMPLE_DIR, f"{model_name}_epoch_{epoch}.png")
    cv2.putText(grid, f"Epoch: {epoch}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.imwrite(path, grid)

def apply_tta(model, noisy):
    """8-way dihedral TTA: {no-flip, hflip} x {rot0, rot90, rot180, rot270}.
    Averages the model's output over all 8 transforms of the input, each
    inverted back to the original orientation before averaging."""
    outs = []
    for flip in [False, True]:
        for k in [0, 1, 2, 3]:
            aug_in = torch.flip(noisy, dims=[-1]) if flip else noisy
            aug_in = torch.rot90(aug_in, k, dims=[-2, -1])
            aug_out = model(aug_in).clamp(0, 1)
            aug_out = torch.rot90(aug_out, -k, dims=[-2, -1])
            if flip:
                aug_out = torch.flip(aug_out, dims=[-1])
            outs.append(aug_out)
    return torch.stack(outs).mean(dim=0).clamp(0, 1)


def overlapped_inference(model, image, patch_size=64, stride=32, margin=16):
    """
    Groups all image patches into a single batch to utilize GPU parallelism.
    """
    B, C, H, W = image.shape
    pad_h = (stride - (H - patch_size) % stride) % stride
    pad_w = (stride - (W - patch_size) % stride) % stride
    image_padded = F.pad(image, (margin, margin + pad_w, margin, margin + pad_h), mode='reflect')
    H_p, W_p = image_padded.shape[2:]
    
    # 1. Extract all patches
    patches = []
    coords = []
    for y in range(margin, H_p - margin - patch_size // 2 + 1, stride):
        for x in range(margin, W_p - margin - patch_size // 2 + 1, stride):
            # Extract EXACTLY patch_size x patch_size
            patch = image_padded[:, :, y-margin : y-margin+patch_size, x-margin : x-margin+patch_size]
            if patch.shape[2:] != (patch_size, patch_size):
                continue
            patches.append(patch)
            coords.append((y, x))
    
    # 2. Run model on batch
    patches_tensor = torch.cat(patches, dim=0) 
    batch_out = []
    sub_batch_size = 4 
    for i in range(0, patches_tensor.size(0), sub_batch_size):
        chunk = patches_tensor[i : i + sub_batch_size]
        with torch.no_grad():
            batch_out.append(model(chunk))
    
    batch_out = torch.cat(batch_out, dim=0)
    output = torch.zeros((B, C, H_p, W_p), device=DEVICE)
    count = torch.zeros((B, C, H_p, W_p), device=DEVICE)
    
    # 3. Scatter back the central part (stride x stride)
    for i, (y, x) in enumerate(coords):
        out_patch = batch_out[i:i+1]
        output[:, :, y:y+stride, x:x+stride] += out_patch[:, :, margin:margin+stride, margin:margin+stride]
        count[:, :, y:y+stride, x:x+stride] += 1.0

    output = output[:, :, margin:margin+H, margin:margin+W] / count[:, :, margin:margin+H, margin:margin+W].clamp(min=1.0)
    return output.clamp(0, 1)

def evaluate(args):
    # Model Loading
    model_name = args.model.lower()
    print(f"[{model_name.upper()}] Initializing Architecture...")
    if model_name == 'dncnn':
        from models.dncnn import DnCNN
        model = DnCNN()
    elif model_name == 'red_cnn':
        from models.red_cnn import RED_CNN
        model = RED_CNN()
    elif model_name == 'wgan_vgg':
        from models.wgan_vgg import WGANGenerator
        model = WGANGenerator()
    elif model_name == 'edcnn':
        from models.edcnn import EDCNN
        model = EDCNN()
    elif model_name == 'ctformer':
        from models.ctformer import CTFormer
        model = CTFormer()
    elif model_name == 'unet':
        from models.unet import DoGUNet
        model = DoGUNet()
    elif model_name == 'uformer':
        from models.uformer import Uformer
        model = Uformer(img_size=64, in_chans=1, dd_in=1, embed_dim=32, 
                        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], win_size=8)
    elif model_name == 'leda':
        from models.leda import LEDA
        model = LEDA()
    elif model_name == 'proposed_model':
        from models.proposed_red_cnn import Proposed_RED_CNN
        model = Proposed_RED_CNN()
    elif model_name == 'proposed_model_no_ceelpa':
        from models.proposed_red_cnn_no_ceelpa import Proposed_RED_CNN_No_CEELPA
        model = Proposed_RED_CNN_No_CEELPA()
    elif model_name == 'improved_red_cnn_csa':
        from models.improved_red_cnn_csa import ImprovedREDCNN_CSA
        model = ImprovedREDCNN_CSA()
    elif model_name == 'bm3d':
        # BM3D does not require a learned checkpoint
        # We'll handle inside evaluation loop
        model = None
    elif model_name == 'nlm':
        # Non-Local Means (NLM) also does not need a checkpoint
        model = None

    print("   Model structure created.")

    # For BM3D and NLM there is no checkpoint to load
    if model_name in ['bm3d', 'nlm']:
        print('   No checkpoint needed for this algorithm.')
    else:
        suffix_str = f"_{args.suffix}" if args.suffix else ""
        ckpt_path = os.path.join(BASE_DIR, "checkpoints", f"{model_name}{suffix_str}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found at {ckpt_path}")
            return
        
        print(f"   Loading weights from: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
        model.to(DEVICE).eval()
        print("   Weights loaded and moved to GPU.")


    # Data
    # Determine which test directories to use
    if args.gt_dir and args.noisy_dir:
        gt_dir = args.gt_dir
        noisy_dir = args.noisy_dir
        print(f"   Using custom test directories: GT={gt_dir}, Noisy={noisy_dir}")
                # Collect files from both directories, sort them, and pair by order (assuming matching order)
        gt_files = sorted([f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f))])
        noisy_files = sorted([f for f in os.listdir(noisy_dir) if os.path.isfile(os.path.join(noisy_dir, f))])
        pair_count = min(len(gt_files), len(noisy_files))
        pairs = [(os.path.join(noisy_dir, noisy_files[i]), os.path.join(gt_dir, gt_files[i])) for i in range(pair_count)]
        print(f"   Found {len(pairs)} matching pairs in custom directories.")

        test_ds = CTDenoisingDataset(base_dir=BASE_DIR, split='Testing', patch_size=0, augment=False, image_size=256, pairs_override=pairs)
    else:
        print(f"   Scanning testing directory: {TEST_IMG_DIR}")
        test_ds = CTDenoisingDataset(BASE_DIR, split='Testing', patch_size=0, image_size=256)
    print(f"   Done. Found {len(test_ds)} images for testing.")
    
    psnrs, ssims, rmses, gmsds = [], [], [], []
    records = []

    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for idx in tqdm(range(len(test_ds))):
            noisy, gt = test_ds[idx]
            noisy = noisy.unsqueeze(0).to(DEVICE)
            gt = gt.unsqueeze(0).to(DEVICE)

            if model_name == 'ctformer':
                out = overlapped_inference(model, noisy, patch_size=64, margin=16)
            elif model_name == 'uformer':
                # U-former needs patches multiple of 128 (8*2^4). Patch 64 + Margin 32 = 128.
                out = overlapped_inference(model, noisy, patch_size=64, margin=32)
            elif model_name == 'cncl':
                out_full = model(noisy).clamp(0, 1)
                out = out_full[:, 0:1, :, :]  # Restored content channel
            elif model_name == 'leda':
                out = model(noisy).clamp(0, 1)

            elif model_name == 'bm3d':
                # Convert to numpy, apply BM3D, convert back to torch tensor
                noisy_np = noisy[0, 0].cpu().numpy()
                # Estimate noise sigma from noisy image (scaled to [0,1])
                sigma_est = np.mean(np.abs(noisy_np - noisy_np.mean()))
                if bm3d is None:
                    raise ImportError('bm3d package not installed')
                denoised_np = bm3d(noisy_np, sigma_psd=sigma_est)
                out = torch.from_numpy(denoised_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
            elif model_name == 'nlm':
                # Apply Non-Local Means using skimage
                noisy_np = noisy[0, 0].cpu().numpy()
                sigma_est = np.mean(estimate_sigma(noisy_np))
                denoised_np = denoise_nl_means(noisy_np, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)
                out = torch.from_numpy(denoised_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
            elif getattr(args, 'tta', False):
                out = apply_tta(model, noisy)
            else:
                out = model(noisy).clamp(0, 1)

            # Compute metrics (including GMSD)
            p_list, s_list, r_list, g_list = compute_metrics(gt, out, noisy)
            p, s, r, g = p_list[0], s_list[0], r_list[0], g_list[0]

            psnrs.append(p); ssims.append(s); rmses.append(r); gmsds.append(g)

            out_np = (out[0, 0].cpu().numpy() * 255).astype(np.float32)
            gt_np  = (gt[0, 0].cpu().numpy() * 255).astype(np.float32)

            fname = os.path.basename(test_ds.pairs[idx][1])
            records.append({'image': fname, 'psnr': p, 'ssim': s, 'rmse': r, 'gmsd': g})

            # Save visual comparison
            noisy_np = (noisy[0,0].cpu().numpy() * 255).astype(np.uint8)
            grid = np.hstack([noisy_np, out_np.astype(np.uint8), gt_np.astype(np.uint8)])
            cv2.putText(grid, f"PSNR: {p:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            cv2.imwrite(os.path.join(IMG_DIR, f"{model_name}_{fname}.png"), grid)

    # Calculate Mean & SD
    m_psnr, s_psnr = np.mean(psnrs), np.std(psnrs)
    m_ssim, s_ssim = np.mean(ssims), np.std(ssims)
    m_rmse, s_rmse = np.mean(rmses), np.std(rmses)
    m_gmsd, s_gmsd = np.mean(gmsds), np.std(gmsds)

    # Balanced Best-Result Selection (PSNR + SSIM - RMSE) using Z-scores
    z_psnr = (np.array(psnrs) - m_psnr) / (s_psnr + 1e-8)
    z_ssim = (np.array(ssims) - m_ssim) / (s_ssim + 1e-8)
    z_rmse = (np.array(rmses) - m_rmse) / (s_rmse + 1e-8)
    z_gmsd = (np.array(gmsds) - m_gmsd) / (s_gmsd + 1e-8)
    
    combined_scores = z_psnr + z_ssim - z_rmse - z_gmsd
    best_idx = np.argmax(combined_scores)
    best_img = records[best_idx]

    # Restrict which metrics get printed/saved (all 4 are always computed above —
    # best-result selection needs GMSD regardless of this filter)
    requested = {m.strip().lower() for m in args.metrics.split(',') if m.strip()}
    show = (lambda key: not requested or key in requested)

    # Reporting (ASCII to avoid encoding errors)
    print(f"\n{'='*65}")
    print(f"Model: {model_name.upper()}")
    print(f"{'Metric':<15} | {'Mean +/- SD':<20}")
    print(f"{'-'*15}-|-{'-'*20}")
    if show('psnr'): print(f"{'Global PSNR':<15} | {m_psnr:>8.4f} +/- {s_psnr:.4f} dB")
    if show('ssim'): print(f"{'Global SSIM':<15} | {m_ssim:>8.4f} +/- {s_ssim:.4f}")
    if show('rmse'): print(f"{'Global RMSE':<15} | {m_rmse:>8.4f} +/- {s_rmse:.4f}")
    if show('gmsd'): print(f"{'Global GMSD':<15} | {m_gmsd:>8.4f} +/- {s_gmsd:.4f}")
    print(f"{'='*65}")
    print(f"BALANCED BEST RESULT: {best_idx+1} ({best_img['image']})")
    print(f"   PSNR: {best_img['psnr']:.4f} | SSIM: {best_img['ssim']:.4f} | RMSE: {best_img['rmse']:.4f}")
    print(f"{'='*65}\n")

    # Determine the output CSV filename based on suffix
    suffix_str = f"_{args.suffix}" if args.suffix else ""
    out_csv = os.path.join(BASE_DIR, f"results_{model_name}{suffix_str}.csv")
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image','psnr','ssim','rmse','gmsd'])
        writer.writeheader()
        writer.writerows([{**rec, 'gmsd': g} for rec, g in zip(records, gmsds)])
    print(f"Results saved: {out_csv}")

    # Save Global Metrics Summary
    summary_txt = os.path.join(BASE_DIR, f"global_metrics_{model_name}{suffix_str}.txt")
    with open(summary_txt, 'w') as f:
        f.write(f"MODEL: {model_name.upper()}\n")
        f.write(f"SUFFIX: {args.suffix}\n")
        f.write(f"{'='*30}\n")
        if show('psnr'): f.write(f"Global PSNR: {m_psnr:.4f} +/- {s_psnr:.4f} dB\n")
        if show('ssim'): f.write(f"Global SSIM: {m_ssim:.4f} +/- {s_ssim:.4f}\n")
        if show('rmse'): f.write(f"Global RMSE: {m_rmse:.4f} +/- {s_rmse:.4f}\n")
        if show('gmsd'): f.write(f"Global GMSD: {m_gmsd:.4f} +/- {s_gmsd:.4f}\n")
        f.write(f"{'='*30}\n")
        f.write(f"BALANCED BEST RESULT: {best_img['image']}\n")
        f.write(f"   PSNR: {best_img['psnr']:.4f} | SSIM: {best_img['ssim']:.4f}\n")
    print(f"Global summary saved: {summary_txt}")

    # Save a special copy of the best result image to root for easy access
    best_img_name = f"{model_name}_{best_img['image']}"
    best_img_src = os.path.join(IMG_DIR, best_img_name)
    best_img_dst = os.path.join(BASE_DIR, f"BEST_RESULT_{model_name}{suffix_str}.png")
    import shutil
    if os.path.exists(best_img_src):
        shutil.copy(best_img_src, best_img_dst)
        print(f"Best result image saved to root: {best_img_dst}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['dncnn','red_cnn','wgan_vgg','edcnn','ctformer','unet','uformer','proposed_model','proposed_model_no_ceelpa','proposed_model_amdg','proposed_model_v2','cncl','leda','bm3d','nlm','improved_red_cnn_csa'])
    parser.add_argument('--suffix', type=str, default='',
                        help='Optional suffix to append to the output results CSV (e.g., no_msfe)')
    parser.add_argument('--gt_dir', type=str, default='',
                        help='Path to ground truth (clean) images directory for custom evaluation')
    parser.add_argument('--noisy_dir', type=str, default='',
                        help='Path to noisy input images directory for custom evaluation')
    parser.add_argument('--metrics', type=str, default='',
                        help='Comma-separated subset to print/save (e.g. "psnr,ssim,rmse"). '
                             'All 4 metrics are still computed (best-result selection needs GMSD); '
                             'this only narrows what is printed and written to global_metrics_*.txt.')
    parser.add_argument('--tta', action='store_true',
                        help='Apply 8-way dihedral test-time augmentation (average over 4 rotations x hflip on/off) '
                             'during inference. Not used for ctformer/uformer (overlapped_inference), '
                             'cncl, leda, bm3d, or nlm, which have their own dedicated inference paths.')
    evaluate(parser.parse_args())
