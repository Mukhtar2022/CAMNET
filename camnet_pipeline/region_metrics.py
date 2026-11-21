"""Region-based image-quality metrics reported in the CAMNet paper.

All images are float32 arrays in [0, 1] (data range 1.0), the same scale as PSNR/SSIM/RMSE.

  Edge PSNR  PSNR on the 10% of pixels with the strongest ground-truth Sobel gradient magnitude.
  EPI        Edge preservation index: Pearson correlation between the Laplacian-filtered
             denoised image and the Laplacian-filtered reference.
  ST-PSNR    Soft-tissue PSNR: PSNR on body pixels (reference > 0.05) whose reference intensity
             lies within one standard deviation of the mean body intensity of that slice.

PSNR and SSIM use scikit-image with data_range=1.0. RMSE is the root mean squared error.
Means are taken over test slices; standard deviations use ddof=1.
"""
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn

EPS = 1e-12
METRICS = ['PSNR', 'SSIM', 'RMSE', 'EdgePSNR', 'EPI', 'ST-PSNR']
HIGHER_IS_BETTER = {'PSNR': True, 'SSIM': True, 'RMSE': False, 'EdgePSNR': True, 'EPI': True, 'ST-PSNR': True}


def sobel_mag(x):
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def edge_psnr(gt, out):
    g = sobel_mag(gt)
    mask = g >= np.percentile(g, 90)                       # top-10% ground-truth edge pixels
    return 10 * np.log10(1.0 / np.mean((gt[mask] - out[mask]) ** 2))


def epi(gt, out):
    lg = cv2.Laplacian(gt, cv2.CV_32F); lo = cv2.Laplacian(out, cv2.CV_32F)
    lg -= lg.mean(); lo -= lo.mean()
    return float(np.sum(lg * lo) / np.sqrt(np.sum(lg ** 2) * np.sum(lo ** 2)))


def soft_tissue_mask(gt):
    body = gt > 0.05                                       # exclude air / background
    mu, sd = gt[body].mean(), gt[body].std()
    return body & (np.abs(gt - mu) <= sd)


def st_psnr(gt, out):
    m = soft_tissue_mask(gt)
    return 10 * np.log10(1.0 / max(np.mean((gt[m] - out[m]) ** 2), EPS))


def all_metrics(gt, out):
    """Return [PSNR, SSIM, RMSE, EdgePSNR, EPI, ST-PSNR] for one slice."""
    gt = np.asarray(gt, np.float32); out = np.asarray(out, np.float32)
    return [psnr_fn(gt, out, data_range=1.0), ssim_fn(gt, out, data_range=1.0),
            float(np.sqrt(np.mean((gt - out) ** 2))), edge_psnr(gt, out), epi(gt, out), st_psnr(gt, out)]
