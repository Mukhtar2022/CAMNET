"""
dataset.py  –  Paired CT Image Dataset for denoising baselines
================================================================
Loads matched (noisy_input, clean_target) image pairs from:
    {BASE_DIR}/{split}_Noisy/   →  input
    {BASE_DIR}/{split}/         →  target (ground truth)

All images are resized to 256x256, converted to grayscale float32
tensors in [0, 1].

During training, random 64x64 (or configurable) patches are cropped.
During validation / testing, full 256x256 images are used.
"""

import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class CTDenoisingDataset(Dataset):
    """
    Parameters
    ----------
    base_dir   : str  – root folder (contains Training/, Testing/, Validation/ ...)
    split      : str  – 'Training' | 'Testing' | 'Validation'
    patch_size : int  – square patch size for training crops (0 = full image)
    augment    : bool – random hflip / vflip / 90° rot during training
    image_size : int  – resize target before patching (default 256)
    return_noise: bool – if True, returns (noisy, [clean, noise])
    """

    def __init__(self, base_dir=None, split='Training',
                 patch_size=64, augment=True, image_size=256, return_noise=False, pairs_override=None):
        self.patch_size   = patch_size
        self.augment      = augment and (split == 'Training')
        self.image_size   = image_size
        self.return_noise = return_noise

        self.pairs = []
        if pairs_override is not None:
            self.pairs = pairs_override
        else:
            gt_dir    = os.path.join(base_dir, split)
            noisy_dir = os.path.join(base_dir, f"{split}_Noisy")

            gt_paths = sorted(
                glob.glob(os.path.join(gt_dir,    '*.png')) +
                glob.glob(os.path.join(gt_dir,    '*.jpg'))
            )
            for gt_p in gt_paths:
                fname   = os.path.basename(gt_p)
                noisy_p = os.path.join(noisy_dir, fname)
                if os.path.exists(noisy_p):
                    self.pairs.append((noisy_p, gt_p))

            if len(self.pairs) == 0:
                raise RuntimeError(f"No matched pairs found for split='{split}' in {base_dir}")

        # Cache all images in RAM — eliminates per-epoch disk I/O
        print(f"   Caching {len(self.pairs)} image pairs into RAM...", flush=True)
        self._cache = [(self._read_gray(n), self._read_gray(g)) for n, g in self.pairs]
        print(f"   Done.", flush=True)

    def __len__(self):
        return len(self.pairs)

    def _read_gray(self, path):
        # Consistent scaling for both DICOM and regular images
        ext = os.path.splitext(path)[1].lower()
        is_dicom = False
        if ext in ['.ima', '.dcm'] or ext == '':
            try:
                import pydicom
                pydicom.dcmread(path, stop_before_pixels=True)
                is_dicom = True
            except Exception:
                is_dicom = False

        if is_dicom:
            import pydicom
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            # Apply CT window -160 to 240 HU
            img = np.clip(img, -160, 240)
            img = (img + 160) / (240 + 160)  # → [0, 1]
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Fallback: try common alternatives if extension mismatch
                for alt_ext in ['.png', '.jpg', '.jpeg', '.tif']:
                    alt_path = os.path.splitext(path)[0] + alt_ext
                    if os.path.exists(alt_path):
                        img = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None: break
            if img is None:
                raise FileNotFoundError(f"Unable to read image at {path}")
            img = img.astype(np.float32) / 255.0
        # Resize if necessary
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx):
        noisy, gt = self._cache[idx]
        # Work on copies so augmentation doesn't corrupt the cache
        noisy = noisy.copy()
        gt    = gt.copy()

        # ── Random patch crop (training only) ──────────────────────
        if self.patch_size > 0 and self.patch_size < self.image_size:
            h, w = noisy.shape
            top  = random.randint(0, h - self.patch_size)
            left = random.randint(0, w - self.patch_size)
            noisy = noisy[top:top+self.patch_size, left:left+self.patch_size]
            gt    = gt   [top:top+self.patch_size, left:left+self.patch_size]

        # ── Data augmentation ───────────────────────────────────────
        if self.augment:
            if random.random() > 0.5:
                noisy = np.fliplr(noisy).copy()
                gt    = np.fliplr(gt).copy()
            if random.random() > 0.5:
                noisy = np.flipud(noisy).copy()
                gt    = np.flipud(gt).copy()
            k = random.randint(0, 3)
            if k > 0:
                noisy = np.rot90(noisy, k).copy()
                gt    = np.rot90(gt,    k).copy()

        # ── To tensor: add channel dim → (1, H, W) ─────────────────
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)
        gt_t    = torch.from_numpy(gt).unsqueeze(0)

        if self.return_noise:
            # NoiseMap = noisy - gt
            noise_t = noisy_t - gt_t
            target  = torch.cat([gt_t, noise_t], dim=0) # [2, H, W]
            return noisy_t, target

        return noisy_t, gt_t


def get_dataloaders(base_dir, patch_size=64, batch_size=16,
                    num_workers=0, image_size=256, return_noise=False):
    """
    Returns train_loader, val_loader, test_loader.
    """
    train_ds = CTDenoisingDataset(base_dir, 'Training',   patch_size=patch_size, augment=True,  image_size=image_size, return_noise=return_noise)
    val_ds   = CTDenoisingDataset(base_dir, 'Validation',  patch_size=0,          augment=False, image_size=image_size, return_noise=return_noise)
    test_ds  = CTDenoisingDataset(base_dir, 'Testing',     patch_size=0,          augment=False, image_size=image_size, return_noise=return_noise)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_global_pairs(base_dir):
    """Gathers all matched pairs from Training and Validation only.
    Testing is excluded to prevent data leakage in cross-validation."""
    all_pairs = []
    for split in ['Training', 'Validation']:
        gt_dir    = os.path.join(base_dir, split)
        noisy_dir = os.path.join(base_dir, f"{split}_Noisy")
        if not os.path.exists(gt_dir) or not os.path.exists(noisy_dir):
            continue
            
        gt_paths = sorted(
            glob.glob(os.path.join(gt_dir, '*.png')) + glob.glob(os.path.join(gt_dir, '*.jpg'))
        )
        for gt_p in gt_paths:
            fname   = os.path.basename(gt_p)
            noisy_p = os.path.join(noisy_dir, fname)
            if os.path.exists(noisy_p):
                all_pairs.append((noisy_p, gt_p))
    return all_pairs

def get_kfold_dataloaders(pairs_train, pairs_val, pairs_test, patch_size=64, batch_size=16, num_workers=0, image_size=256, return_noise=False):
    """Returns dataloaders using explicitly provided lists of file pairs."""
    train_ds = CTDenoisingDataset(split='Training', patch_size=patch_size, augment=True, image_size=image_size, return_noise=return_noise, pairs_override=pairs_train)
    val_ds   = CTDenoisingDataset(split='Validation', patch_size=0, augment=False, image_size=image_size, return_noise=return_noise, pairs_override=pairs_val)
    test_ds  = CTDenoisingDataset(split='Testing', patch_size=0, augment=False, image_size=image_size, return_noise=return_noise, pairs_override=pairs_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_mayo_pairs(mayo_dir, val_patient='L310', val_count=211):
    """Builds Training/Validation pairs from the Mayo Grand Challenge dataset
    (9 patients: L067, L096, L109, L143, L192, L286, L291, L310, L333).
    L506 is intentionally excluded here -- it is the fixed cross-dataset
    test set used elsewhere (see get_mayo_test_pairs). Full-dose and
    quarter-dose files do not share filenames, so pairing is done by sorted
    order within each patient (filenames sort as L067_..., L096_..., ...,
    so a global sort keeps every patient's slices contiguous and in order).

    val_patient is held out from training entirely; only the first val_count
    of its slices are used for validation (211, matching the L506 test set
    size exactly) -- any remainder is simply unused, never leaked into
    training, to keep the split patient-disjoint."""
    import re
    fd_dir = os.path.join(mayo_dir, 'Full-dose', 'full_1mm_merged')
    qd_dir = os.path.join(mayo_dir, 'Lower-dose', 'quarter_1mm_merged')
    fd_files = sorted(glob.glob(os.path.join(fd_dir, '*.IMA')))
    qd_files = sorted(glob.glob(os.path.join(qd_dir, '*.IMA')))
    if len(fd_files) != len(qd_files):
        raise RuntimeError(f"Mismatched Mayo pair counts: {len(fd_files)} full-dose vs {len(qd_files)} quarter-dose")

    pairs_train, val_candidates = [], []
    for qd_p, fd_p in zip(qd_files, fd_files):
        m = re.search(r'L\d{3}', os.path.basename(fd_p))
        patient = m.group(0) if m else None
        if patient == val_patient:
            val_candidates.append((qd_p, fd_p))
        else:
            pairs_train.append((qd_p, fd_p))

    if len(val_candidates) < val_count:
        raise RuntimeError(f"val_patient={val_patient} only has {len(val_candidates)} slices, need {val_count}")
    pairs_val = val_candidates[:val_count]
    return pairs_train, pairs_val


def get_mayo_test_pairs(mayo_dir):
    """L506's fixed 211-slice cross-dataset test set (same folders already
    used by evaluate.py's --gt_dir/--noisy_dir mechanism for Mayo testing)."""
    fd_dir = os.path.join(mayo_dir, 'L506_Testing_211')
    qd_dir = os.path.join(mayo_dir, 'L506_Testing_Noisy_211')
    fd_files = sorted(glob.glob(os.path.join(fd_dir, '*.IMA')))
    qd_files = sorted(glob.glob(os.path.join(qd_dir, '*.IMA')))
    if len(fd_files) != len(qd_files):
        raise RuntimeError(f"Mismatched L506 test pair counts: {len(fd_files)} full-dose vs {len(qd_files)} quarter-dose")
    return list(zip(qd_files, fd_files))


def get_piglet_pairs(piglet_dir, train_frac=0.8, val_frac=0.1):
    """Builds Training/Validation/Testing pairs from the Piglet CT dataset.

    This is a single-subject volume (one PatientID, one series -- FD_850 is
    the 300 mAs full-dose ground truth, LD25_850 the 75 mAs quarter-dose
    noisy input, 850 slices each). Filename numeric suffixes do NOT match
    DICOM slice order, so slices are sorted by InstanceNumber and split into
    contiguous anatomical blocks (80/10/10) rather than randomly, to avoid
    leaking near-duplicate adjacent slices across the train/val/test
    boundary the way a random per-slice split would."""
    import pydicom
    fd_dir = os.path.join(piglet_dir, 'FD_850')
    qd_dir = os.path.join(piglet_dir, 'LD25_850')
    common = sorted(set(os.listdir(fd_dir)) & set(os.listdir(qd_dir)))
    if not common:
        raise RuntimeError(f"No matching Piglet FD/LD25 filenames found in {fd_dir} / {qd_dir}")

    records = []
    for f in common:
        ds = pydicom.dcmread(os.path.join(fd_dir, f), stop_before_pixels=True)
        records.append((int(ds.InstanceNumber), f))
    records.sort(key=lambda r: r[0])

    n = len(records)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    def make_pairs(recs):
        return [(os.path.join(qd_dir, f), os.path.join(fd_dir, f)) for _, f in recs]

    pairs_train = make_pairs(records[:n_train])
    pairs_val = make_pairs(records[n_train:n_train + n_val])
    pairs_test = make_pairs(records[n_train + n_val:])
    return pairs_train, pairs_val, pairs_test


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    BASE_DIR = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"
    train_l, val_l, test_l = get_dataloaders(BASE_DIR, patch_size=64, batch_size=16, num_workers=0)
    noisy, gt = next(iter(train_l))
    print(f"Train batch — noisy: {noisy.shape}, gt: {gt.shape}")
