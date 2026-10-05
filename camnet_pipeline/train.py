import os
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LinearLR, SequentialLR
from torch.amp import autocast, GradScaler

from dataset import get_dataloaders
from evaluate import compute_metrics, save_best_sample, overlapped_inference
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

# ─────────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR  = os.path.join(BASE_DIR, "logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

MODEL_CONFIGS = {
    'dncnn': dict(
        patch_size  = 40,           # Original DnCNN paper patch
        batch_size  = 64,           # Paper batch
        epochs      = 50,
        lr          = 1e-3,
        lr_milestones=[30, 45],
        loss        = 'mse',
        optim       = 'adam'
    ),
    'red_cnn': dict(
        patch_size  = 55,           # Reverted to Original RED-CNN paper patch (REPLICATING 35.45 dB)
        batch_size  = 16,           # Paper batch
        epochs      = 50,
        lr          = 1e-4,
        lr_milestones=[30, 45],
        loss        = 'mse',        # Reverted to Vanilla MSE for baseline study
        optim       = 'adam'
    ),
    'wgan_vgg': dict(
        patch_size  = 64,           # WGAN-VGG paper uses 64x64
        batch_size  = 16,
        epochs      = 50,
        lr          = 1e-4,
        lr_milestones=[30, 45],
        loss        = 'wgan_vgg',
        optim       = 'adam',
        n_critic    = 4
    ),
    'edcnn': dict(
        patch_size  = 64,
        batch_size  = 16,
        epochs      = 50,
        lr          = 1e-4,
        lr_milestones=[30, 45],
        loss        = 'mse',
        optim       = 'adam'
    ),
    'ctformer': dict(
        patch_size  = 64,
        batch_size  = 8,            # CT-Former is memory intensive
        epochs      = 50,
        lr          = 1e-4,
        lr_milestones=[30, 45],
        loss        = 'mse',        # Weighted MSE 100x in original
        optim       = 'adam'
    ),
    'unet': dict(
        patch_size  = 64,
        batch_size  = 16,
        epochs      = 50,
        lr          = 1e-4,
        lr_milestones=[30, 45],
        loss        = 'mse',
        optim       = 'adam'
    ),
    'uformer': dict(
        patch_size  = 64,
        batch_size  = 8,
        epochs      = 50,
        lr          = 2e-4,
        lr_milestones=[30, 45],
        loss        = 'l1',
        optim       = 'adam_w',
        adam_betas  = (0.9, 0.999),
        weight_decay= 0.02
    ),
    'proposed_model': dict(
        patch_size  = 64,           # RPD Report
        batch_size  = 8,            # RPD Preliminary result batch
        epochs      = 50,
        lr          = 1e-4,         # RPD Recommended LR
        lr_milestones=[60, 90],     # scaled proportionally for a 100-epoch run (was [30,45] @ 50 epochs)
        loss        = 'compound_custom', # Huber + SSIM + Gradient
        optim       = 'adam'
    ),
    'proposed_model_no_ceelpa': dict(
        patch_size  = 64,           # Identical to full model for fair ablation
        batch_size  = 8,
        epochs      = 50,
        lr          = 1e-4,
        lr_milestones=[30, 45],
        loss        = 'compound_custom', # Same compound loss
        optim       = 'adam'
    ),
    'cncl': dict(
        patch_size  = 64,           # Geng et al. 2022
        batch_size  = 8,            # Paper default
        epochs      = 50,
        lr          = 2e-4,         # Paper default
        lr_milestones=[30, 45],
        loss        = 'cncl_gan',   # Pixel L1 + LSGAN
        optim       = 'adam',
        lambda_pixel = 100          # Paper default
    ),
    'leda': dict(
        patch_size  = 64,
        batch_size  = 8,
        epochs      = 50,
        lr          = 2e-4,
        lr_milestones=[30, 45],
        loss        = 'l1',         # Running standard L1 for LEDA-ESAU natively
        optim       = 'adam'
    ),
    'improved_red_cnn_csa': dict(
        patch_size  = 64,           # Source paper used 64x64 patches
        batch_size  = 16,           # Source paper batch
        epochs      = 50,           # Source paper used 60; standardised to 50 to match this pipeline
        lr          = 1e-4,         # Source paper LR (Adam)
        lr_milestones=[30, 45],     # Source paper decayed x0.1 at epoch 20/60; rescaled to this pipeline's 50-epoch schedule
        loss        = 'mse_ssim',   # Source paper: L = 1.0*MSE + 1.0*(1-SSIM)
        optim       = 'adam'
    )
}

# ─────────────────────────────────────────────────────────────────
# Loss Helper Classes
# ─────────────────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:35]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        # input/target: [B, 1, H, W] in [0, 1]
        fake = input.repeat(1, 3, 1, 1)
        real = target.repeat(1, 3, 1, 1)
        
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        
        return self.criterion(fake_feature, real_feature)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.exp(torch.tensor([-(x - window_size//2)**2 / float(2*sigma**2) for x in range(window_size)]))
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self). __init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x, y):
        grad_x_pred = F.conv2d(x, self.kernel_x, padding=1)
        grad_y_pred = F.conv2d(x, self.kernel_y, padding=1)
        grad_x_gt = F.conv2d(y, self.kernel_x, padding=1)
        grad_y_gt = F.conv2d(y, self.kernel_y, padding=1)
        
        loss = F.mse_loss(grad_x_pred, grad_x_gt) + F.mse_loss(grad_y_pred, grad_y_gt)
        return loss

def compute_gradient_penalty(netD, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = netD(interpolates)
    fake = torch.full((real_samples.size(0), 1, d_interpolates.size(2), d_interpolates.size(3)), 1.0, device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ─────────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────────

def load_model(name, suffix=''):
    if name == 'dncnn':
        from models.dncnn import DnCNN
        return DnCNN(image_channels=1)
    elif name == 'red_cnn':
        from models.red_cnn import RED_CNN
        return RED_CNN()
    elif name == 'wgan_vgg':
        from models.wgan_vgg import WGANGenerator
        return WGANGenerator()
    elif name == 'edcnn':
        from models.edcnn import EDCNN
        return EDCNN()
    elif name == 'ctformer':
        from models.ctformer.CTformer import CTFormer
        return CTFormer()
    elif name == 'unet':
        from models.unet import UNet
        return UNet(n_channels=1, n_classes=1)
    elif name == 'proposed_model':
        from models.proposed_red_cnn import Proposed_RED_CNN
        return Proposed_RED_CNN()
    elif name == 'proposed_model_no_ceelpa':
        from models.proposed_red_cnn_no_ceelpa import Proposed_RED_CNN_No_CEELPA
        return Proposed_RED_CNN_No_CEELPA()
    elif name == 'uformer':
        from models.uformer import Uformer
        return Uformer(img_size=64, in_chans=1, dd_in=1, embed_dim=32, 
                       depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], win_size=8)
    elif name == 'cncl':
        from models.cncl import GeneratorUNet
        return GeneratorUNet()
    elif name == 'leda':
        from models.leda import LEDA_ESAU
        return LEDA_ESAU()
    elif name == 'improved_red_cnn_csa':
        from models.improved_red_cnn_csa import ImprovedREDCNN_CSA
        return ImprovedREDCNN_CSA()
    else: raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────

def run_training_loop(args, cfg, train_l, val_l, test_l, patch_size, batch_size, epochs, lr, loss_type, milestones, suffix_str, fold=None, start_epoch_override=None):
    from evaluate import compute_metrics, save_best_sample, overlapped_inference
    
    fold_prefix = f"[Fold {fold}] " if fold is not None else ""
    return_noise = (args.model == 'cncl')
    print(f"{fold_prefix}Building model: {args.model}...")
    model = load_model(args.model, getattr(args, 'suffix', '')).to(DEVICE)
    if args.model == 'proposed_model' and (args.disable_ceelpa or args.disable_rca or args.disable_msfe):
        # Only route through the architecture-ablation wrapper when a component is
        # actually being disabled. Otherwise keep the exact same class (Proposed_RED_CNN)
        # that evaluate.py loads checkpoints into, so training and evaluation always
        # share one computation graph.
        from models.proposed_red_cnn_ablation import Proposed_RED_CNN_Abal
        model = Proposed_RED_CNN_Abal(disable_ceelpa=args.disable_ceelpa,
                                     disable_rca=args.disable_rca,
                                     disable_msfe=args.disable_msfe).to(DEVICE)
    if args.ckpt_path:
        if os.path.exists(args.ckpt_path):
            model.load_state_dict(torch.load(args.ckpt_path, map_location=DEVICE))
            print(f"{fold_prefix}Loaded checkpoint from {args.ckpt_path}")
        else:
            print(f"{fold_prefix}Checkpoint path {args.ckpt_path} not found")
    print(f"{fold_prefix}Model moved to {DEVICE}.")
    
    # We dynamically inject the fold number into the checkpoint name
    best_ckpt = os.path.join(CKPT_DIR, f"{args.model}{suffix_str}_best.pth")
    
    if args.resume:
        if os.path.exists(best_ckpt):
            print(f"{fold_prefix}Loading weights from: {best_ckpt}")
            model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
            print(f"{fold_prefix}Resumed successfully.")
        else:
            print(f"{fold_prefix}No checkpoint found at {best_ckpt}")
            return

    print(f"{fold_prefix}Initializing Loss Functions...")
    loss_vgg = VGGPerceptualLoss() if loss_type in ['wgan_vgg', 'compound', 'pdlm_loss'] else None
    criterion_mse   = nn.MSELoss()
    print(f"{fold_prefix}   -> Setting up Huber Loss...")
    criterion_huber = nn.HuberLoss(delta=1.0)
    
    print(f"{fold_prefix}   -> Setting up SSIM Loss (moving to GPU)...")
    criterion_ssim  = SSIMLoss().to(DEVICE)
    
    print(f"{fold_prefix}   -> Setting up Gradient Loss (moving to GPU)...")
    criterion_grad  = GradientLoss().to(DEVICE)
    
    weight_decay = getattr(args, 'weight_decay', 0.0) or 0.0
    print(f"{fold_prefix}   -> Configuring Adam Optimizer (lr={lr}, weight_decay={weight_decay})...")
    adam_betas    = cfg.get('adam_betas', (0.9, 0.999))
    optimizer_g   = optim.Adam(model.parameters(), lr=lr, betas=adam_betas, weight_decay=weight_decay)

    # Mixed-precision training: same math, same architecture, computed in fp16 where
    # safe -- typically 1.5-2.5x faster on modern GPUs. No-ops safely on CPU.
    # CTFormer is excluded: its attention/softmax layers overflow to inf/nan under
    # fp16 autocast (confirmed via repeated NaN-at-epoch-2 runs), and gradient
    # clipping alone can't fix a forward pass that's already nan -- full fp32 avoids
    # the failure mode entirely, at negligible cost on this Pascal GPU (no Tensor Cores).
    amp_enabled = (DEVICE.type == 'cuda') and (args.model != 'ctformer')
    scaler = GradScaler(DEVICE.type, enabled=amp_enabled)
    
    warmup_arg = getattr(args, 'warmup_epochs', None)
    warmup_epochs = warmup_arg if warmup_arg is not None else 5
    if warmup_epochs > 0:
        print(f"{fold_prefix}   -> Setting up LR Scheduler (with {warmup_epochs}-epoch Warmup)...")
        warmup_sched = LinearLR(optimizer_g, start_factor=0.01, total_iters=warmup_epochs)
        shifted_milestones = [m - warmup_epochs for m in milestones if m > warmup_epochs]
        main_sched = MultiStepLR(optimizer_g, milestones=shifted_milestones, gamma=0.1)
        scheduler_g = SequentialLR(optimizer_g, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs])
    else:
        print(f"{fold_prefix}   -> Setting up LR Scheduler (NO warmup)...")
        scheduler_g = MultiStepLR(optimizer_g, milestones=milestones, gamma=0.1)


    net_d, optimizer_d = None, None
    if loss_type == 'wgan_vgg':
        from models.wgan_vgg import WGANDiscriminator
        net_d = WGANDiscriminator().to(DEVICE)
        optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=adam_betas)
    elif loss_type == 'cncl_gan':
        from models.cncl import Discriminator
        net_d = Discriminator(in_channels=2).to(DEVICE)
        from models.cncl import weights_init_normal
        net_d.apply(weights_init_normal)
        optimizer_d = optim.Adam(net_d.parameters(), lr=lr, betas=(0.5, 0.999))

    log_path = os.path.join(LOG_DIR, f"{args.model}{suffix_str}_train_log.csv")
    if not os.path.exists(log_path) and fold is None:
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch','loss','psnr','ssim','rmse','time'])

    best_psnr = 0.0

    start_ep = start_epoch_override if start_epoch_override is not None else args.start_epoch
    for epoch in range(start_ep, epochs + 1):
        print(f"\n{fold_prefix}Starting Epoch {epoch}/{epochs}...")
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for noisy, gt in train_l:
            noisy, gt = noisy.to(DEVICE), gt.to(DEVICE)
            
            if loss_type == 'wgan_vgg':
                for _ in range(cfg.get('n_critic', 4)):
                    optimizer_d.zero_grad()
                    fake_images = model(noisy).detach()
                    real_validity = net_d(gt)
                    fake_validity = net_d(fake_images)
                    gp = compute_gradient_penalty(net_d, gt, fake_images)
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gp
                    d_loss.backward()
                    optimizer_d.step()

                optimizer_g.zero_grad()
                gen_images = model(noisy)
                fake_validity = net_d(gen_images)
                loss = 0.1 * loss_vgg(gen_images, gt) - torch.mean(fake_validity)
                loss.backward()
                optimizer_g.step()
                running_loss += loss.item()

            elif loss_type == 'cncl_gan':
                real_content = gt[:, 0:1, :, :]
                optimizer_d.zero_grad()
                fake_out = model(noisy).detach()
                fake_content = fake_out[:, 0:1, :, :]
                
                pred_real = net_d(noisy, real_content)
                loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
                
                pred_fake = net_d(noisy, fake_content)
                loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
                
                loss_d = (loss_real + loss_fake) * 0.5
                loss_d.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                gen_out = model(noisy)
                gen_content = gen_out[:, 0:1, :, :]
                pred_fake = net_d(noisy, gen_content)
                loss_gan = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
                loss_pixel = F.l1_loss(gen_out, gt)
                loss_g = loss_gan + cfg.get('lambda_pixel', 100) * loss_pixel
                loss_g.backward()
                optimizer_g.step()
                running_loss += loss_g.item()

            elif loss_type == 'compound_custom':
                optimizer_g.zero_grad()
                with autocast(DEVICE.type, enabled=amp_enabled):
                    out = model(noisy)
                    out_clamped = out.clamp(0, 1)
                    l_huber = criterion_huber(out_clamped, gt)
                    l_ssim  = criterion_ssim(out_clamped, gt)
                    l_grad  = criterion_grad(out_clamped, gt)

                    if hasattr(args, 'loss_override') and args.loss_override == 'huber':
                        loss = l_huber
                    elif hasattr(args, 'loss_override') and args.loss_override == 'ssim':
                        loss = l_huber + l_ssim
                    elif hasattr(args, 'loss_override') and args.loss_override == 'gradient':
                        loss = l_huber + l_grad
                    elif hasattr(args, 'loss_override') and args.loss_override == 'ssim_only':
                        loss = 1.0 * l_ssim
                    elif hasattr(args, 'loss_override') and args.loss_override == 'grad_only':
                        loss = 1.0 * l_grad
                    elif hasattr(args, 'loss_override') and args.loss_override == 'ssim_grad':
                        loss = 0.5 * l_ssim + 0.1 * l_grad
                    else:
                        loss = args.huber_weight * l_huber + args.ssim_weight * l_ssim + args.grad_weight * l_grad
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer_g)
                scaler.update()
                running_loss += loss.item()

            elif loss_type == 'mse_ssim':
                optimizer_g.zero_grad()
                with autocast(DEVICE.type, enabled=amp_enabled):
                    out = model(noisy)
                    out_clamped = out.clamp(0, 1)
                    loss = criterion_mse(out_clamped, gt) + criterion_ssim(out_clamped, gt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer_g)
                scaler.update()
                running_loss += loss.item()

            else:
                optimizer_g.zero_grad()
                with autocast(DEVICE.type, enabled=amp_enabled):
                    out = model(noisy)
                    if loss_type == 'l1': loss = F.l1_loss(out, gt)
                    else: loss = criterion_mse(out, gt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer_g)
                scaler.update()
                running_loss += loss.item()

        model.eval()
        val_psnrs, val_ssims, val_rmses = [], [], []
        with torch.no_grad(), autocast(DEVICE.type, enabled=amp_enabled):
            for v_noisy, v_gt in val_l:
                v_noisy, v_gt = v_noisy.to(DEVICE), v_gt.to(DEVICE)
                if args.model in ['ctformer', 'uformer']:
                    out = overlapped_inference(model, v_noisy, patch_size=64, margin=16 if args.model=='ctformer' else 32)
                elif args.model == 'cncl':
                    out_full = model(v_noisy).clamp(0, 1)
                    out = out_full[:, 0:1, :, :]
                    v_gt = v_gt[:, 0:1, :, :]
                else:
                    out = model(v_noisy).clamp(0, 1)
                p, s, r, _ = compute_metrics(v_gt, out.float(), v_noisy)
                val_psnrs.extend(p); val_ssims.extend(s); val_rmses.extend(r)
                last_pair = (v_noisy, out, v_gt)

        m_p, s_p = np.mean(val_psnrs), np.std(val_psnrs)
        m_s, s_s = np.mean(val_ssims), np.std(val_ssims)
        m_r, s_r = np.mean(val_rmses), np.std(val_rmses)
        print(f"{fold_prefix}Epoch [{epoch:3d}/{epochs}] Loss: {running_loss/len(train_l):.6f} | "
              f"Val PSNR: {m_p:.4f}+/-{s_p:.4f} | Val SSIM: {m_s:.4f} | "
              f"Val RMSE: {m_r:.4f} | ({time.time()-t0:.1f}s)")
        
        if m_p > best_psnr:
            best_psnr = m_p
            torch.save(model.state_dict(), best_ckpt)
            # Only save the image dump if we are NOT in K-fold, or if we are in fold 1 to avoid overwrite spam
            if fold is None or fold == 1:
                save_best_sample(args.model, *last_pair, epoch)
        
        if fold is None:
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, running_loss/len(train_l), m_p, m_s, m_r, time.time()-t0])
        scheduler_g.step()

    # --- K-Fold Evaluation (After Training Completes) ---
    if fold is not None and test_l is not None:
        print(f"\n{fold_prefix}Evaluatiing Best Checkpoint on Strict Blind Testing Fold...")
        model.load_state_dict(torch.load(best_ckpt))
        model.eval()
        t_psnrs, t_ssims, t_rmses = [], [], []
        with torch.no_grad(), autocast(DEVICE.type, enabled=amp_enabled):
            for t_noisy, t_gt in test_l:
                t_noisy, t_gt = t_noisy.to(DEVICE), t_gt.to(DEVICE)
                if args.model in ['ctformer', 'uformer']:
                    out = overlapped_inference(model, t_noisy, patch_size=64, margin=16 if args.model=='ctformer' else 32)
                elif args.model == 'cncl':
                    out = model(t_noisy).clamp(0, 1)[:, 0:1, :, :]
                    t_gt = t_gt[:, 0:1, :, :]
                else:
                    out = model(t_noisy).clamp(0, 1)
                p, s, r, _ = compute_metrics(t_gt, out.float(), t_noisy)
                t_psnrs.extend(p); t_ssims.extend(s); t_rmses.extend(r)
        
        return np.mean(t_psnrs), np.mean(t_ssims), np.mean(t_rmses)
    
    return best_psnr, 0, 0



def main(args):
    # Resolve batch size alias before printing configuration
    if args.batch is not None:
        args.batch_size = args.batch
    # Set random seed (must happen before any randomness)
    set_seed(args.seed)
    # === Training Configuration ===
    print("\n=== Training Configuration ===")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss override: {args.loss_override}")
    print(f"Seed: {args.seed}")
    print(f"Suffix: {args.suffix if args.suffix else '(none)'}")
    print(f"Start epoch: {args.start_epoch}")
    cfg = MODEL_CONFIGS[args.model]
    patch_size = args.patch_size if args.patch_size else cfg['patch_size']
    batch_size = args.batch_size if args.batch_size else cfg['batch_size']
    epochs     = args.epochs     if args.epochs     else cfg['epochs']
    lr         = args.lr         if args.lr         else cfg['lr']
    loss_type  = cfg['loss']
    milestones = cfg.get('lr_milestones', [])
    return_noise = (args.model == 'cncl')
    
    # K-FOLD CROSS VALIDATION STRATEGY
    if hasattr(args, 'kfold') and args.kfold > 1:
        print(f"\n================= {args.kfold}-FOLD CROSS VALIDATION: {args.model.upper()} =================")
        from dataset import get_global_pairs, get_kfold_dataloaders
        from sklearn.model_selection import KFold
        
        all_pairs = get_global_pairs(BASE_DIR)
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
        folds = list(kf.split(all_pairs))
        
        fold_results = []
        for fold_idx in range(getattr(args, 'start_fold', 1) - 1, args.kfold):
            print(f"\n--- Starting Fold {fold_idx + 1}/{args.kfold} ---")
            suffix_str = f"_fold_{fold_idx+1}"
            if args.suffix: suffix_str += f"_{args.suffix}"
            
            # Tri-Split Implementation
            test_idx = folds[fold_idx][1]
            val_fold_idx = (fold_idx + 1) % args.kfold
            val_idx = folds[val_fold_idx][1]
            train_idx = np.setdiff1d(folds[fold_idx][0], val_idx)
            
            p_train = [all_pairs[i] for i in train_idx]
            p_val   = [all_pairs[i] for i in val_idx]
            p_test  = [all_pairs[i] for i in test_idx]
            
            train_l, val_l, test_l = get_kfold_dataloaders(
                p_train, p_val, p_test, 
                patch_size=patch_size, batch_size=batch_size, return_noise=return_noise
            )
            
            current_start_epoch = args.start_epoch if fold_idx == (getattr(args, 'start_fold', 1) - 1) else 1
            
            # Execute disjoint train
            test_psnr, test_ssim, test_rmse = run_training_loop(
                args, cfg, train_l, val_l, test_l, 
                patch_size, batch_size, epochs, lr, loss_type, milestones, suffix_str, fold=fold_idx+1,
                start_epoch_override=current_start_epoch
            )
            
            fold_results.append((test_psnr, test_ssim, test_rmse))
            print(f">>> Fold {fold_idx+1} Strict Testing Score -> PSNR: {test_psnr:.4f} | SSIM: {test_ssim:.4f} | RMSE: {test_rmse:.4f}\n")
            
        # Final Summary
        print(f"\n================= K-FOLD METRICS ({args.model.upper()}) =================")
        psnrs = [r[0] for r in fold_results]
        ssims = [r[1] for r in fold_results]
        rmses = [r[2] for r in fold_results]
        
        print(f"Final Averaged PSNR: {np.mean(psnrs):.4f} +/- {np.std(psnrs):.4f}")
        print(f"Final Averaged SSIM: {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}")
        print(f"Final Averaged RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
        print("================================================================")
        
    elif args.piglet:
        # PIGLET DATASET TRAINING PIPELINE (single-subject volume, 80/10/10 contiguous split by InstanceNumber)
        print(f"\n==================== TRAINING ON PIGLET: {args.model.upper()} ====================")
        from dataset import get_piglet_pairs, get_kfold_dataloaders
        piglet_dir = os.path.join(BASE_DIR, "piglet-dataset")
        print("   Scanning Piglet dataset (contiguous 80/10/10 split by InstanceNumber)...")
        pairs_train, pairs_val, pairs_test = get_piglet_pairs(piglet_dir)
        print(f"   Piglet split -> train: {len(pairs_train)}, val: {len(pairs_val)}, test: {len(pairs_test)}")
        train_l, val_l, test_l = get_kfold_dataloaders(
            pairs_train, pairs_val, pairs_test,
            patch_size=patch_size, batch_size=batch_size, return_noise=return_noise
        )
        suffix_str = "_piglet" + (f"_{args.suffix}" if args.suffix else "")
        run_training_loop(args, cfg, train_l, val_l, test_l, patch_size, batch_size, epochs, lr, loss_type, milestones, suffix_str, fold=None)

    elif args.mayo:
        # MAYO GRAND CHALLENGE TRAINING PIPELINE (9 patients train/val, L506 held out for testing)
        print(f"\n==================== TRAINING ON MAYO: {args.model.upper()} ====================")
        from dataset import get_mayo_pairs, get_mayo_test_pairs, get_kfold_dataloaders
        mayo_dir = os.path.join(BASE_DIR, "Mayo-folder")
        print("   Scanning Mayo-folder (this may take a moment)...")
        pairs_train, pairs_val = get_mayo_pairs(mayo_dir, val_patient='L310', val_count=211)
        pairs_test = get_mayo_test_pairs(mayo_dir)
        print(f"   Mayo split -> train: {len(pairs_train)}, val: {len(pairs_val)} (L310), test: {len(pairs_test)} (L506)")
        train_l, val_l, test_l = get_kfold_dataloaders(
            pairs_train, pairs_val, pairs_test,
            patch_size=patch_size, batch_size=batch_size, return_noise=return_noise
        )
        suffix_str = "_mayo" + (f"_{args.suffix}" if args.suffix else "")
        run_training_loop(args, cfg, train_l, val_l, test_l, patch_size, batch_size, epochs, lr, loss_type, milestones, suffix_str, fold=None)

    else:
        # STANDARD TRAINING PIPELINE
        print(f"\n==================== TRAINING: {args.model.upper()} ====================")
        from dataset import get_dataloaders
        print("   Scanning Dataset folders (this may take a moment)...")
        train_l, val_l, test_l = get_dataloaders(BASE_DIR, patch_size=patch_size, batch_size=batch_size, return_noise=return_noise, num_workers=0)
        print(f"   Done. Found {len(train_l.dataset)} training samples.")
        suffix_str = f"_{args.suffix}" if args.suffix else ""
        run_training_loop(args, cfg, train_l, val_l, None, patch_size, batch_size, epochs, lr, loss_type, milestones, suffix_str, fold=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['dncnn','red_cnn','wgan_vgg','edcnn','ctformer','unet', 'uformer', 'proposed_model', 'proposed_model_no_ceelpa', 'cncl', 'leda', 'improved_red_cnn_csa'])
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--batch', type=int, help='Alias for batch_size (optional)')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--kfold', type=int, default=0, help='Number of folds for Cross Validation (0 to disable)')
    parser.add_argument('--mayo', action='store_true', help='Train on the Mayo Grand Challenge dataset (9 patients) instead of COVID-19; L506 held out for testing, L310 (211 slices) held out for validation')
    parser.add_argument('--piglet', action='store_true', help='Train on the Piglet dataset (single-subject, 850 slices); contiguous 80/10/10 train/val/test split by InstanceNumber')
    parser.add_argument('--suffix', type=str, default='', help='Optional suffix to append to logs and checkpoints')
    parser.add_argument('--disable_ceelpa', action='store_true', help='Disable CEELPA module for ablation')
    parser.add_argument('--disable_rca', action='store_true', help='Disable RCA blocks for ablation')
    parser.add_argument('--disable_msfe', action='store_true', help='Disable MSFE blocks for ablation')
    parser.add_argument('--huber_weight', type=float, default=1.0, help='Weight (alpha) on the Huber term in compound_custom loss')
    parser.add_argument('--ssim_weight', type=float, default=0.5, help='Weight (beta) on the SSIM term in compound_custom loss')
    parser.add_argument('--grad_weight', type=float, default=0.1, help='Weight (phi) on the Gradient term in compound_custom loss')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 weight decay on the Adam optimizer (default 0.0, matches prior behavior)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--ckpt_path', type=str, default='', help='Path to checkpoint to load before training')
    parser.add_argument('--loss_override', type=str, choices=['huber', 'ssim', 'gradient', 'ssim_grad', 'ssim_only', 'grad_only'], help='Overrides the default compound loss for ablation')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=None, help='LR warmup length (default 5)')
    parser.add_argument('--start_fold', type=int, default=1, help='Which fold to start/resume from')
    main(parser.parse_args())
