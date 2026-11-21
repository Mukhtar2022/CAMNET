"""Reproduce the Edge PSNR, EPI and ST-PSNR results of the CAMNet paper.

Experiments
  covid            COVID-19 comparison table (11 methods, 128 test slices)
  mayo             Mayo Grand Challenge table, models trained on Mayo (11 methods, 211-slice L506 test set)
  piglet           Piglet table, models trained on Piglet (11 methods, 85 test slices)
  loss_ablation    Loss-function ablation table (8 configurations, COVID-19)
  ceepa_ablation   Full CAMNet vs w/o CEEPA (COVID-19; ST-PSNR sentence in the Ablation Study)

Usage (from this folder):
  python evaluate_region_metrics.py --experiment covid
  python evaluate_region_metrics.py --experiment mayo piglet
  python evaluate_region_metrics.py --experiment covid --methods CAMNet RED-CNN     # subset

Each method's PSNR is first checked against the value in the paper (tolerance 1e-3 dB).
Outputs, written to --out_dir (default ./region_metrics_results):
  <experiment>_per_slice.csv   one row per method and test slice
  <experiment>_summary.csv     mean and std (ddof=1) per method and metric
  <experiment>_wilcoxon.csv    two-sided Wilcoxon signed-rank test of CAMNet (or SGH / full model) against
                               every other method, Holm-Bonferroni adjusted over all comparisons x metrics
                               of the experiment

Data paths follow the rest of this pipeline (BASE_DIR in evaluate.py); override with --data_root.
Intensities use the same windowing as dataset.py, which produced the published tables.
"""
import os, sys, csv, argparse
import numpy as np, torch
from scipy.stats import wilcoxon
from skimage.restoration import denoise_nl_means, estimate_sigma

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as EV                                   # DEVICE, overlapped_inference, bm3d
from dataset import CTDenoisingDataset, get_mayo_test_pairs, get_piglet_pairs
import region_metrics as RM

KIND = {'DnCNN': 'dncnn', 'EDCNN': 'edcnn', 'RED-CNN': 'red_cnn', 'Improved RED-CNN-CSA': 'improved_red_cnn_csa',
        'LEDA': 'leda', 'U-Former': 'uformer', 'CT-Former': 'ctformer', 'CAMNet': 'proposed_model'}

# (method, model kind or classical variant, checkpoint stem in checkpoints/, PSNR in the paper)
EXPERIMENTS = {
 'covid': [
    ('Noisy Input', 'noisy', None, 32.8813), ('NLM', 'nlm_a', None, 34.5741), ('BM3D', 'bm3d_a', None, 27.7491),
    ('DnCNN', 'dncnn', 'dncnn', 33.1253), ('EDCNN', 'edcnn', 'edcnn', 35.0820),
    ('RED-CNN', 'red_cnn', 'red_cnn_final_journal_run', 35.5880),
    ('Improved RED-CNN-CSA', 'improved_red_cnn_csa', 'improved_red_cnn_csa', 34.0794),
    ('CT-Former', 'ctformer', 'ctformer', 33.6095), ('LEDA', 'leda', 'leda', 35.7281),
    ('U-Former', 'uformer', 'uformer', 35.9890), ('CAMNet', 'proposed_model', 'proposed_model_epo50', 36.1676)],
 'mayo': [
    ('Noisy Input', 'noisy', None, 29.7404), ('NLM', 'nlm_b', None, 29.9396), ('BM3D', 'bm3d_b', None, 30.0766),
    ('DnCNN', 'dncnn', 'dncnn_mayo', 32.5777), ('EDCNN', 'edcnn', 'edcnn_mayo', 32.8079),
    ('RED-CNN', 'red_cnn', 'red_cnn_mayo', 34.1926),
    ('Improved RED-CNN-CSA', 'improved_red_cnn_csa', 'improved_red_cnn_csa_mayo', 34.0253),
    ('LEDA', 'leda', 'leda_mayo', 34.2183), ('U-Former', 'uformer', 'uformer_mayo', 33.8891),
    ('CT-Former', 'ctformer', 'ctformer_mayo', 31.9778), ('CAMNet', 'proposed_model', 'proposed_model_mayo', 34.3062)],
 'piglet': [
    ('Noisy Input', 'noisy', None, 30.6480), ('NLM', 'nlm_a', None, 30.9434), ('BM3D', 'bm3d_a', None, 30.2468),
    ('DnCNN', 'dncnn', 'dncnn_piglet', 30.8500), ('EDCNN', 'edcnn', 'edcnn_piglet', 31.3286),
    ('RED-CNN', 'red_cnn', 'red_cnn_piglet', 31.5381),
    ('Improved RED-CNN-CSA', 'improved_red_cnn_csa', 'improved_red_cnn_csa_piglet', 31.4254),
    ('LEDA', 'leda', 'leda_piglet', 31.2803), ('U-Former', 'uformer', 'uformer_piglet', 31.1489),
    ('CT-Former', 'ctformer', 'ctformer_piglet', 31.2857),
    ('CAMNet', 'proposed_model', 'proposed_model_piglet', 31.5771)],
 'loss_ablation': [
    ('RED-CNN (baseline)', 'red_cnn', 'red_cnn_final_journal_run', 35.5880),
    ('Huber only', 'proposed_model', 'proposed_model_abl_huber', 36.1545),
    ('SSIM only', 'proposed_model', 'proposed_model_abl_ssim', 35.7038),
    ('Gradient only', 'proposed_model', 'proposed_model_abl_gradient', 35.7584),
    ('Huber + SSIM', 'proposed_model', 'proposed_model_abl_huber_ssim', 35.7462),
    ('Huber + Gradient', 'proposed_model', 'proposed_model_abl_huber_gradient', 35.8544),
    ('SSIM + Gradient', 'proposed_model', 'proposed_model_abl_ssim_gradient', 35.7442),
    ('SGH', 'proposed_model', 'proposed_model_epo50', 36.1676)],
 'ceepa_ablation': [
    ('w/o CEEPA', 'proposed_model_no_ceepa', 'proposed_model_no_ceelpa_ablation_run', 35.8469),
    ('Full CAMNet', 'proposed_model', 'proposed_model_epo50', 36.1676)],
}
REFERENCE = {'covid': 'CAMNet', 'mayo': 'CAMNet', 'piglet': 'CAMNet', 'loss_ablation': 'SGH', 'ceepa_ablation': 'Full CAMNet'}


def build(kind):
    if kind == 'dncnn':
        from models.dncnn import DnCNN; return DnCNN()
    if kind == 'red_cnn':
        from models.red_cnn import RED_CNN; return RED_CNN()
    if kind == 'edcnn':
        from models.edcnn import EDCNN; return EDCNN()
    if kind == 'ctformer':
        from models.ctformer import CTFormer; return CTFormer()
    if kind == 'uformer':
        from models.uformer import Uformer
        return Uformer(img_size=64, in_chans=1, dd_in=1, embed_dim=32, depths=[2] * 9, win_size=8)
    if kind == 'leda':
        from models.leda import LEDA; return LEDA()
    if kind == 'improved_red_cnn_csa':
        from models.improved_red_cnn_csa import ImprovedREDCNN_CSA; return ImprovedREDCNN_CSA()
    if kind == 'proposed_model':
        from models.proposed_red_cnn import Proposed_RED_CNN; return Proposed_RED_CNN()
    if kind == 'proposed_model_no_ceepa':
        from models.proposed_red_cnn_no_ceelpa import Proposed_RED_CNN_No_CEELPA; return Proposed_RED_CNN_No_CEELPA()
    raise ValueError(kind)


def classical(kind, nz):
    """NLM / BM3D settings used for each table (variant a: COVID-19 and Piglet, variant b: Mayo scripts)."""
    if kind == 'nlm_a':
        se = np.mean(estimate_sigma(nz))
        return denoise_nl_means(nz, h=1.15 * se, fast_mode=True, patch_size=5, patch_distance=6)
    if kind == 'bm3d_a':
        return EV.bm3d(nz, sigma_psd=np.mean(np.abs(nz - nz.mean())))
    if kind == 'nlm_b':
        se = float(np.mean(estimate_sigma(nz, channel_axis=None)))
        return np.clip(denoise_nl_means(nz, h=0.8 * se, fast_mode=True, patch_size=5, patch_distance=11), 0, 1)
    if kind == 'bm3d_b':
        return np.clip(EV.bm3d(nz, sigma_psd=float(np.mean(estimate_sigma(nz, channel_axis=None)))), 0, 1)
    raise ValueError(kind)


def test_set(exp, root):
    if exp == 'mayo':
        pairs = get_mayo_test_pairs(os.path.join(root, 'Mayo-folder'))
    elif exp == 'piglet':
        pairs = get_piglet_pairs(os.path.join(root, 'piglet-dataset'))[2]
    else:
        return CTDenoisingDataset(root, split='Testing', patch_size=0, image_size=256)
    return CTDenoisingDataset(root, split='Testing', patch_size=0, augment=False, image_size=256, pairs_override=pairs)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def run(exp, args):
    ds = test_set(exp, args.data_root)
    data = [(ds[i][0], ds[i][1][0].numpy().astype(np.float32)) for i in range(len(ds))]
    print(f'\n######## {exp}  ({len(data)} test slices)', flush=True)
    R = {}
    for name, kind, ck, paper in EXPERIMENTS[exp]:
        if args.methods and name not in args.methods and name != REFERENCE[exp]:
            continue
        model = None
        if ck:
            path = os.path.join(args.ckpt_dir, ck + '_best.pth')
            model = build(kind)
            model.load_state_dict(torch.load(path, map_location=EV.DEVICE), strict=True)
            model.to(EV.DEVICE).eval()
        rows = []
        with torch.no_grad():
            for noisy, gt in data:
                nz = noisy[0].numpy().astype(np.float32); x = noisy.unsqueeze(0).to(EV.DEVICE)
                if kind == 'noisy':
                    out = nz
                elif model is None:
                    out = classical(kind, nz)
                elif kind == 'ctformer':
                    out = EV.overlapped_inference(model, x, patch_size=64, margin=16)[0, 0].cpu().numpy()
                elif kind == 'uformer':
                    out = EV.overlapped_inference(model, x, patch_size=64, margin=32)[0, 0].cpu().numpy()
                else:
                    out = model(x).clamp(0, 1)[0, 0].cpu().numpy()
                rows.append(RM.all_metrics(gt, np.asarray(out, np.float32)))
        if model is not None:
            del model; torch.cuda.empty_cache()
        a = R[name] = np.array(rows)
        ok = 'PSNR matches paper' if abs(a[:, 0].mean() - paper) < 1e-3 else f'PSNR MISMATCH (paper {paper})'
        print(f'  {name:<22}' + '  '.join(f'{k} {a[:, j].mean():.4f} ± {a[:, j].std(ddof=1):.4f}'
                                       for j, k in enumerate(RM.METRICS)) + f'   [{ok}]', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f'{exp}_per_slice.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'slice'] + RM.METRICS)
        for k, a in R.items():
            w.writerows([k, i] + [f'{v:.6f}' for v in r] for i, r in enumerate(a))
    with open(os.path.join(args.out_dir, f'{exp}_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method'] + [f'{k}_{s}' for k in RM.METRICS for s in ('mean', 'std')])
        for k, a in R.items():
            w.writerow([k] + [f'{v:.4f}' for j in range(len(RM.METRICS)) for v in (a[:, j].mean(), a[:, j].std(ddof=1))])

    ref = REFERENCE[exp]
    if ref not in R or len(R) < 2:
        return
    tests = []
    for k, a in R.items():
        if k == ref:
            continue
        for j, met in enumerate(RM.METRICS):
            d = R[ref][:, j] - a[:, j]
            tests.append([k, met, d.mean(), int(np.sum(d > 0 if RM.HIGHER_IS_BETTER[met] else d < 0)),
                          wilcoxon(R[ref][:, j], a[:, j]).pvalue])
    adj = holm([t[-1] for t in tests])
    with open(os.path.join(args.out_dir, f'{exp}_wilcoxon.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['method', 'metric', f'mean_diff_{ref}_minus_method', f'slices_{ref}_better', 'n', 'p_wilcoxon', 'p_holm'])
        for t, ph in zip(tests, adj):
            w.writerow(t[:4] + [len(data), f'{t[4]:.3g}', f'{ph:.3g}'])
    print(f'\n  Rank of {ref} (1 = best) and Holm-adjusted Wilcoxon p against the best other method:')
    for j, met in enumerate(RM.METRICS):
        means = {k: a[:, j].mean() for k, a in R.items()}
        order = sorted(means, key=lambda k: -means[k] if RM.HIGHER_IS_BETTER[met] else means[k])
        best = next(k for k in order if k != ref)
        ph = next(p for t, p in zip(tests, adj) if t[0] == best and t[1] == met)
        print(f'  {met:<9} rank {order.index(ref) + 1}/{len(order)}   best other {best} ({means[best]:.4f})   p_holm={ph:.2g}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', nargs='+', required=True, choices=list(EXPERIMENTS))
    ap.add_argument('--methods', nargs='*', default=[], help='evaluate only these methods (the reference model is always included)')
    ap.add_argument('--data_root', default=EV.BASE_DIR, help='folder with Testing/, Testing_Noisy/, Mayo-folder/ and piglet-dataset/')
    ap.add_argument('--ckpt_dir', default=os.path.join(HERE, '..', 'checkpoints'))
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'region_metrics_results'))
    args = ap.parse_args()
    for exp in args.experiment:
        run(exp, args)


if __name__ == '__main__':
    main()
