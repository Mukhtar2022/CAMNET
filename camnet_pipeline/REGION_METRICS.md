# Region-based metrics: Edge PSNR, EPI and ST-PSNR

These files reproduce the three region-based metrics reported in the CAMNet paper, alongside
PSNR, SSIM and RMSE, for the COVID-19, Mayo Grand Challenge and Piglet comparison tables and the
COVID-19 loss-function and CEEPA ablations.

| File | Purpose |
|---|---|
| `region_metrics.py` | Metric definitions |
| `evaluate_region_metrics.py` | Runs every method of a table, checks its PSNR against the paper, writes per-slice values, summaries and Wilcoxon tests |

## Definitions

All images are in [0, 1] (data range 1.0).

- **Edge PSNR**: PSNR on the 10% of pixels with the strongest ground-truth Sobel gradient magnitude.
- **EPI** (edge preservation index): correlation between the Laplacian-filtered denoised and reference images.
- **ST-PSNR** (soft-tissue PSNR): PSNR on body pixels (reference > 0.05) within one standard deviation of
  the mean body intensity of the reference slice.

Means are over test slices. Standard deviations use ddof = 1.

## Usage

Run from `camnet_pipeline/`. Data paths follow `BASE_DIR` in `evaluate.py`. Use `--data_root` to point
to a folder that holds `Testing/`, `Testing_Noisy/`, `Mayo-folder/` and `piglet-dataset/`.

```
python evaluate_region_metrics.py --experiment covid
python evaluate_region_metrics.py --experiment mayo
python evaluate_region_metrics.py --experiment piglet
python evaluate_region_metrics.py --experiment loss_ablation
python evaluate_region_metrics.py --experiment ceepa_ablation
```

`--methods` evaluates a subset (the reference model is always included). Results are written to
`region_metrics_results/`:

- `<experiment>_per_slice.csv`: every metric for every method and test slice
- `<experiment>_summary.csv`: mean and standard deviation
- `<experiment>_wilcoxon.csv`: two-sided Wilcoxon signed-rank test of CAMNet (SGH for the loss ablation,
  the full model for the CEEPA ablation) against each other method, Holm-Bonferroni adjusted over all
  comparisons and metrics of the experiment

NLM and BM3D need the `bm3d` and `scikit-image` packages and run on the CPU, so the `covid`, `mayo` and
`piglet` experiments take longest.

## Checkpoints used (`../checkpoints/`)

| Experiment | CAMNet | Baselines |
|---|---|---|
| covid | `proposed_model_epo50_best.pth` | `dncnn`, `edcnn`, `red_cnn_final_journal_run`, `improved_red_cnn_csa`, `ctformer`, `leda`, `uformer` (`*_best.pth`) |
| mayo | `proposed_model_mayo_100ep_best.pth` | the same baselines with the `_mayo` suffix (`red_cnn_mayo`, ...) |
| piglet | `proposed_model_piglet_100ep_wd1e-5_best.pth` | the same baselines with the `_piglet` suffix |
| loss_ablation | `proposed_model_epo50_best.pth` (SGH) | `proposed_model_abl_{huber,ssim,gradient,huber_ssim,huber_gradient,ssim_gradient}_best.pth`, `red_cnn_final_journal_run_best.pth` |
| ceepa_ablation | `proposed_model_epo50_best.pth` | `proposed_model_no_ceelpa_ablation_run_best.pth` (w/o CEEPA) |

NLM and BM3D settings: COVID-19 and Piglet use h = 1.15 sigma (patch 5, distance 6) and BM3D with
sigma_psd = mean absolute deviation. Mayo uses h = 0.8 sigma (patch 5, distance 11) and BM3D with the
wavelet noise estimate, as in `evaluate_nlm_mayo.py` and `evaluate_bm3d_mayo.py`.
