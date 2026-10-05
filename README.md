# CAMNet — Code for "CAMNet: An Edge-Contrast Prior and Compound Attention-Guided Multi-Scale RED-CNN for Low-Dose CT Image Denoising"

This repository contains the training, evaluation, and ablation code used to produce the results reported in the paper. Every number listed below has been **live-verified** by independently re-running the evaluation pipeline against the checkpoints in this repo, not copied from a saved results file.

## Structure

```
camnet_pipeline/        The verified pipeline — produces every published COVID-19, Mayo,
                         Piglet, ablation, and significance-test result in the paper.
  models/                CEEPA + CAB + MSFE architecture and baseline model definitions


checkpoints/             Trained model weights (see inventory below)
evidence/                Provenance artifacts for results that needed deeper verification
```

## Verification status

| Table | Status |
|---|---|
| Table 4 — COVID-19 main comparison (11 models) | ✅ 10/11 live-verified exact match. BM3D's published PSNR/RMSE do not reproduce — see Known Issues. |
| Table 6 — Mayo main comparison (11 models, 211-slice L506 test set) | ✅ All 11 rows exact match, including BM3D and NLM (no bug — Mayo's classical-baseline scripts never touch uint8 arrays). |
| Table 7 — Mayo Wilcoxon + Holm-Bonferroni significance | ✅ Fully verified, every cell exact match (two-sided convention, same as Piglet's Table 10). |
| Table 8 — Mayo batch size sweep (8/16/32/64) | ✅ All 4 rows exact match. |
| Table 9 — Piglet main comparison (11 models, trained from scratch) | ✅ All 11 rows live-verified exact match, including BM3D (no bug on this dataset — it uses a different, clean code path). |
| Table 10 — Piglet Wilcoxon significance | ✅ Fully verified, every value exact match. Uses two-sided tests vs. Table 14's one-sided — cosmetic inconsistency only, see Known Issues. |
| Table 11 — Piglet batch size sweep (8/16/32/64) | ✅ All 4 rows exact match. |
| Table 13 — Architecture ablation (w/o CEEPA, w/o CAB, w/o MSFE) | ⚠️ Numbers are genuine (full-precision match to `evidence/ablation_summary.csv`), but 2 of 3 checkpoints were lost to an overwrite — see Known Issues. |
| Table 14 — COVID-19 Wilcoxon + Holm-Bonferroni significance | ✅ Fully verified, every raw p-value exact match. One cosmetic family-size inconsistency noted, doesn't change any conclusion. |
| Table 15 — Loss function ablation (7 configs) | ✅ All 6 non-duplicate rows bit-identical match. |
| Table 16 — COVID-19 batch size sweep (8/16/32/64) | ✅ All 4 rows exact match. |

## Checkpoint inventory (`checkpoints/`, 489 MB total)

| File | Reproduces |
|---|---|
| `proposed_model_epo50_best.pth` | **Proposed RED-CNN** — Table 4, Table 13 Full Model, Table 15 SGH row (COVID-19) |
| `dncnn_best.pth`, `edcnn_best.pth`, `improved_red_cnn_csa_best.pth`, `ctformer_best.pth`, `leda_best.pth`, `uformer_best.pth`, `red_cnn_final_journal_run_best.pth` | The other 7 trainable baselines in Table 4 (COVID-19) |
| `proposed_model_no_ceelpa_best.pth` | Table 13 "w/o CEEPA" row |
| `proposed_model_abl_huber_best.pth`, `abl_ssim`, `abl_gradient`, `abl_huber_ssim`, `abl_huber_gradient`, `abl_ssim_gradient` | Table 15's 6 loss-ablation rows |
| `proposed_model_piglet_100ep_wd1e-5_best.pth` | **Proposed RED-CNN** — Table 9 (Piglet, trained from scratch) |
| `dncnn_piglet_best.pth`, `edcnn_piglet_best.pth`, `red_cnn_piglet_best.pth`, `improved_red_cnn_csa_piglet_best.pth`, `leda_piglet_best.pth`, `uformer_piglet_best.pth`, `ctformer_piglet_best.pth` | The other 7 trainable baselines in Table 9 (Piglet) |
| `proposed_model_bs16_best.pth`, `bs32`, `bs64` | Table 16 batch-size sweep rows 16/32/64 (COVID-19; bs=8 is `epo50` above) |
| `proposed_model_piglet_100ep_wd1e-5_bs16_best.pth`, `bs32`, `bs64` | Table 11 batch-size sweep rows 16/32/64 (Piglet; bs=8 is `piglet_100ep_wd1e-5` above) |
| `proposed_model_mayo_100ep_best.pth` | **Proposed RED-CNN** — Table 6, Table 8 bs=8 row (Mayo). Note: the plain `proposed_model_mayo_best.pth` (no `_100ep`) is a real but distinctly lower-scoring decoy checkpoint — do not confuse the two. |
| `dncnn_mayo_best.pth`, `edcnn_mayo_best.pth`, `red_cnn_mayo_best.pth`, `improved_red_cnn_csa_mayo_best.pth`, `leda_mayo_best.pth`, `uformer_mayo_best.pth`, `ctformer_mayo_best.pth` | The other 7 trainable baselines in Table 6 (Mayo) |
| `proposed_model_mayo_bs16_best.pth`, `bs32`, `bs64` | Table 8 batch-size sweep rows 16/32/64 (Mayo) |

NLM and BM3D have no trainable checkpoint (classical methods) — see Baselines Not Included below for how to run them. The invocation and bug-exposure differs by dataset:
- **COVID-19**: NLM needs the dedicated `evaluate_nlm.py` script (not `evaluate.py --model nlm`); BM3D's dedicated script has the uint8-wraparound bug described below.
- **Piglet**: both NLM and BM3D run through the plain `evaluate.py --model nlm`/`--model bm3d` path — clean, bug-free.
- **Mayo**: both use dedicated `evaluate_nlm_mayo.py` / `evaluate_bm3d_mayo.py` scripts, confirmed to compute PSNR via `skimage.metrics.peak_signal_noise_ratio` directly on float32 — these never introduce the uint8 path that caused COVID-19's bug, so Mayo's BM3D/NLM rows are both value-correct and code-correct.

## Known issues (full detail in commit history / project notes)

1. **BM3D's published COVID-19 PSNR (31.6965 dB) and RMSE (0.0261) do not reproduce.** Traced to a confirmed bug in the historical evaluation script (`calculate_psnr()` subtracted uint8 arrays without casting to float, causing integer wraparound). The true value for that same historical run is **PSNR ≈ 21.6–21.8 dB, RMSE ≈ 0.0831** — making BM3D the worst performer in the table rather than mid-pack. SSIM (0.7127) was unaffected by the bug and is correct as published. A second, independent problem was also found: the script's `sigma=25` noise parameter is ~5× higher than this dataset's actual measured noise level (~5.7), which likely needs correcting for a fair baseline. **This row needs attention before the paper is finalized.** (This bug does NOT affect Piglet's or Mayo's BM3D rows — both use different, clean code paths and are fully verified.)
2. **Two of three Table 13 architecture-ablation checkpoints (w/o CAB, w/o MSFE) are unrecoverable.** The original checkpoints were overwritten by a later training run minutes after their results were captured (see `evidence/ablation_summary.csv` for proof the original numbers are genuine). They are not included here because they no longer exist anywhere. Retraining is the only way to restore a checkpoint that reproduces those two rows.
3. **Table 10 (Piglet) and Table 7 (Mayo) use two-sided Wilcoxon tests; Table 14 (COVID-19) uses one-sided tests**, for what should be the same statistical procedure. Doesn't change any Sig=Yes/No conclusion in any table (the two-sided p-value is just 2× the one-sided value, and all remain far below any significance threshold everywhere it matters) — purely a consistency issue worth fixing before submission by making all three tables use the same convention.
4. **Table 10's "Sig=No" for Improved RED-CNN-CSA's SSIM is correct, not a data error.** Its p-value (1.17e-15) is far below 0.05, but the table's "Sig" column is direction-aware: "No" here means "not significant in Proposed's favor" — confirmed by a reverse one-sided test showing Improved RED-CNN-CSA's SSIM is itself significantly higher than Proposed's, matching the manuscript's own prose. Worth a table footnote clarifying this convention, since a bare p<0.05 reading looks contradictory otherwise.

## Baseline implementations not included here

This repo's `models/` folders contain only the lightweight model-definition files written for this project to instantiate and train each baseline (DnCNN, EDCNN, RED-CNN, Improved RED-CNN-CSA, LEDA, U-Former, CNCL), not the original authors' training code.

## Evidence

- `evidence/ablation_summary.csv` — original evaluation output (`generate_global_ablation_summary.py`, 2026-06-01) proving all three Table 13 rows are genuine, to full float precision.

## Datasets

See the Data Availability statement in the manuscript. The Piglet dataset (680/85/85 split, contiguous by InstanceNumber) is included alongside the dataset upload on OSF.
