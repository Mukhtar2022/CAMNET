import subprocess
import os

ROOT = r"C:/Users/HP/Documents/COVID-19-EXPIREMENT"
GT_DIR = os.path.join(ROOT, "piglet-dataset", "subset_125", "FD_850")
NOISY_DIR = os.path.join(ROOT, "piglet-dataset", "subset_125", "LD25_850")

MODELS = [
    "ldct_input",  # baseline handled separately
    "nlm",
    "dncnn",
    "edcnn",
    "red_cnn",
    "leda",
    "uformer",
    "ctformer",
    "proposed_model",
    "bm3d",
]

def run():
    for name in MODELS:
        if name == "ldct_input":
            # baseline: compute metrics where output == noisy
            subprocess.run([
                "python",
                "benchmark_results/generate_ldct_baseline.py",
                "--gt_dir",
                GT_DIR,
                "--noisy_dir",
                NOISY_DIR,
            ], cwd=ROOT, check=True)
            continue
            
        suffix = "epo50" if name == "proposed_model" else None
        
        args = [
            "python",
            "evaluate.py",
            "--model",
            name,
            "--gt_dir",
            GT_DIR,
            "--noisy_dir",
            NOISY_DIR,
        ]
        if suffix:
            args += ["--suffix", suffix]
        
        print(f"\n=== Running {name.upper()} ===")
        subprocess.run(args, cwd=ROOT, check=True)

if __name__ == "__main__":
    run()
