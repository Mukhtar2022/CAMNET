import subprocess
import os

BASE_DIR = r"C:\\Users\\HP\\Documents\\COVID-19-EXPIREMENT"

# Models to evaluate (as per user request)
models = [
    "nlm",
    "bm3d",
    "dncnn",
    "edcnn",
    "red_cnn",
    "leda",
    "uformer",
    "ctformer",
    "proposed_model_epo50_best"  # custom checkpoint name
    # Note: "ldct" (noisy baseline) is handled directly in ROI generation; no checkpoint required.
]

# Mapping for special checkpoint names (evaluate.py expects model name, suffix optional)
# For the proposed model, we will use the checkpoint filename directly via suffix argument.
# The checkpoint file is "proposed_model_epo50_best.pth" in the checkpoints folder.
# We'll pass suffix="epo50_best" and model="proposed_model" (the script maps to Proposed_RED_CNN).

def run_evaluation(model_name):
    # Piglet dataset paths (noisy and ground truth)
    # Determine if a subset directory exists for cross‑dataset evaluation
    subset_path = os.path.join(BASE_DIR, "piglet-dataset", "subset_125")
    if os.path.isdir(subset_path):
        gt_dir = os.path.join(subset_path, "LD25_850")
        noisy_dir = os.path.join(subset_path, "FD_850")
    else:
        gt_dir = os.path.join(BASE_DIR, "piglet-dataset", "LD25_850")
        noisy_dir = os.path.join(BASE_DIR, "piglet-dataset", "FD_850")
    if model_name == "proposed_model_epo50_best":
        cmd = ["python", "evaluate.py", "--model", "proposed_model", "--suffix", "epo50_best",
               "--gt_dir", gt_dir, "--noisy_dir", noisy_dir]
    else:
        cmd = ["python", "evaluate.py", "--model", model_name,
               "--gt_dir", gt_dir, "--noisy_dir", noisy_dir]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=BASE_DIR, check=True)

if __name__ == "__main__":
    for m in models:
        try:
            run_evaluation(m)
        except subprocess.CalledProcessError as e:
            print(f"Error running model {m}: {e}")
    # After all evaluations, combine metrics and generate ROI visualizations
    subprocess.run(["python", "combine_metrics_and_roi.py"], cwd=BASE_DIR, check=True)
    print("All done! Combined metrics and ROI images are in the piglet-dataset folder.")
