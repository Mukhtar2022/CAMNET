import subprocess
import os

BASE_DIR = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"
EVAL_SCRIPT = os.path.join(BASE_DIR, "evaluate.py")

# Models to re-run
deep_learning_models = [
    ("dncnn", ""),
    ("red_cnn", ""),
    ("edcnn", ""),
    ("leda", ""),
    ("uformer", ""),
    ("ctformer", ""),
    ("proposed_model_v2", "add_rca_msfe_v2") # Champion Version
]

print("="*60)
print("STARTING FINAL BENCHMARKING OF ALL MODELS")
print("="*60)

for model, suffix in deep_learning_models:
    cmd = ["python", EVAL_SCRIPT, "--model", model]
    if suffix:
        cmd.extend(["--suffix", suffix])
    
    print(f"\n>>> Running evaluation for: {model.upper()} {suffix}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating {model}: {e}")

print("\n" + "="*60)
print("BENCHMARKING COMPLETE")
print("="*60)
