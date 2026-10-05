# ctformer_benchmark.py – benchmark only the CT‑Former model
# ------------------------------------------------------------
# This script loads the CT‑Former class, instantiates it with its default
# configuration (single‑channel input), and reports:
#   • Trainable parameters (M)
#   • FLOPs (G) using thop
#   • Average inference time (ms) over several runs
# The results are saved to a CSV file in the current working directory
# (Kaggle: /kaggle/working when run inside a kernel).
# ------------------------------------------------------------

import os, sys, time, importlib, inspect
import torch
from thop import profile
import pandas as pd

# ------------------------------------------------------------
# Helper: warm‑up then average timing (ms)
# ------------------------------------------------------------
def benchmark_callable(fn, *args, warmup: int = 5, repeats: int = 30):
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats * 1000.0  # ms

# ------------------------------------------------------------
# Locate the project root (folder containing the `models` directory).
# ------------------------------------------------------------
def find_project_root(start_path: str = None):
    if start_path is None:
        start_path = os.path.abspath(os.path.dirname(__file__))
    cur = start_path
    while True:
        if os.path.isdir(os.path.join(cur, "models")):
            return cur
        parent = os.path.abspath(os.path.join(cur, os.pardir))
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("Could not locate a folder containing a 'models' sub‑directory.")

project_root = r"C:/Users/HP/Documents/COVID-19-EXPIREMENT"
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"Project root added to PYTHONPATH -> {project_root}")

# ------------------------------------------------------------
# Import the CT‑Former class and instantiate it
# ------------------------------------------------------------
module = importlib.import_module("models.ctformer.CTformer")
CTFormer = getattr(module, "CTFormer")

# Use the default configuration – expects a single-channel image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CTFormer().to(device)
model.eval()

# ------------------------------------------------------------
# Compute metrics
# ------------------------------------------------------------

# Parameter count (in millions)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
params_m = round(param_count / 1e6, 4)

# FLOPs -- dummy input: CT-Former's T2T module is hardcoded for 64x64 (num_patches=529)
_dummy = torch.randn(1, 1, 64, 64, device=device)
macs, _ = profile(model, inputs=(_dummy,), verbose=False)
flops = macs * 2  # 1 MAC = 2 FLOPs
flops_g = round(flops / 1e9, 4)

# Inference time (average ms)
time_ms = round(benchmark_callable(model, _dummy), 2)

print(f"CT-Former -> Params: {params_m} M, FLOPs: {flops_g} G, Time: {time_ms} ms")

# ------------------------------------------------------------
# Save to CSV (Kaggle working directory or current folder)
# ------------------------------------------------------------
csv_path = os.path.join(os.getenv("KAGGLE_WORKING_DIR", "."), "ctformer_complexity.csv")
pd.DataFrame([
    {
        "Model": "CT-Former",
        "Parameters (M)": params_m,
        "FLOPs (G)": flops_g,
        "Inference Time (ms)": time_ms,
    }
]).to_csv(csv_path, index=False)
print(f"\n[INFO] Summary saved to: {csv_path}")

