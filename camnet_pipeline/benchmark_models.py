# ------------------------------------------------------------
# 1️⃣ Install the FLOPs profiler (run once)
# ------------------------------------------------------------
# pip install -q thop   # Install the FLOPs profiler (run once) (uncomment if needed)

# ------------------------------------------------------------
# 2️⃣ Imports & utilities
# ------------------------------------------------------------
import os, sys, time, importlib, inspect
import numpy as np, pandas as pd
import torch
from thop import profile

# ------------------------------------------------------------
# 3️⃣ Locate the project root (where the `models/` folder lives)
# ------------------------------------------------------------
def find_project_root(start_path: str = None):
    """Return the directory that contains a `models` sub‑directory.
    If `start_path` is omitted we start from the directory of this script.
    The function walks upward until it finds the `models` folder.
    """
    if start_path is None:
        start_path = os.path.abspath(os.path.dirname(__file__))
    # Walk upward from start_path
    current = start_path
    while True:
        if os.path.isdir(os.path.join(current, "models")):
            return current
        parent = os.path.abspath(os.path.join(current, os.pardir))
        if parent == current:
            break
        current = parent
    raise FileNotFoundError("Could not locate a folder containing a `models` sub‑directory.")
project_root = find_project_root()
if project_root not in sys.path:
    sys.path.append(project_root)          # enable absolute imports from the root
print(f"Project root added to PYTHONPATH -> {project_root}")

# ------------------------------------------------------------
# 4️⃣ Helper to load a function from an arbitrary file (used for NLM)
# ------------------------------------------------------------
def load_function_from_path(file_path: str, func_name: str):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("temp_module", pathlib.Path(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, func_name)

# ------------------------------------------------------------
# 5️⃣ Registry of models to benchmark
# ------------------------------------------------------------
# (module_path, entry_name, kind)
# kind = None   → classic NumPy‑style function (e.g., NLM)
# kind = "torch"→ PyTorch nn.Module class
registry = {
    # Classic algorithms
    "BM3D"        : ("BM3D.BM3D", "BM3D_Step1", None),
    "NLM"         : ("NLM/Non-Local-Means-main/main.py", "nonLocalMeans", None),

    # PyTorch models (module, class, kind)
    "cncl"        : ("models.cncl", "GeneratorUNet", "torch"),
    "dncnn"       : ("models.dncnn", "DnCNN", "torch"),
    "edcnn"       : ("models.edcnn", "EDCNN", "torch"),
    "leda"        : ("models.leda", "LEDA_ESAU", "torch"),
    "ct-former"   : ("models.ctformer.CTformer", "CTFormer", "torch"),
    "red-cnn"     : ("models.red_cnn", "RED_CNN", "torch"),
    "u-former"    : ("models.uformer", "Uformer", "torch"),
    # Corrected entry – file is `pdlm_ldctid.py` in the project root
    "pdlm_ldctid" : ("pdlm_ldctid", "PDLM_LDCTID", "torch")
}

# ------------------------------------------------------------
# 6️⃣ Dummy inputs (adjust size if a model expects something else)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dummy_1c = torch.randn(1, 1, 256, 256, device=device)
_dummy_3c = torch.randn(1, 3, 256, 256, device=device)

# ------------------------------------------------------------
# 7️⃣ Helper to benchmark a callable (classic or torch)
# ------------------------------------------------------------
def benchmark_callable(fn, *args, warmup: int = 5, repeats: int = 30):
    """Runs warm‑up passes then returns the average runtime in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats * 1000   # ms

# ------------------------------------------------------------
# 8️⃣ Main evaluation loop
# ------------------------------------------------------------
results = []

for model_name, (mod_path, entry_name, kind) in registry.items():
    print(f"\nProcessing: {model_name}")

    # ------------------------------------------------
    # Classic NumPy‑based functions (BM3D, NLM)
    # ------------------------------------------------
    if kind is None:
        try:
            if model_name == "NLM":
                func = load_function_from_path(os.path.join(project_root, mod_path), entry_name)
            else:
                mod = importlib.import_module(mod_path)
                func = getattr(mod, entry_name)
        except Exception as e:
            results.append({
                "Model": model_name,
                "Parameters (M)": "‑",
                "FLOPs (G)"      : "‑",
                "Inference Time (ms)": f"Import error: {e}",
                "Epochs": "‑"
            })
            print(f"❌ Import error for {model_name}: {e}")
            continue
        img = np.random.rand(256, 256).astype(np.float32) * 255
        elapsed_ms = benchmark_callable(func, img)
        results.append({
            "Model": model_name,
            "Parameters (M)": 0,
            "FLOPs (G)"      : 0,
            "Inference Time (ms)": round(elapsed_ms, 2),
            "Epochs": "‑"
        })
        print(f"✅ {model_name} → {elapsed_ms:.2f} ms")
        continue

    # ------------------------------------------------
    # PyTorch models
    # ------------------------------------------------
    try:
        mod = importlib.import_module(mod_path)
        ModelClass = getattr(mod, entry_name)
    except Exception as e:
        results.append({
            "Model": model_name,
            "Parameters (M)": "‑",
            "FLOPs (G)"      : "‑",
            "Inference Time (ms)": f"Import error: {e}",
            "Epochs": "‑"
        })
        print(f"❌ Import error for {model_name}: {e}")
        continue

    # ----- Instantiate the model ---------------------------------
    try:
        sig = inspect.signature(ModelClass.__init__)
        params = sig.parameters
        if model_name == "u-former":
            model = ModelClass(in_chans=3, dd_in=1).to(device)
        elif "image_channels" in params:
            model = ModelClass(image_channels=1).to(device)
        elif "in_channels" in params:
            model = ModelClass(in_channels=1).to(device)
        elif "in_ch" in params:
            model = ModelClass(in_ch=1).to(device)
        else:
            model = ModelClass().to(device)
    except Exception as e:
        results.append({
            "Model": model_name,
            "Parameters (M)": "‑",
            "FLOPs (G)"      : "‑",
            "Inference Time (ms)": f"Instantiation error: {e}",
            "Epochs": "‑"
        })
        print(f"❌ Instantiation error for {model_name}: {e}")
        continue

    model.eval()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs
    try:
        dummy_input = _dummy_3c if model_name == "u-former" else _dummy_1c
        macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
        flops = macs * 2
        print("    FLOPs = macs * 2   # 1 MAC = 2 FLOPs")
    except Exception as e:
        macs, flops = None, None
        print(f"⚠️ FLOPs calculation failed for {model_name}: {e}")

    # Choose dummy input based on model channel expectations
    if model_name == "u-former":
        dummy_input = _dummy_1c
    else:
        dummy_input = _dummy_3c
    elapsed_ms = benchmark_callable(model, dummy_input)

    results.append({
        "Model": model_name,
        "Parameters (M)": round(param_count / 1e6, 4),
        "FLOPs (G)"      : round(flops / 1e9, 4) if flops is not None else "‑",
        "Inference Time (ms)": round(elapsed_ms, 2),
        "Epochs": "‑"
    })
    print(
        f"✅ {model_name} → Params: {param_count/1e6:.4f} M, "
        f"FLOPs: {flops/1e9 if flops else '‑'} G, "
        f"Time: {elapsed_ms:.2f} ms"
    )

# ------------------------------------------------------------
# 9️⃣ Assemble results into a pretty DataFrame
# ------------------------------------------------------------
df = pd.DataFrame(results)
display(df)

# ------------------------------------------------------------
# 🔟 Save to CSV (optional but handy for later analysis)
# ------------------------------------------------------------
csv_dir = os.getenv("KAGGLE_WORKING_DIR", project_root)
csv_path = os.path.join(csv_dir, "model_complexity_summary.csv")
print(f"\n💾 Summary saved to: {csv_path}")
