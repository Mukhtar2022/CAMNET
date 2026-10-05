"""
benchmark_all_models.py
=======================
Computes Parameters (M), FLOPs (G), and Inference Time (ms) for:
  DnCNN, EDCNN, RED-CNN, NLM, BM3D, CT-Former, LEDA, U-Former, CNCL, PDLM_LDCTID

Run:  python benchmark_all_models.py
Output: model_complexity_all.csv
"""

import os, sys, time, importlib, importlib.util, pathlib
import numpy as np
import torch
import pandas as pd

ROOT = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")
print(f"[INFO] PyTorch: {torch.__version__}")

# ── helpers ──────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_time_torch(fn, dummy, warmup=5, repeats=20):
    with torch.no_grad():
        for _ in range(warmup):
            fn(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            fn(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats * 1000.0   # ms

def measure_time_numpy(fn, img, warmup=3, repeats=10):
    for _ in range(warmup):
        fn(img)
    t0 = time.time()
    for _ in range(repeats):
        fn(img)
    return (time.time() - t0) / repeats * 1000.0   # ms

def get_flops(model, dummy):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return round(macs * 2 / 1e9, 4)   # 1 MAC = 2 FLOPs
    except Exception as e:
        print(f"  [WARN] FLOPs failed: {e}")
        return "N/A"

def load_fn_from_file(filepath, funcname):
    spec = importlib.util.spec_from_file_location("_tmp", pathlib.Path(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, funcname)

# ── result accumulator ───────────────────────────────────────────────────────

results = []

def add_result(name, params_m, flops_g, time_ms, note=""):
    tag = f" [{note}]" if note else ""
    print(f"  -> Params: {params_m} M | FLOPs: {flops_g} G | Time: {time_ms} ms{tag}")
    results.append({
        "Model"             : name,
        "Parameters (M)"   : params_m,
        "FLOPs (G)"        : flops_g,
        "Inference Time (ms)": time_ms,
        "Note"             : note,
    })

# dummy inputs (standard 256x256 and CT-Former specific 64x64)
dummy_256 = torch.randn(1, 1, 256, 256, device=device)
dummy_64  = torch.randn(1, 1,  64,  64, device=device)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DnCNN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/10] DnCNN ...")
try:
    from models.dncnn import DnCNN
    model = DnCNN(image_channels=1).to(device).eval()
    p = round(count_params(model) / 1e6, 4)
    f = get_flops(model, dummy_256)
    t = round(measure_time_torch(model, dummy_256), 2)
    add_result("DnCNN", p, f, t)
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("DnCNN", "ERR", "ERR", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDCNN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/10] EDCNN ...")
try:
    from models.edcnn import EDCNN
    model = EDCNN(in_ch=1).to(device).eval()
    p = round(count_params(model) / 1e6, 4)
    f = get_flops(model, dummy_256)
    t = round(measure_time_torch(model, dummy_256), 2)
    add_result("EDCNN", p, f, t)
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("EDCNN", "ERR", "ERR", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 3. RED-CNN  (needs >= 40x40 due to no-padding convs)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/10] RED-CNN ...")
try:
    from models.red_cnn import RED_CNN
    model = RED_CNN().to(device).eval()
    p = round(count_params(model) / 1e6, 4)
    f = get_flops(model, dummy_256)
    t = round(measure_time_torch(model, dummy_256), 2)
    add_result("RED-CNN", p, f, t)
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("RED-CNN", "ERR", "ERR", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 4. NLM  (classical - no params/FLOPs, measure time only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/10] NLM ...")
try:
    nlm_path = os.path.join(ROOT, "NLM", "Non-Local-Means-main", "main.py")
    nlm_fn   = load_fn_from_file(nlm_path, "nonLocalMeans")
    img_np   = (np.random.rand(256, 256) * 255).astype(np.float32)
    # Provide default params (bigWindowSize, smallWindowSize, h) as used in the NLM script
    params = (20, 6, 14)
    t = round(measure_time_numpy(lambda x: nlm_fn(x, params=params), img_np), 2)
    add_result("NLM", "N/A", "N/A", t, "classical - no learnable params")
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("NLM", "N/A", "N/A", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 5. BM3D  (classical - no params/FLOPs, measure time only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/10] BM3D ...")
try:
    bm3d_mod = importlib.import_module("BM3D.BM3D")
    bm3d_fn  = getattr(bm3d_mod, "BM3D_Step1")
    img_np   = (np.random.rand(256, 256) * 255).astype(np.float32)
    t = round(measure_time_numpy(bm3d_fn, img_np), 2)
    add_result("BM3D", "N/A", "N/A", t, "classical - no learnable params")
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("BM3D", "N/A", "N/A", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 6. CT-Former  (hardcoded 64x64 T2T module)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/10] CT-Former ...")
try:
    from models.ctformer.CTformer import CTFormer
    model = CTFormer(
        img_size=64, tokens_type="performer", in_chans=1,
        embed_dim=64, depth=1, num_heads=8,
        kernel=4, stride=4, mlp_ratio=2., token_dim=64
    ).to(device).eval()
    p = round(count_params(model) / 1e6, 4)
    f = get_flops(model, dummy_64)
    t = round(measure_time_torch(model, dummy_64), 2)
    add_result("CT-Former", p, f, t, "64x64 patch (T2T fixed)")
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("CT-Former", "ERR", "ERR", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 7. LEDA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/10] LEDA ...")
try:
    from models.leda import LEDA_ESAU
    model = LEDA_ESAU(in_channels=1, out_channels=1).to(device).eval()
    p = round(count_params(model) / 1e6, 4)
    f = get_flops(model, dummy_256)
    t = round(measure_time_torch(model, dummy_256), 2)
    add_result("LEDA", p, f, t)
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("LEDA", "ERR", "ERR", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 8. U-Former
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/10] U-Former ...")
try:
    from models.uformer import Uformer
    model = Uformer(
        img_size=256, in_chans=1, dd_in=1, embed_dim=32,
        depths=[2,2,2,2,2,2,2,2,2], win_size=8
    ).to(device).eval()
    p = round(count_params(model) / 1e6, 4)
    f = get_flops(model, dummy_256)
    t = round(measure_time_torch(model, dummy_256), 2)
    add_result("U-Former", p, f, t)
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("U-Former", "ERR", "ERR", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 9. CNCL (GeneratorUNet - dual U-Net, outputs 2ch)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9/10] CNCL ...")
try:
    from models.cncl import GeneratorUNet
    model = GeneratorUNet().to(device).eval()
    p = round(count_params(model) / 1e6, 4)
    f = get_flops(model, dummy_256)
    t = round(measure_time_torch(model, dummy_256), 2)
    add_result("CNCL", p, f, t)
except Exception as e:
    print(f"  [ERROR] {e}")
    add_result("CNCL", "ERR", "ERR", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 10. PDLM_LDCTID (our proposed model from pdlm_ldctid.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10/10] PDLM_LDCTID ...")
try:
    pdlm_spec = importlib.util.spec_from_file_location(
        "pdlm_ldctid", pathlib.Path(ROOT) / "pdlm_ldctid.py"
    )
    pdlm_mod = importlib.util.module_from_spec(pdlm_spec)
    pdlm_spec.loader.exec_module(pdlm_mod)
    # Try common class names
    cls = None
    for name in ["PDLM_LDCTID", "Proposed_RED_CNN", "ProposedModel", "Model"]:
        if hasattr(pdlm_mod, name):
            cls = getattr(pdlm_mod, name)
            print(f"  Found class: {name}")
            break
    if cls is None:
        raise AttributeError("No known model class found in pdlm_ldctid.py")
    model = cls().to(device).eval()
    p = round(count_params(model) / 1e6, 4)
    f = get_flops(model, dummy_256)
    t = round(measure_time_torch(model, dummy_256), 2)
    add_result("PDLM_LDCTID", p, f, t)
except Exception as e:
    print(f"  [ERROR] {e}")
    # Fallback: try the proposed_red_cnn model
    try:
        print("  Fallback: trying models/proposed_red_cnn.py ...")
        from models.proposed_red_cnn import Proposed_RED_CNN
        model = Proposed_RED_CNN().to(device).eval()
        p = round(count_params(model) / 1e6, 4)
        f = get_flops(model, dummy_256)
        t = round(measure_time_torch(model, dummy_256), 2)
        add_result("PDLM_LDCTID (proposed_red_cnn)", p, f, t)
    except Exception as e2:
        print(f"  [ERROR fallback] {e2}")
        add_result("PDLM_LDCTID", "ERR", "ERR", "ERR", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Save & Display
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
csv_out = os.path.join(ROOT, "model_complexity_all.csv")
df.to_csv(csv_out, index=False)

print("\n" + "="*72)
print("  MODEL COMPLEXITY SUMMARY")
print("="*72)
print(df.to_string(index=False))
print("="*72)
print(f"\n[INFO] Results saved to: {csv_out}")
