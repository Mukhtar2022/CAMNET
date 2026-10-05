import os
import subprocess

BASE_DIR = r"C:\Users\HP\Documents\COVID-19-EXPIREMENT"

# Active models to evaluate and their arguments
models_to_run = [
    {"model": "dncnn", "suffix": ""},
    {"model": "red_cnn", "suffix": "final_journal_run"},
    {"model": "edcnn", "suffix": ""},
    {"model": "ctformer", "suffix": ""},
    {"model": "uformer", "suffix": ""},
    {"model": "leda", "suffix": ""},
    {"model": "proposed_model", "suffix": "epo50"},
]

def main():
    print("=" * 80)
    print("STARTING EVALUATION OF ALL DEEP LEARNING BENCHMARK MODELS")
    print("=" * 80)
    
    # 1. Run evaluations to generate individual CSV files
    for run in models_to_run:
        model_name = run["model"]
        suffix = run["suffix"]
        
        cmd = ["python", "evaluate.py", "--model", model_name]
        if suffix:
            cmd += ["--suffix", suffix]
            
        print(f"\n[EXEC] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=BASE_DIR, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running model {model_name}: {e}")
        
    # 2. Run classical baselines (if not already run)
    print("\n[EXEC] Running NLM and BM3D baseline evaluations...")
    try:
        subprocess.run(["python", "evaluate_nlm.py", "--limit", "128"], cwd=BASE_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running NLM baseline: {e}")
        
    try:
        subprocess.run(["python", "evaluate_bm3d.py", "--limit", "128"], cwd=BASE_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running BM3D baseline: {e}")

    # 3. Aggregate all results into the final benchmark table
    print("\n[EXEC] Running comparison aggregation...")
    try:
        subprocess.run(["python", "compare_all_final.py"], cwd=BASE_DIR, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running comparison aggregation: {e}")
    
    print("\nAll model evaluations and comparisons have been successfully completed!")

if __name__ == "__main__":
    main()
