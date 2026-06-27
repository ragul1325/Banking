"""
=====================================================
BANKING PROJECT - MAIN RUNNER
=====================================================
Run this file to execute the complete pipeline:
  1. Data Generation
  2. EDA
  3. Loan Default Prediction
  4. Customer Segmentation
  5. Recommendation Engine
=====================================================
"""

import subprocess
import sys
import os
import time

STEPS = [
    ("1_generate_data.py",  "Step 1: Data Generation"),
    ("2_eda.py",            "Step 2: Exploratory Data Analysis"),
    ("3_loan_default.py",   "Step 3: Loan Default Prediction"),
    ("4_segmentation.py",   "Step 4: Customer Segmentation"),
    ("5_recommendation.py", "Step 5: Recommendation Engine"),
]

def run_step(script, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"  ✅ Completed in {elapsed:.1f}s")
    else:
        print(f"  ❌ FAILED after {elapsed:.1f}s")
        sys.exit(1)

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  PREDICTIVE ANALYTICS & RECOMMENDATION SYSTEMS")
    print("  IN BANKING — FULL PROJECT PIPELINE")
    print("="*55)
    total_start = time.time()

    for script, label in STEPS:
        run_step(script, label)

    total = time.time() - total_start
    print(f"\n{'='*55}")
    print(f"  ✅ ALL STEPS COMPLETE  ({total:.1f}s total)")
    print(f"{'='*55}")
    print("\nOutput files:")
    for root, dirs, files in os.walk("outputs"):
        for f in files:
            path = os.path.join(root, f)
            print(f"  📄 {path}")
    print("\nModels saved:")
    for f in os.listdir("models"):
        print(f"  🤖 models/{f}")
