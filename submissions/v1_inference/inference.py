"""Thin submission kernel for Nemotron v1.

GPU is OFF — this notebook does no inference itself; it only writes
submission.csv pointing at the LoRA adapter we trained on Colab and
uploaded as `ky7240/nemotron-v1-adapter`. The official metric notebook
runs vLLM with that adapter on its own GPU.

Workflow:
1. Train on Colab Pro+ A100 → adapter at /content/drive/MyDrive/
   nemotron-2026/adapter_v1/.
2. Upload that folder to Kaggle as Dataset `ky7240/nemotron-v1-adapter`.
3. This notebook attaches that dataset and writes submission.csv.
4. Submit to the competition; LB confirmed in ~2 h.

This kernel does NOT need a GPU, so it bypasses the 30 h weekly quota.
"""

import os
import pandas as pd

# Adapter directory (= mounted from the attached Kaggle Dataset)
LORA_DIR = "/kaggle/input/nemotron-v1-adapter"

assert os.path.exists(LORA_DIR), (
    f"Adapter dataset not mounted at {LORA_DIR}. Make sure ky7240/"
    f"nemotron-v1-adapter is attached via 'Add data'."
)
print("Adapter directory contents:")
for fn in sorted(os.listdir(LORA_DIR)):
    sz = os.path.getsize(os.path.join(LORA_DIR, fn))
    print(f"  {fn}: {sz:,} bytes")

# Build submission.csv. The official metric notebook reads the
# 'prediction' column as the adapter directory path (a single path is
# repeated for every test row).
TEST_PATH = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/test.csv"
test = pd.read_csv(TEST_PATH)
row_id_col = test.columns[0]
print(f"test set: {len(test)} rows, id column: {row_id_col!r}")

submission = pd.DataFrame({
    row_id_col: test[row_id_col],
    "prediction": LORA_DIR,
})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(f"✓ wrote submission.csv with {len(submission)} rows")
print(submission.head(3))
