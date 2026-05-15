import os, sys
print("=== /kaggle/input contents ===")
if os.path.isdir("/kaggle/input"):
    for entry in sorted(os.listdir("/kaggle/input")):
        full = os.path.join("/kaggle/input", entry)
        marker = "/" if os.path.isdir(full) else ""
        print(f"  {entry}{marker}")
        if os.path.isdir(full):
            for sub in sorted(os.listdir(full))[:8]:
                p = os.path.join(full, sub)
                sz = os.path.getsize(p) if os.path.isfile(p) else "<dir>"
                print(f"    {sub}: {sz}")
else:
    print("  (no /kaggle/input)")
print("=== competition data check ===")
test_path = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/test.csv"
print(f"test.csv exists: {os.path.exists(test_path)}")
# Write a dummy submission so the kernel reports COMPLETE (otherwise no
# submission means submit step would fail anyway).
import pandas as pd
if os.path.exists(test_path):
    test = pd.read_csv(test_path)
    print(f"test rows: {len(test)}, cols: {list(test.columns)[:3]}")
    sub = pd.DataFrame({test.columns[0]: test[test.columns[0]], "prediction": "/kaggle/input/diagnostic"})
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print("wrote dummy submission.csv")
