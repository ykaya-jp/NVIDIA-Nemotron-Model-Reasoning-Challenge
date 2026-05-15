"""Baseline submission with andreyyunoshev/huikang-nemotron-adapter-mirror.

While our v1 LoRA is blocked on Colab quota, attach huikang's
public LB-0.85 adapter mirror (1.5 GB, adapter_config.json +
adapter_model.safetensors). Reference for our future v1/v2."""
import os, pandas as pd
LORA_DIR = "/kaggle/input/huikang-nemotron-adapter-mirror"
assert os.path.exists(LORA_DIR), f"adapter not at {LORA_DIR}"
print("adapter contents:")
for fn in sorted(os.listdir(LORA_DIR)):
    p = os.path.join(LORA_DIR, fn)
    sz = os.path.getsize(p) if os.path.isfile(p) else "<dir>"
    print(f"  {fn}: {sz}")
TEST_PATH = "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/test.csv"
test = pd.read_csv(TEST_PATH)
row_id_col = test.columns[0]
print(f"test set: {len(test)} rows, id={row_id_col!r}")
sub = pd.DataFrame({row_id_col: test[row_id_col], "prediction": LORA_DIR})
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"wrote {len(sub)} rows"); print(sub.head(3))
