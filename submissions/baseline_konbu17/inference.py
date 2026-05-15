"""Baseline submission with konbu17/nemotron-sft-lora-cot-selection.

Correct submission format (= confirmed via ryanholbrook_submission_demo):
  /kaggle/working/submission.zip containing adapter_config.json +
  adapter_model.safetensors. The official metric notebook unzips this
  and runs vLLM + LoRARequest with the adapter."""
import os, shutil, subprocess

# Find the adapter folder among attached datasets — exact mount path
# differs depending on whether competition_sources is set.
ADAPTER_SRC = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "adapter_config.json" in files and "adapter_model.safetensors" in files:
        ADAPTER_SRC = root
        print(f"found adapter at: {ADAPTER_SRC}")
        break
assert ADAPTER_SRC, "no adapter_config.json found anywhere under /kaggle/input"

OUT = "/kaggle/working"
for fn in os.listdir(ADAPTER_SRC):
    src = os.path.join(ADAPTER_SRC, fn)
    if os.path.isfile(src):
        shutil.copy(src, OUT)
        sz = os.path.getsize(os.path.join(OUT, fn))
        print(f"copied {fn} ({sz:,} bytes)")

subprocess.run(
    f"cd {OUT} && zip -m submission.zip adapter_config.json adapter_model.safetensors",
    shell=True, check=True,
)
print("✓ submission.zip created at /kaggle/working/submission.zip")
print(f"  size: {os.path.getsize(os.path.join(OUT, 'submission.zip')):,} bytes")
