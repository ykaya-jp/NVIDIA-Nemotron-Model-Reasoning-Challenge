"""Hot-reloadable Colab training pipeline.

Why this file exists: the Colab notebook used to inline cells 4 and 5
(data build + SFT training). That meant every fix forced the user to
restart the Colab session, redo pip install (~5 min) and re-download
the 60 GB Nemotron-30B weights (~5-10 min) before the new code ran.

This module centralises the cell-4/5 logic so the notebook can be a
thin shell: cell 1 `git pull`s this file, cells 4/5 import-then-run.
A single re-run of cell 1 + cells 4/5 picks up any of my fixes —
install + model load (cells 2 & 3) stay live.

Public entry points:
    build_dataset(rows, tokenizer, max_seq_len) -> Dataset
    train_lora(model, tokenizer, ds, output_dir) -> SFTTrainer
"""

from __future__ import annotations

import json
import os
from collections import Counter

from datasets import Dataset, concatenate_datasets

INFERENCE_USER_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. " "For example: `\\boxed{your answer}`"
)

# Codex review 3 §3 — upsample under-represented sources so they
# survive epoch-1 gradient mass.
UPSAMPLE = {"solver_bit": 13, "solver_cipher": 2}


def _to_prompt_completion(rec: dict) -> dict:
    """Conversational prompt-completion format.

    TRL 0.22's `completion_only_loss=True` is only honoured when the
    dataset is prompt-completion; a single 'text' field would silently
    fall back to language-modeling mode and compute loss on the prompt
    as well.
    """
    user_content = rec["user"] + INFERENCE_USER_SUFFIX
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "completion": [{"role": "assistant", "content": rec["assistant"]}],
    }


def load_rows(sft_path: str) -> list[dict]:
    assert os.path.exists(sft_path), f"SFT data missing at {sft_path}"
    with open(sft_path) as f:
        rows = [json.loads(line) for line in f]
    print(f"loaded {len(rows)} verifier-backed records")
    print("source distribution:", Counter(r["source"] for r in rows))
    return rows


def build_dataset(rows: list[dict], tokenizer, max_seq_len: int) -> Dataset:
    """Return an upsampled prompt-completion Dataset + print length audit."""
    ds = Dataset.from_list([_to_prompt_completion(r) for r in rows])

    extras = []
    for r in rows:
        n = UPSAMPLE.get(r["source"], 1)
        if n > 1:
            extras.extend([r] * (n - 1))
    if extras:
        extras_ds = Dataset.from_list([_to_prompt_completion(r) for r in extras])
        ds = concatenate_datasets([ds, extras_ds]).shuffle(seed=42)
    print(f"after upsampling: {len(ds)} records")

    # Audit (R3 from docs/dev/2026-05-13-pre-run-audit.md): see how
    # many records get truncated at max_seq_len. A truncation that
    # drops the trailing `\boxed{...}` is a silent label loss.
    def _approx_len(ex):
        txt = (
            tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
            + ex["completion"][0]["content"]
        )
        return len(tokenizer(txt, add_special_tokens=False)["input_ids"])

    step = max(1, len(ds) // 200)
    sample_idx = list(range(0, len(ds), step))[:200]
    lens = [_approx_len(ds[i]) for i in sample_idx]
    n_trunc = sum(1 for L in lens if L > max_seq_len)
    p95 = sorted(lens)[int(len(lens) * 0.95)]
    print(
        f"length audit on {len(lens)}-sample: "
        f"max={max(lens)} p95={p95} "
        f"truncated@{max_seq_len}={n_trunc} "
        f"({n_trunc / len(lens):.1%})"
    )
    if n_trunc / len(lens) > 0.10:
        print(
            f"⚠️  >10% of sampled records exceed max_seq_len={max_seq_len}; "
            f"consider raising it (memory permitting) or shortening the CoT."
        )
    return ds


def train_lora(model, tokenizer, ds: Dataset, *, output_dir: str, max_seq_len: int):
    """Run SFT + LoRA via TRL's prompt-completion + completion-only-loss path."""
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=1,
        optim="adamw_torch_fused",
        seed=42,
        report_to="none",
        max_length=max_seq_len,
        packing=False,
        completion_only_loss=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()
    print("✓ training done")
    return trainer
