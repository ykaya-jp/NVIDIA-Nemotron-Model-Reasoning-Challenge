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
    """Run SFT + LoRA via TRL's prompt-completion + completion-only-loss path.

    Notes
    -----
    - `output_dir` should point at a Drive-mounted path (e.g.
      /content/drive/MyDrive/nemotron-2026/checkpoints) so checkpoints
      survive a Colab session drop. The 2026-05-14 night run got to
      step 171/206 (83%) before the session died with no checkpoint
      written because save_steps=2000 > total steps (206). Lesson:
      checkpoint frequently AND on persistent storage.
    - `learning_rate=1e-4` (down from 2e-4) — the night-run loss
      plateaued around step 160 at 0.85, which the discussion thread
      audit (docs/research/2026-05-14-discussion-audit.md §2) ties to
      our 13× bit upsample drifting toward catastrophic forgetting.
      Halving lr is a conservative mitigation that doesn't redo data.
    - `resume_from_checkpoint=True` — if a checkpoint exists at
      output_dir from a previous run, TRL auto-resumes from it
      (optimizer + scheduler + step count restored).
    """
    # peft<0.16 still ships a LoRA torchao dispatcher that calls
    # `is_torchao_available()` and raises ImportError if the
    # installed torchao is older than 0.16. Colab pre-installs
    # torchao 0.10, and even cell 2's `pip uninstall torchao` can be
    # undone by a subsequent install that pulls it back in. We don't
    # use torchao quantization at all, so short-circuit the check so
    # the dispatcher returns "no torchao backend, try the next one".
    import peft.import_utils as _peft_import_utils

    _peft_import_utils.is_torchao_available = lambda: False
    # Defensive: peft.tuners.lora.torchao may have already imported
    # `is_torchao_available` by name at module-load time.
    try:
        import peft.tuners.lora.torchao as _peft_torchao

        _peft_torchao.is_torchao_available = lambda: False
    except Exception:
        pass

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    # Codex audit 4 §WARN (2026-05-16): "all-linear" expands via PEFT's
    # module tree and may include MoE gating / Mamba-side projections
    # that vLLM's LoRARequest does not load → silent zero-improvement
    # adapter at inference. The two working 0.85-LB public kernels
    # (dgxchen via Unsloth, konbu17 via plain PEFT) both use an explicit
    # 8-module list. Match that exact set.
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj",
            "out_proj",
            "up_proj",
            "down_proj",
        ],
    )

    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        learning_rate=1e-4,  # ← lowered from 2e-4 (catastrophic-forgetting mitigation)
        warmup_ratio=0.03,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=50,  # ← was 2000; 50 = checkpoint roughly every 1-1.5 h on A100
        save_total_limit=3,  # keep last 3 checkpoints to bound Drive usage
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

    # Resume if a previous run left a checkpoint at output_dir, else
    # start fresh. TRL detects checkpoint subdirs of the form
    # `checkpoint-<step>/` automatically.
    import os as _os

    has_ckpt = _os.path.isdir(output_dir) and any(
        d.startswith("checkpoint-") for d in _os.listdir(output_dir)
    )
    if has_ckpt:
        print(f"↻ resuming from latest checkpoint under {output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    print("✓ training done")
    try:
        import torch as _torch

        peak_gb = _torch.cuda.max_memory_allocated() / 1024**3
        print(f"   peak VRAM = {peak_gb:.1f} GB (Codex WARN 4 audit)")
    except Exception:
        pass
    return trainer
