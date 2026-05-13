# NVIDIA Nemotron Model Reasoning Challenge

> Kaggle competition: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
> Deadline: 2026-06-15 · Prize: $106,388 · Teams: 2,959

## ★ Train on Colab Pro+ A100 — 1 click

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ykaya-jp/NVIDIA-Nemotron-Model-Reasoning-Challenge/blob/main/notebooks/train_lora_colab.ipynb)

クリックで `notebooks/train_lora_colab.ipynb` が Colab で開きます。

1. **Runtime → Change runtime type → A100 GPU + High-RAM** を選んで Save
2. **Runtime → Run all** で 6-10 h training (自動 Drive mount + GitHub clone + transformers/peft/TRL LoRA SFT、 Unsloth は Nemotron-Nano load bug = unslothai/unsloth#3480 のため使用しない)
3. 完了後 adapter が `/content/drive/MyDrive/nemotron-2026/adapter_v1/` に保存される
4. その adapter フォルダを Kaggle Dataset として upload (= `ky7240/nemotron-v1-adapter`)
5. [submission kernel](https://www.kaggle.com/code/ky7240/nemotron-v1-inference) を開いて Add data で adapter dataset を attach → Save Version → Submit to Competition

## 戦略要点 (v3)

- **6 deterministic Python solver** (Roman / Physics / Unit / Cipher / Bit + Equation は Day 2) が `src/.../solver_*.py` にあり、 5418 件の verifier-backed Chain-of-Thought を生成 (`data/processed/train_sft_verifier_only.jsonl`)
- LoRA adapter (rank 32) を Nemotron-3-Nano-30B-A3B-BF16 にかぶせて SFT
- 公式 metric notebook 解析済: `rel_tol = 1e-2` 寛容、 hybrid inference 不可、 `\boxed{}` balanced-brace は公式実装済
- LB projection: **0.87-0.91 帯** (= 銀-金射程)、 優勝確率 8-15% (Codex calibrated)
- 詳細: `docs/strategy/winning-strategy.dense.md` / 親 plan `~/.claude/plans/iterative-dancing-pearl.md`

## LB Gate (= 事前約束)

| 公式 LB | 判定 | 次の action |
|---|---|---|
| ≥ 0.865 | Phase 1 success | Phase 2 (4-way ablation) へ |
| 0.850-0.864 | partial | 1 件 targeted patch + 再 submit |
| < 0.850 | abort | 公開 dgxchen baseline に revert |

## 補助 (= 必要に応じて)

- **Kaggle Dataset (SFT data backup)**: https://www.kaggle.com/datasets/ky7240/nemotron-sft-v1
- **Codex review 履歴**: `docs/strategy/codex-review-2026-05-13.md`
- **HANDOFF doc (= 朝起床時の 4 step)**: `docs/dev/HANDOFF-2026-05-14.md`
