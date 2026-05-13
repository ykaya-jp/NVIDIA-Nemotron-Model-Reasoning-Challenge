# 2026-05-13 pre-run audit — train_lora_colab.ipynb

> 起源: ユーザー指示「同じような問題が起こりそうな可能性や個所はないか事前に確認してほしい」
> 反省: cell 単位の reactive single-fix push を 4 連発した後、 残り cell を **先回り audit** すべきだった
> 親 教訓: `~/projects/kaggle/CLAUDE.md` §0.5 「silent except 禁止 / Codex review 必須化」 (orbit-wars Phase β.1 14h silent bug 由来)

## 確認手順

1. notebook の cell 1-6 を 1 cell ずつ読み返し、 想定通り 動作するか質問形式で点検
2. TRL 0.22.2 公式 docs (https://huggingface.co/docs/trl/sft_trainer) で `completion_only_loss` の前提を再確認
3. Nemotron-3-Nano tokenizer の `apply_chat_template` 仕様を web 検索で再確認
4. 各 risk を severity 付きで列挙、 高 severity から先に fix
5. 修正後 `/codex:review` を 1 回走らせる (= 親 §0.5 で MUST 化、 朝に user が走らせる)

## 発見した risk 一覧

### R1 【CRITICAL silent failure】 `completion_only_loss=True` の format 不整合

- **症状**: 8h training が無事完走、 LB が 0.85 から動かない (= 想定 0.87-0.91)
- **原因**: TRL 0.22 公式 docs 「To train on completion only, use a **prompt-completion** dataset」。 我々の旧 dataset は `{"text": prompt + assistant}` = **language modeling 形式**。 TRL は flag を silent ignore し、 全 token に loss 計上 → prompt の繰り返し学習で over-regularize
- **fix**: `render()` を `{"prompt": [{"role":"user","content":...}], "completion": [{"role":"assistant","content":...}]}` に変更、 SFTConfig から `dataset_text_field` を削除 (= TRL が conversational 形式を auto detect)
- **副次効果**: `enable_thinking=True` を render 側で渡せなくなる → ただし Nemotron-3-Nano は chat template の default で `<think>` token id 12 を自動付与するので無問題 (出典: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/vllm_cookbook.ipynb)

### R2 【MEDIUM】 `gradient_checkpointing_enable()` の 2 重指定

- **症状**: 「RuntimeError: element 0 of tensors does not require grad」 が training 開始直後に出る可能性
- **原因**: Cell 3 で `model.gradient_checkpointing_enable()` を呼んだ後、 SFTConfig が default `gradient_checkpointing=True` で再 setup。 PEFT wrap 経路で `prepare_model_for_kbit_training` が `input_require_grads` を再 wiring する際、 既に enable 済の state と競合
- **fix**: Cell 3 の呼び出しを削除、 SFTConfig 任せに統一

### R3 【LOW (audit のみ)】 `max_length=2048` で長 record の truncate

- **症状**: bit solver の per-bit trace が長く 2048 超 → 末尾の `\boxed{...}` が truncate 消失 → silent label loss
- **fix**: Cell 4 末尾に **200 サンプル分の token 長分布 audit print** を追加。 max / p95 / truncate% を表示。 truncate% > 10% なら警告で user 通知 (= 朝起きたら Cell 4 出力で即可視化)
- **判断**: 10% 以下なら学習継続、 > 10% なら MAX_SEQ_LEN を 3072 / 4096 に上げる (memory permitting) or CoT 短縮を検討

### R4 【LOW】 INFERENCE_USER_SUFFIX の train-inference mismatch

- **症状**: 公式 metric notebook が user prompt にどう suffix を付けるか不明、 mismatch なら inference 時 `\boxed{}` 産出を model が「学習時の suffix なし」 として認識しない可能性
- **判断**: 保守的に維持 (= 学習時に「`\boxed{}` 形式で答えろ」 を明示)。 公式 metric が suffix を付けない場合でも、 model は instruction-tuned で「user が指示してれば従う」 性質、 害は最小
- **fix**: 不要 (= audit のみ)

### R5 (確認のみ) — 問題なし

- `optim="adamw_torch_fused"` は torch 2.7.1 + CUDA 12.8 で supported、 LoRA params のみに operate
- `attn_implementation="eager"` は Mamba-Transformer hybrid の必須設定 (= NVIDIA 公式 references で確認)
- `mamba_ssm==2.2.5`, `causal_conv1d==1.5.2` は torch 2.7.1+cu128 で動作確認済 stack (= DataCamp tutorial)

## 修正 commit に含めるもの

1. Cell 3: `model.gradient_checkpointing_enable()` 削除 (R2)
2. Cell 4: `render()` を prompt-completion 形式に変更 + 長分布 audit print 追加 (R1, R3)
3. Cell 5: SFTConfig から `dataset_text_field` 削除 + `completion_only_loss` の意味を comment で明文化 (R1)
4. .ipynb 再生成 (= cell ID + structure stable)
5. 本 audit doc 起票

## Codex review 推奨

朝起床時 (= user が起きたら) **`/codex:review --base origin/main~3`** で 3 commit 分まとめてレビュー。 親 §0.5 「実装段階: submission build / training script / inference adapter commit 前に Codex review × 1」 を遵守。 自動 gate が走っていても **明示的に 1 回手動 review** を保険として走らせる。

## 出典

- TRL SFT Trainer docs (= prompt-completion 仕様、 completion_only_loss): https://huggingface.co/docs/trl/sft_trainer
- TRL SFTConfig source (= dataset_text_field default、 chat template handling): https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py
- Nemotron-3-Nano chat template (`enable_thinking` default): https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/vllm_cookbook.ipynb
- DataCamp Nemotron-3-Nano LoRA tutorial (= 動作確認済 stack): https://www.datacamp.com/tutorial/fine-tuning-nvidia-nemotron-3-nano
