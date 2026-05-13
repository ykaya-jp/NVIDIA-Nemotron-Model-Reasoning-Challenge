# 2026-05-14 待機時間 research master summary

> 起源: v1 SFT training (= 8-11h Colab) 待ち時間、 ユーザー指示「待ちの間にできることないんか」 → 4 worker 並列 + 私 main + Codex review (= 計 6 並列) で audit。
> 朝起きた user は本 doc を最初に読めば、 v1 LB 結果に応じた v2 アクションが即決定できる。

---

## 0. 要約 (= 30 秒で読む)

5 source の audit / review から得た **朝 LB 確定後に取る action 一覧**:

| v1 LB | 判定 | 即実行 action (本 doc から引用) |
|---|---|---|
| **≥ 0.865** | success | Phase 2 進入 + Rejection Sampling SFT 検討 (§5 #1) + Per-Category CV setup (§3 worker A #2) |
| **0.850-0.864** | partial | logprob filter + upsample cap 13x → 5x patch (§2 worker B #1) + lr 5e-5 へ降圧 (§2 worker B #2) |
| **< 0.850** | abort | dgxchen baseline (= public 0.85) に revert、 戦略 pivot |

**最 critical 発見** (= 全 audit 中 最高 priority):
1. **Catastrophic forgetting risk** (= discussion audit, §2): 我々の bit upsample 13x が この pattern を踏む。 v2 で 5x に下げる候補
2. **Binary leading-zero mismatch** (= metric audit, §4): bit solver の output format 統一必要、 v1.5 patch 候補
3. **Memory budget 楽観** (= Codex WARN 4, §6): 30B BF16 + LoRA + activation で peak 70GB+、 朝 `torch.cuda.max_memory_allocated()` を確認

---

## 1. 並列実行サマリー

| Worker | 担当 | 状態 | 成果 doc |
|---|---|---|---|
| A | Kaggle CODES audit (= top 30 kernel review) | ✅ 完了 | docs/research/2026-05-14-codes-audit.md |
| B | Discussion forum 巡回 | ✅ 完了 | docs/research/2026-05-14-discussion-audit.md |
| C | 公式 metric notebook deep audit | ✅ 完了 | docs/research/2026-05-14-metric-deep-audit.md |
| D | Equation/Symbol solver DSL + prototype | ✅ 完了 (= 4.69% coverage / 98.65% precision、 公開界隈で初の deterministic 解) | docs/research/2026-05-14-equation-dsl.md + src/.../solver_symbol.py + tests/test_solver_symbol.py |
| E | Codex review HEAD 10 commits | ✅ 完了 (= ALLOW with WARN × 4) | (= 本 doc §6 に要約) |
| 私 main | 過去 LLM-reasoning コンペ winner writeup 集約 | ✅ 完了 | docs/research/2026-05-14-past-comp-winners.md |

---

## 2. Discussion forum audit (= Worker B、 CRITICAL finding 2 件)

> 出典: docs/research/2026-05-14-discussion-audit.md (= 142 行、 10 topic / 848 message 読了)

### 2.1 🔴 CRITICAL: Logprob filtering 必須 (= LB +0.02-0.05 期待値)

- 発信者: tahaalam2009 (= Kaggle discussion thread)
- 主張: **hard category (cryptarithm) の CoT は SFT で学習困難**、 logprob threshold で filter + upsampling cap 3x を追加
- 我々への適用: v2 で SFT data 再生成時、 model logprob で **学習しにくい record を除外**

### 2.2 🔴 CRITICAL: Catastrophic forgetting risk

- 発信者: 複数 top-team
- 主張: **過度な upsampling (= 14x 以上) は base model の汎用知識を上書き**、 LB が下がる
- 推奨 hyperparameter: **lr 5e-5 + gradient accum 64** で stabilize (= 0.73 → 0.82-0.84 という事例)
- **我々の現 setup**: bit upsample **13x** (= ぎりぎり閾値)、 lr 2e-4、 grad accum 32
- 判定: v1 LB が 0.85 から動かなければ catastrophic forgetting 疑い → v2 で upsample 5x + lr 5e-5 + grad accum 64

### 2.3 Host 公式 clarification: なし (= rule 変更 0 件)

binary 比較 fix は 2026-04-06 に既 rescore 完了、 v1 への制約なし。

---

## 3. Kaggle CODES audit (= Worker A、 採用候補 3 件)

> 出典: docs/research/2026-05-14-codes-audit.md (= 215 行、 top 30 kernel + 5 件 deep read)

### 3.1 🟩 Adaptive SVD Merge (= mirzayasirabdullah07/nvidia-nemotron-model-notebook、 80 vote、 2026-05-12 最新)

- 内容: Mamba `gate_proj + x_proj → in_proj` の variance-retained SVD merge (98% threshold)
- 効果: inference latency **-10-20%** 期待
- 採用判断: v2-v3 で検討、 effort S (= 半日)

### 3.2 🟨 Per-Category CV (= konbu17/nemotron-tong-style-cot-sft-updated-v2、 90 vote、 rank 26)

- 内容: 6 puzzle type 別評価分離 (= GC/NC/UC/TE/ET/BM の per-type score)、 category imbalance detection
- 採用判断: **Phase 2 ablation で必須** (= 我々の per-type 学習効果を測る setup の reference)、 effort S

### 3.3 🟥 Model Enumeration (= rohanrk1813/nvidia-comp、 73 vote)

- 内容: 14 model version の系統図、 code 価値低、 reference のみ
- 採用判断: 不要

---

## 4. 公式 metric notebook deep audit (= Worker C、 mismatch risk 3 件)

> 出典: docs/research/2026-05-14-metric-deep-audit.md (= 212 行、 8 section、 nvidia-nemotron-metric.ipynb 完全解析)

### 4.1 🔴 A1 Binary leading-zero mismatch (CRITICAL)

- 症状: 公式 metric が `"00011"` vs `"11"` を **full string match** で判定 → 我々の bit solver が leading zero 付きで出力すると silent -0.5%+
- 対策: bit solver の output で leading zero 不付与 (= v1.5 patch 候補)

### 4.2 🔴 A2 Unit/symbol non-numeric (CRITICAL)

- 症状: `"3.5 m"` や `"\frac{1}{2}"` は float cast exception → string fallback で case-insensitive 比較、 ただし numeric 期待の record は失敗
- 対策: SFT で numeric-only output 強制 (= 単位 suffix 禁止)

### 4.3 🟠 A3 Empty `\boxed{}` fallback (HIGH)

- 症状: model が空 `\boxed{}` 出力すると step 3-4 fallback → 後続テキストが誤抽出
- 対策: SFT data に empty boxed 禁止 + inference で generation length 強制

### 4.4 確認 finding (= 重要)

- **`enable_thinking=True` が metric notebook の正式 path** = 我々の prompt-completion 形式 (= chat template default で thinking enabled) と整合、 OK
- prompt format: `test.csv` の prompt 列 + suffix `\nPlease put your final answer inside \boxed{...}` = **我々の `INFERENCE_USER_SUFFIX` と byte-identical** に近い、 OK

---

## 5. 過去コンペ winner writeup (= 私 main thread)

> 出典: docs/research/2026-05-14-past-comp-winners.md (= 200 行、 AIMO 1-2 / NeurIPS 2023 / RLVR の 4 領域)

### 5.1 結論: AIMO 系の最強技法は **公式 metric 制約で大半 不可**

| AIMO 技法 | 我々の comp | 採否 |
|---|---|---|
| Tool-Integrated Reasoning (= Python REPL) | tool 禁止 | ❌ 不可 |
| Self-Consistency / Majority voting | single inference 強制 | ❌ 不可 |
| GenSelect (= N candidates から best 選択) | 同上 | ❌ 不可 |
| 2-stage SFT (CoT only) | 採用可能 | ✅ 採用済 (= stage 1) |
| LoRA + careful data curation | 採用可能 | ✅ 採用済 |
| RLVR (= deterministic verifier reward) | 採用可能 | ✅ **Phase 4 採用予定** |

### 5.2 我々のみが取れる upside path

1. **【MUST 検討】 Rejection Sampling SFT (= Phase 3.5)**:
   - Phase 1 SFT 済 model で N candidates 生成 → verifier で correct のみ keep → 再 SFT
   - NuminaMath stage 1.5 相当、 tool 不要
   - 効果見込: **+0.01-0.02 LB**、 effort M
2. **【SHOULD 検討】 Multi-seed adapter merge (= Phase 5)**:
   - 2-3 seed adapter 訓練 → peft `add_weighted_adapter` で merge
   - 公式 metric は adapter 1 つ前提だが post-training merge は可能
   - 効果見込: **+0.005-0.015 LB**、 effort S
3. **【MAY 検討】 GRPO (= Phase 4 optional)**:
   - Day 19 dry-run gate pass (= Kaggle T4 9h 完走 condition) のみ
   - 効果見込: +0.02-0.04 LB、 ただし失敗 risk 高、 effort L

### 5.3 我々の独自 strengths (= AIMO より強い側)

- **deterministic verifier-backed CoT**: AIMO は LLM-generated CoT、 我々は **Python solver の 100% 正解 trace** = data quality は AIMO より高
- **6 puzzle types specialization**: AIMO は math olympiad 全般、 我々は 6 fixed types で specialization
- **rel_tol=1e-2 寛容**: AIMO は exact match、 我々は 1% tolerance

---

## 6. Codex review HEAD 10 commits (= Worker E)

> 出典: codex-rescue agent 報告 (= 本 doc に verbatim 引用)
> 判定: **ALLOW with WARN × 4** (= BLOCK なし、 v1 走行継続 OK)

### 6.1 WARN 1: mamba_ssm CUDA build 互換性 (= 推定 OK)

`mamba_ssm==2.2.5` は sdist 配布、 torch 2.7.1+cu128 でのビルド成功は確認済 (= training 走行中)。 朝 import smoke test を Cell 2 末尾に追加候補。

### 6.2 WARN 2: dataset 形式 — OK

`_to_prompt_completion` の `prompt` / `completion` 形式は TRL 0.22.2 で正式 supported。 `completion_only_loss=True` が正しく honored。

### 6.3 WARN 3: monkey patch 副作用 — LoRA 非量子化なら 実害なし

`is_torchao_available = lambda: False` は torchao 量子化 path を壊すが、 我々は BF16 LoRA で量子化なし → 影響なし。 `list_repo_templates` 404 swallow も `additional_chat_templates` がある repo は通常 path、 OK。

### 6.4 🟠 WARN 4: メモリ予算 楽観 (HIGH)

- 30B BF16 weights ≈ 60-64GB + LoRA params + Adam states + activation (seq 2048) + CUDA workspace + fragmentation = **peak 70GB 台後半 の可能性**
- 対策: 朝 Cell 5 完了時に `torch.cuda.max_memory_allocated()` を log 確認
- 超過時の fallback: MAX_SEQ_LEN を 1536 に下げる v2 patch

### 6.5 🟠 WARN 5: silent label loss (HIGH)

- truncation 警告のみで続行、 length probe は chat template apply 後 token 数と不一致の可能性
- 朝 Cell 4 出力の `truncated@2048=X (X.X%)` 行を必ず確認、 > 10% なら問題

---

## 7. 朝 v1 LB 確定後の即実行 path (= 3 scenario)

### 7.1 SUCCESS path (LB ≥ 0.865)

1. **Phase 2 進入宣言** (= plan v3 §2)
2. v2 SFT data 設計開始 (= 以下を反映):
   - **Per-Category CV setup** (§3.2) で per-type 学習効果を測る
   - **Rejection Sampling SFT** (§5.2 #1) を検討
   - **Bit solver の leading-zero 統一** (§4.1) を v1.5 inference patch として先行投入
   - **Symbol/Equation solver の SFT data 追加** (= worker D 成果、 74 records、 verifier-backed)
3. 4-way ablation 並列 SFT (= A baseline / B answer-only / C solver-CoT v1 / D solver-CoT v1.5 with Symbol) を Colab Pro+ で kick off

### 7.2 PARTIAL path (LB 0.850-0.864)

1. **Targeted patch v1.5**:
   - logprob filter (§2.1) を SFT data に適用 → 再 SFT (= 8h)
   - bit solver leading-zero 統一 (§4.1)
2. 同時に v2 prep:
   - lr 2e-4 → 5e-5 へ降圧 (§2.2)
   - upsample 13x → 5x (§2.2)
   - grad accum 32 → 64 (§2.2)

### 7.3 ABORT path (LB < 0.850)

1. **戦略 pivot 宣言** (= plan v3 §0 LB Gate)
2. dgxchen baseline (= public 0.85) に revert、 何が壊れたかの postmortem
3. catastrophic forgetting 疑いが第一容疑 → upsample 13x が原因の蓋然性高
4. Phase 2 開始は 2-3 日遅延、 残 30 日で挽回

---

## 8. 出典 一覧

- discussion audit: docs/research/2026-05-14-discussion-audit.md
- codes audit: docs/research/2026-05-14-codes-audit.md
- metric deep audit: docs/research/2026-05-14-metric-deep-audit.md
- past comp winners: docs/research/2026-05-14-past-comp-winners.md
- equation DSL: docs/research/2026-05-14-equation-dsl.md + src/.../solver_symbol.py (= 74/1555 = 4.69% coverage、 98.65% precision、 commit 497fe40)
- Codex review: 本 doc §6
- 親 plan v3: ~/.claude/plans/iterative-dancing-pearl.md
- 親 CLAUDE.md (= 全プロジェクト原則): ~/.claude/CLAUDE.md
- 親親 CLAUDE.md (= Kaggle 共通): ~/projects/kaggle/CLAUDE.md
