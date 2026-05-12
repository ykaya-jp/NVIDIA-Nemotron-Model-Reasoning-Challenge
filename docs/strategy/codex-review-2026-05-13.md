# Codex Adversarial Review — Nemotron Plan v1 (2026-05-13)

> 起動: `/codex:adversarial-review` via codex:codex-rescue subagent
> 対象: `~/.claude/plans/iterative-dancing-pearl.md` v1
> Codex 出力 (全文): `/tmp/claude-1000/-home-yusuke-kaya/12f2ee10-1e10-46ce-a0be-7e6eb2f6e4b7/tasks/a093ae98dc506f7bc.output`
> review 後の plan: v2 (= 親 plan で overwrite 済)、 本 repo `docs/strategy/winning-strategy.dense.md`

## 1. Codex finding summary (= 7 critical points)

### 1.1 P1 Time Budget for Deterministic Solvers
- **Probability**: High
- **Symptom**: Bit manipulation solver が「per-output-bit boolean function over 1-3 input bits」 model に fit せず、 train-fit ambiguity 高い。 Symbol 8 日だけで shallow placeholder 化、 P2 が weak traces で start
- **Fallback (v2 取込)**: Bit 48h hard cap、 60% CV 未達なら freeze。 Symbol broad DSL 構築せず、 high-precision pattern + aggressive `None`

### 1.2 GRPO + NeMo RL Kaggle 互換性
- **Probability**: Medium-high
- **Symptom**: NeMo RL GRPO が訓練できても、 Kaggle T4×2 (15GB×2、 unified ではない)、 9h 制限、 max_model_len 4096 で deploy 失敗、 OOM or 推論 timeout
- **Fallback (v2 取込)**: P4 GRPO を **optional candidate**、 Day 22 で dry-run gate (= 100 puzzle で 9h extrapolation pass) 必須、 fail なら廃棄して Day 23-24 を P5 に

### 1.3 Searchformer CoT +0.005-0.015 LB は unverified
- **Probability**: Medium-high
- **Symptom**: Synthetic trace で format 改善するが accuracy 悪化、 topic 697491 「better dataset scored worse」 と同型
- **Fallback (v2 取込)**: P2 で 4-way SFT ablation (baseline / answer-only / solver-CoT / hard-only-CoT)、 LB lift 確証なければ採用しない

### 1.4 4 コンペ並走 (= 70-80% Nemotron + supervision 3 comps)
- **Probability**: High
- **Symptom**: P3-P5 で booking 崩壊、 CV/LB notes lag、 adapter 命名 inconsistent、 Colab artifacts と Kaggle notebooks drift
- **Fallback (v2 取込)**: 既存 3 comps を **explicit maintenance mode until Day 25** (= 6/7)、 emergency only

### 1.5 5-8% 優勝確率は楽観
- **Probability**: High that 5-8% is optimistic
- **Calibrated estimate**: **1-3%** (= 2959 teams、 非公開 edge 再発見、 private LB variance survival、 5 hidden assumption の compound)
- **v2 取込**: 優勝 1-3%、 金 20-25%、 銀 70-80%、 銅 90% に下方修正

### 1.6 Silent Failure Modes (= 7 種)
1. **Evaluation mismatch**: Local exact-match と Kaggle boxed extraction が byte-level 不一致、 0.5-1 pp erase
2. **Public LB overfitting**: 0.860/0.870 sharp tier で repeated submission が public 特有 quirks に over-fit
3. **Synthetic CoT contamination**: Solver artifact / train-only identifier / answer-position shortcut が trace に混入
4. **Wrong abstention behavior**: Solver が confident wrong 返すと hybrid routing が pure LLM より悪化
5. **Inference timeout without score**: K=3-5 sampling × 30B on T4×2 × 9h で完走失敗
6. **Artifact reproducibility failure**: Colab A100 / NeMo RL config / LoRA adapter / quantized base / Kaggle offline dataset の drift
7. **Submission format bugs**: Escaped braces、 quotes、 pipes、 empty predictions、 duplicate IDs、 row-order mismatch、 non-deterministic extraction

**v2 取込**: P5 Day 25-28 を **silent failure 防衛 audit** に再構成、 evaluation byte-level 一致確認、 self-consistency offline timing gate、 artifact reproducibility check、 submission format full test

## 2. Codex Top 3 推奨 (= v2 で全て取込)

1. **Deployability testing を P0/P1 に前倒し**: 30B + LoRA inference を T4×2 で 9h extrapolation pass を最初に証明
2. **Additive LB 仮定を gated ablation に置換**: Searchformer、 GRPO、 balanced-brace、 K-sampling は各 evidence 必須
3. **P1 を high-precision solver + robust fallback に narrow**: Bit/Symbol を open-ended 研究にしない

## 3. v2 で対応した changes

| 元 plan の主張 | v2 修正 |
|---|---|
| 「Bit/Symbol 5-7 day 共通」 | Bit 48h hard cap、 Symbol high-precision only |
| 「GRPO mandatory P4」 | GRPO optional、 Day 22 dry-run gate |
| 「Searchformer +0.005-0.015 LB 期待」 | 4-way ablation で証拠ありの場合のみ |
| 「Nemotron 70-80% + 3 comp supervision 週 1-2」 | Nemotron 90%+、 3 comp maintenance mode until Day 25 |
| 「優勝 5-8%」 | 優勝 1-3% に下方修正 |
| 「P0-P6 標準 phase」 | P0+ deployability、 P1 narrow、 P2 ablation、 P4 optional、 P5 robustness、 P6 freeze |
| 「Silent failure は plan 内 risk 列挙」 | silent failure 7 種を P5 で full audit、 Day 25-28 で全 test |

## 4. Codex BLOCK ではない adversarial review の扱い (= 親 §0.5)

Codex review は **「軽微で押し切らない、 必ず address するか反証する」**。 本件は **address (= 全取込)** を選択、 反証なし。 v2 plan で全 finding を構造的に取り込み済。

## 5. 次の Codex review 予定 (= 親 §0.5 継続活用)

- Phase 4 GRPO config 確定前: `/codex:adversarial-review "challenge whether GRPO is worth Day 20-24 budget given dry-run risk"`
- Phase 5 inference kernel 完成前: `/codex:review --base main` で silent bug 探索
- 各 Phase の commit 前に `/codex:review`
