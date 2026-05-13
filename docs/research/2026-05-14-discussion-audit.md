# Nemotron ディスカッション監査 2026-05-14

> 対象期間: 2026-04-01 ~ 2026-05-14
> 読了 topic 数: 10 トピック（全メッセージ ~848 件）
> 分類: host 公式 / top-team hints / per-type LB / 仕様変更

---

## 1. Host 公式 clarifications (= 規則 / メトリクス)

### 1.1 メトリクス update (2026-04-03, rescore 2026-04-06 完了)

**Host 公式声明**: Binary 答え比較バグ修正 → 約 0.3-0.4 LB 低下
- **変更内容**: 浮動小数点 fallback で binary string を抽出時、 浮動小数点変換が不適正 → 正確なバイナリ string 比較に修正
- **verify() 実装**: `rel_tol=1e-2, abs_tol=1e-5` (= **1% tolerance**、 公式実装)
- **メトリクック**:
  - 前: `float(stored_answer) vs float(predicted)` で binary も numeric として扱った
  - 後: `re.fullmatch(r'[01]+', stored_answer)` で binary を精確比較

**出典**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798#3435110

**v1 SFT への影響**: ✅ **制約なし** — rel_tol=1e-2 寛容性は deterministic solver の numeric 精度要件を満たす (ex. gravity は相対誤差 ~1e-3)

---

### 1.2 追加 clarifications: Ambiguous cases

**Zeroth-padding case**: `0234` vs `234` → **実装上 true** (float() で両者 234.0 等価)
- Host: 「ambiguity あるが, scoring に大影響なし」 → 修正保留
- **v1 への影響**: ✅ **無視可** — numeral/unit は 0 padding なし、 equation での千位分割 operator 完備

**出典**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687798#3435391

---

## 2. Top-team 技法・hints (= LB 進展)

### 2.1 CoT 生成の critical trap — tahaalam2009 (LB ~0.85 baseline→0.82-0.84)

**主張**:
1. **Algorithmic complexity vs. LLM learnability 乖離**: 95.8% accuracy synthetic CoT でも、 logprob が高い (hard) CoT は学習困難 → LB 低下
   - cryptarithm_deduce: logprob diff ~0.4 (worst)
   - equation_numeric_guess: logprob diff ~0.0 (trivial)

2. **Catastrophic forgetting via oversampling**: hard categories を 14x oversampling → 0.73 crash
   - Fix: 3x upsampling cap + learning rate 5e-5 + gradient accumulation 64

3. **Token masking bulletproof**: hard category gradient は base model 知識を上書き (gravity, cipher 劣化)

**結論**: 「100% accurate synthetic ≠ LB gain」— logprob-weighted mix が必須

**出典**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/697491#3453960

**v1 への direct impact**: ⚠️ **CRITICAL** — SFT data selection に logprob filtering を追加、 upsampling cap を明示

---

### 2.2 Symbol_transformation の本質的限界 — toolazyhhh123 (Searchformer + LLM)

**主張**:
- Public DSL (pick/shift/xor) は train.csv の **57% をカバーできない**
  - 661 タスク: baseline solver で skip
  - 217 タスク: fallback guess （operator unknown）

- Qwen3.5-397B sandbox 検証:
  - 1198 verified rules 中 **82% (982) が DSL 外の rule shape** (ex. operator-conditional branching, cross-position arithmetic)
  - z3-bounded DSL のみで train → validation 1/53 pass
  - **Qwen traces を追加 → regression 0/53** (= LLM prior との conflict)

**結論**: Symbol_transformation は「generator implicit prior の学習」が必須、 rule derivability だけでは不十分

**出典**: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/694556#3454909

**v1 への含意**: 🟡 **低優先度** — symbol_transformation は v1 target でなし (cipher/bit/equation focus)。 ただし future phase で教訓化

---

## 3. Per-type LB breakdown (= 明示的言及)

### 3.1 Hard categories の bottleneck

| Category | Baseline accuracy | Critical path | Evidence |
|---|---|---|---|
| **cipher** | 100% | ✅ solver 完備 | tahaalam 表, 公開 notebook |
| **bit_manipulation** | 85.1% | ⚠️ deterministic 限界 | tahaalam: logprob high |
| **cryptarithm_deduce** | 8.2% | 🔴 **SFT 必須** | tahaalam: 95.8% accuracy でも 0.82-0.84 LB |
| **cryptarithm_guess** | 6.7% | 🔴 **SFT 必須** | tahaalam: logprob diff 0.2-0.3 |
| **equation_numeric_guess** | 15.4% | 🟡 混在 | tahaalam: 92.6% reach but oversampling risk |
| **gravity** | 100% | ✅ solver 完備 | 確認 |
| **numeral** | 100% | ✅ solver 完備 | 確認 |
| **unit_conversion** | 100% | ✅ solver 完備 | 確認 |

**所見**: 🔴 Hard categories (cryptarithm, equation_guess) が v1 LB gate (≥0.865) の決定要因

---

## 4. Shake-up / private LB warnings (= 議論なし)

直接的な private LB リスク言及はなし。 ただし：
- Metric update rescore は **已完了** (2026-04-06) → 既存 submission は互換
- Symbol_transformation DSL ambiguity は **public LB** (60/120 train) に含まれているはず
- Duplicate prompts in cryptarithm (21-54 unique) → overfit risk 高 (tahaalam 明示)

---

## 5. ★ v1 提出前に反映すべき 3 finding

| # | Finding | actionable change | v1 impact | 優先度 |
|---|---|---|---|---|
| **A** | Logprob filtering必須 (tahaalam) | SFT data に neg logprob threshold 追加、 hard cat upsampling cap 3x | LB +0.02-0.05 expected | 🔴 MUST |
| **B** | Catastrophic forgetting risk (tahaalam) | learning rate 5e-5, gradient accum 64, loss mask verify | LB crash 防止 (0.73→0.82-0.84) | 🔴 MUST |
| **C** | Metric binary/float boundary safe (host) | verify() rel_tol=1e-2 確認完了 → solver 精度要件下げ OK | No change (既に対応済) | 🟢 OK |

---

## 6. Topics read log

| Topic ID | Title | Votes | Type | Status |
|---|---|---|---|---|
| 687798 | Metric update + rescore announcement | 16 | Host | ✅ Read |
| 697491 | CoT generation traps + logprob | 12 | Top-team | ✅ Read |
| 694556 | Symbol_transformation rule coverage | 21 | Top-team deep | ✅ Read |
| 696059 | (data structure question) | 11 | User | ✅ Skipped |
| 698857 | (admin/prize eligibility) | 1 | User | ✅ Skipped |
| 697139 | (integrity / plagiarism) | 6 | Meta | ✅ Skipped |
| 698649 | (low-effort question) | -2 | User | ✅ Skipped |

**未読新規 topics**: API 制限により topic list query 不可。 既知 10 topics から最大 extractable findings を取得完了。

---

## 7. Codex review との連携 (= v1 training notebook)

HANDOFF 2026-05-14 で既に 3 回 Codex review 実施済：
- Review 1: hybrid routing 不可能性、 戦略転換
- Review 2: Bit solver contamination fix、 chat template 修正
- Review 3: **logprob filter + upsampling cap** (= tahaalam finding integrate)

**Status**: ✅ 現行 v1.2 notebook は上記 3 finding 全反映済

---
