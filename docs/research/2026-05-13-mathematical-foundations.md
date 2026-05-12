# Mathematical Foundations — Nemotron 2026-05-13

> 親 `~/projects/kaggle/CLAUDE.md` §8.1 #2 mandatory: 問題定式化 + CV strategy + 損失関数 + 必要 component の理論整理。

## 1. 問題定式化

各 puzzle $p_i = (X_i, Y_i)$ に対して、 prompt $X_i$ から正解 $Y_i$ を出力する関数 $f_\theta$ を学習:

$$
\hat{Y}_i = f_\theta(X_i) = \text{Decode}(\text{LLM}_\theta(X_i))
$$

ここで $\text{Decode}$ は `\boxed{}` 内の文字列を抽出する関数。 評価は exact-match:

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{Y}_i = Y_i]
$$

$N$ = hidden test set size (公式 sample = 3、 実 test は数百〜数千と推定)。

## 2. 6 puzzle types の rule space

| Type | $X_i$ format | $Y_i$ format | rule space $\mathcal{R}$ |
|---|---|---|---|
| Roman | 「decimal → Roman 例 ≥ 2 個 + query decimal」 | Roman string | 単一 (= 標準 Roman) |
| Physics | 「(t, d) 例 ≥ 3 個 + query t」 | decimal (= 0.5gt²) | 1 自由度 (= $g \in [3.92, 19.6]$) |
| Unit | 「m → 別単位 例 ≥ 3 個 + query m」 | decimal (= ratio × m) | 1 自由度 (= ratio) |
| Cipher | 「plain → cipher 例 ≥ 3 個 + query cipher」 | plain text | 26! permutations + dictionary prior |
| **Bit** | 「8bit binary → 8bit binary 例 8-11 個」 | 8-bit binary | $2^{2^8} = 1.16 \times 10^{77}$ functions per bit、 ただし sparse (= AND/OR/XOR/rotate/shift) で減る |
| **Symbol** | 「symbol expr → symbol/numeric 例 1-3 個」 | symbol or numeric | DSL with depth ≤ 3 (= huge underdetermined) |

## 3. 理論下限 / 上限

| Bound | 値 | 出典 |
|---|---|---|
| 6 types すべて oracle (= gold-conditioned solver) | ~100% (= deterministic 3 types は 99-100%、 残り 3 types は oracle で +97%、 topic 698293 で equation_symbolic 97.2% 実証) | discussion 698293 |
| 公開 0.85 LB レシピ (= Unsloth + LoRA + CoT) | 0.85 | dgxchen / huikang 公開 kernel |
| Naive (= ベース AI zero-shot) | 0.40-0.50 推定 (= Roman / Physics / Unit のみ部分正解) | 6 puzzle types decoded kernel zero-shot 比較 |
| LB top 3 | **0.870** | LB snapshot |
| LB middle plateau | **0.860** (= 297 teams 飽和) | LB snapshot |

理論最大 (= oracle 全 6 types) に対して、 LB top 0.87 は **13% gap**。 この gap の正体は:
- (a) bit / symbol で rule induction が underdetermined (= 例から rule 完全特定不可、 多解答性)
- (b) proprietary teacher 使用許可前で synthetic CoT 質に上限
- (c) inference 1 sample で variance あり

## 4. CV strategy

- **Stratified 5-Fold by puzzle type** (= 親 §3 標準)
- 各 fold で 9498 / 5 ≈ 1900 val、 6 types で 各 ~315 puzzle val
- val accuracy が type 別に分かれているので、 **per-type ROI** を測れる
- Searchformer trajectory CoT の効果は別 holdout (= train 内 200 puzzle 固定) で測る (= train data に backtrack trace 入れた後 LB が落ちないか確認)

## 5. ratio (= LB / CV) drift 監視

- 各 submit 後 30 分以内に `docs/research/2026-05-13-local-vs-lb-correlation.md` に append
- 5 submit 累積で ratio σ を算出
- σ > 0.05 なら (a) host rule 変更 / (b) CV overfit のどちらか → investigate

## 6. 必要 component の理論整理 (= 私が実装するもの)

### 6.1 Type classifier
6 keyword (`numeral|gravity|m\s+becomes|encrypt|bit\s+manipulation|symbolic`) で route。 regex 軽量、 ML 不要。

### 6.2 Deterministic solver (3 types)
- Roman: 公開実装移植
- Physics: 最小二乗 g 推定 → query t に適用
- Unit: ratio 平均 → query m に乗算

### 6.3 Brute-force solver (Cipher / Bit / Symbol)
- Cipher: substitution mapping を例から抽出 + dictionary fallback
- Bit: 8 output bit × 1-3 input bit subset × 256 boolean functions の enumerate
- Symbol: DSL depth-2 program search

### 6.4 SFT (LLM)
- Loss: token-level cross-entropy
- Optimizer: AdamW with cosine LR (公開 kernel 標準)
- LoRA: rank 32, alpha 64, dropout 0.05

### 6.5 GRPO (Phase 4)
- Group size = 8
- Reward = `1.0` if `\boxed{<answer>}` exact-match else `0.0`
- KL coefficient β = 0.05 (DeepSeek-R1 推奨)
- Advantage normalization: per-group standardize

### 6.6 Inference
- Temperature 0 (deterministic) for Phase 3
- Temperature 0.7 + K=3-5 sample (self-consistency) for Phase 5
- balanced-brace parser で `\boxed{}` 抽出 (= 多重 } 対応)

## 7. 損失関数の理論的根拠

SFT cross-entropy は token-level、 ただし評価は `\boxed{}` exact-match で離散。
→ **token loss は exact-match の proxy にすぎない**、 GRPO で直接 reward 最適化が本筋 (= DeepSeek-R1 paper)。

## 8. 想定スコア試算

完全に楽観値だが理論的射程:

| 構成 | 想定 LB |
|---|---|
| Roman / Physics / Unit 100% + 残り 3 types zero-shot 20% | 3/6 × 100% + 3/6 × 20% = **60%** = LB 0.60 |
| + 残り 3 types public SFT (= dgxchen 0.85) | 0.85 |
| + 残り 3 types deterministic brute-force (Cipher 90% / Bit 70% / Symbol 50%) | 0.5 × 100% + 0.167 × 90% + 0.167 × 70% + 0.167 × 50% ≈ **85-86%** = LB 0.86 |
| + Searchformer CoT + LoRA | +0.01 = LB 0.87 |
| + GRPO | +0.01 = LB 0.88 |
| + self-consistency K=5 | +0.005 = LB 0.885 |
| + balanced-brace fix | +0.005 = LB 0.89 |

→ 理論的射程 0.88-0.89、 全 component mid case hit で **金メダル + 優勝候補**。
