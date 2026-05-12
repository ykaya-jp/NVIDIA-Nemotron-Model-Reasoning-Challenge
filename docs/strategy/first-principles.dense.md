# nvidia-nemotron-model-reasoning-challenge — 第一原理に基づく数学的解析 (dense)

> Worker C (W6 first-principles-deriver) + Worker D (W5 domain-knowledge-synthesizer)、2026-05-12 編集。
> 一次情報源: < engine source / metric definition / 公式 docs URL >

<!--
agent comp (rl) の場合:
  - engine source (`.venv/lib/python3.11/site-packages/kaggle_environments/envs/<slug>/<slug>.py`) を W6 が読み下す
  - 物理定数・state/observation 形式・reward 計算・action validation を closed form で整理
tabular / vision / nlp / timeseries の場合:
  - 評価指標の数式 (例: AUC, RMSE, F1, custom metric) を W6 が math 表記で書き下す
  - data spec (column dtype, target dtype, sample size, time span) を整理
  - W5 が domain SOTA / 教科書知識を並走で補う

すべての公式・定数に file:line または公式 docs URL を必ず付ける。
-->

---

## 0. 表記法・規約・特異点

| 記号 | 意味 | 出典 |
|---|---|---|
| | | |

**ストレージ上の癖 / 落とし穴** (= コードを書く前に必ず把握):

- < 例: 「planet record の slot 2 は X 座標、3 は Y 座標 (engine 内で sample → store で transposed)」 >
- < 例: 「target column は train.csv では int 型だが test sample submission は float 型」 >

---

## 1. 評価指標の数式定義

### 1.1 metric の closed form

<!-- public docs / engine source の数式をそのまま math 表記で。
     TeX 表記は `$ ... $` インライン、`$$ ... $$` ブロック。 -->

$$\text{metric}(y, \hat{y}) = \cdots$$

出典: < public docs URL > / `engine.py:NNN-NNN`

### 1.2 metric が依存する column / state

- < dependency 1 >
- < dependency 2 >

### 1.3 worst case / best case

- 理論最小値: < value > (条件: < ... >)
- 理論最大値: < value > (条件: < ... >)
- starter notebook の score: < value >

---

## 2. ゲーム / データの不変条件 (= 違反すると submission が silent fail / disqualify する)

<!-- agent comp:
     - action validation rule (forbidden moves が silent drop されるか)
     - timeout / step limit
     - resource constraint (ship max, RAM, etc.)
     tabular / nlp / vision:
     - submission file format
     - submission row count (= test set 行数と一致)
     - probability の合計が 1 になる必要があるか
     - leak の avoidance ルール -->

| 条件 | 違反時の挙動 | 出典 |
|---|---|---|
| | | |

---

## 3. 主要 quantity の閉形式 / 公式

<!-- agent comp 例:
     - fleet 速度公式: $v(N) = 1 + (v_{max}-1)(\ln N / \ln 1000)^{1.5}$
     - lead-shot 角度: $\theta = \arctan(\cdots)$
     - forbidden cone half-angle: $\alpha = \arcsin(R_\odot / D)$
     tabular / timeseries 例:
     - target encoding の bias correction
     - fold split の予測有効性
     - feature importance の信頼区間 -->

### 3.1 < quantity 1 >

$$\text{...} = \cdots$$

| < input > | < output > |
|---|---|
| | |

出典: < file:line >

### 3.2 < quantity 2 >

(同形式)

---

## 4. Domain 知識 (W5 出力)

### 4.1 当該 domain の最新 SOTA

<!-- 過去 1-2 年の論文 / blog / Kaggle writeup から、当該 domain の代表的 winning approach を整理。
     < arxiv URL >, < paper title >, < key technique > を箇条書き。 -->

- < SOTA 1 >: < 1 行説明 + URL >
- < SOTA 2 >

### 4.2 教科書 / standard reference の重要事実

- < ref 1 + 何が書かれているか >
- < ref 2 >

### 4.3 該当 lib / framework の docs から拾った非自明な仕様

<!-- Context7 経由で取得した該当 lib (sklearn, lightgbm, transformers, kaggle_environments 等)
     の最新 docs から、よくある誤解や non-obvious な API 挙動を抜粋。 -->

- < lib 1: 仕様 + 出典 URL >
- < lib 2 >

---

## 5. submission / agent template の制約

<!-- file size limit, memory limit, time limit per row/turn, internet 可否 -->

| 項目 | 制約 | 出典 |
|---|---|---|
| File size | | |
| Memory | | |
| Wall time per row/turn | | |
| Internet during inference | | |
| External data 可否 | | |

---

## 6. これらの第一原理から導かれる「やってはいけないこと」

- < anti-pattern 1 + 数式根拠 >
- < anti-pattern 2 >

---

## 7. これらの第一原理から導かれる「優位性の source」

- < advantage 1 + 数式根拠 >
- < advantage 2 >
