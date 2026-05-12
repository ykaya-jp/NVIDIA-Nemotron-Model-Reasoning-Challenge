# nvidia-nemotron-model-reasoning-challenge — Host 公開 dataset 一覧と EDA

> Worker B / W4 (host-dataset-analyzer) 出力。2026-05-12 編集。
> **CRITICAL**: ~/.claude/CLAUDE.md "[2026-05-10] orbit-wars lesson" に基づき、本ファイルの 「Host dataset 一覧」セクションは `kaggle datasets list -s nvidia-nemotron-model-reasoning-challenge` の出力をそのまま貼ることが skill QG-1 の必須条件。EDA セクションは host dataset が存在する場合に最低 1 day 分の DL + 分布実測 (N≥1000 行) を含むことが QG-2 の必須条件。

<!--
このファイルは 2 段構成:
1. CLI 出力 (= Phase 0 で main agent が冒頭に貼る、上書き禁止)
2. EDA (= Phase 2 で W4 が分析結果を埋める)
-->

---

## 1. `kaggle datasets list -s nvidia-nemotron-model-reasoning-challenge` の出力 (Phase 0 取得)

```text
{{KAGGLE_DATASETS_LIST_OUTPUT}}
```

取得日時: 2026-05-12 {{TIMESTAMP}}

<!-- 出力が空 (= host が公開 dataset 出していない) 場合は、ここに「公開 dataset なし」と明示する。
     W4 はこの場合 EDA を skip し、QG-2 は skip 判定で OK。 -->

---

## 2. ダウンロード済 dataset の inventory

| Dataset slug | 規模 | パス | DL 日 | License | 用途 |
|---|---|---|---|---|---|
| `<owner>/<dataset>` | < N MB / N rows > | `data/external/<dataset>/` | 2026-05-12 | < CC0-1.0 / 他 > | < 戦略分析 / IL pretrain / 等 > |

---

## 3. EDA (W4 出力)

### 3.1 schema / 列構成

<!-- 各 dataset の主要 file (parquet/csv/json) の columns + dtype + sample value
     - ここで「N=<実測値> 件」を必ず明示 (QG-2) -->

| file | rows | columns | 主要 column 例 |
|---|---|---|---|
| `<file>` | `<N>` | `<C>` | `<col_a>: <dtype>, <col_b>: <dtype>, ...` |

### 3.2 主要分布 (≥1 軸の実測)

<!-- 必須: 仮説の方向性を data で確証する分布を最低 1 軸書く。
     例:
     - agent comp: actions 分布、player_id 分布、reward 分布
     - tabular: target distribution、各 feature の missing rate / cardinality
     - vision: image size 分布、aspect ratio、class balance
     - timeseries: 時系列の trend / seasonality / horizon span
     - nlp: text length 分布、language 分布、label balance -->

```python
# 再現コマンド (uv run python -c "...") が示せる形で書く
```

| 軸 | p10 | p25 | p50 | p75 | p90 | p99 | 平均 |
|---|---|---|---|---|---|---|---|
| < 軸 1 > | | | | | | | |
| < 軸 2 > | | | | | | | |

**所見**: < この分布から読み取れる仮説の方向性 1-2 行 >

### 3.3 host dataset と train/test の関係

<!-- host が「過去の player の actions 全部」「上位 10% の replay」「銀河マップ全パターン」のような
     付随 data を出すことがある。これらは BC pretrain や feature engineering の元 data になる。
     ここで「train.csv との重複の有無」「test.csv の真値が含まれていないか」を確認する。 -->

- < 確認項目 1 >
- < 確認項目 2 >

### 3.4 重要な caveat

<!-- 例: 「2026-04 後半の engine bug 修正前 dataset は壊れた episode を含む、IL は 04-30 以降に絞れ」
     のような取り扱い注意事項 -->

- < caveat 1 >

---

## 4. 戦略への直接含意

| # | 仮説 | dataset 由来の確証データ | Phase 1 への含意 |
|---|---|---|---|
| 1 | | | |
| 2 | | | |

---

## 5. 未取得 / 巨大すぎて DL skip した dataset

| Dataset | 規模 | skip 理由 | 取得が必要になる条件 |
|---|---|---|---|
| | | (例: > 20GB のため Phase 1 では skip) | (例: IL pretrain phase に入った時) |
