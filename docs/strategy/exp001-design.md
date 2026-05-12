# exp001 Design — Phase 1 Synthesis from 6 Parallel Research Tracks

> **Status**: Phase 2-3 synthesis (= research summary + 構造原理が異なる 3+ 案).
> **判断は開発者**。中央 (= main agent) は推奨案を出さない (~/.claude/CLAUDE.md 中立指示原則 / 去好去悪).
> **実装は次プラン** (`/plan kaggle-nvidia-nemotron-model-reasoning-challenge-baseline`) で。
> **Date**: 2026-05-12

---

## 1. Context

< nvidia-nemotron-model-reasoning-challenge の概要、賞金、締切、チーム数、評価指標 >。

我々の現状: < 未提出 / 提出済 (rank N, score S) / starter only >。

< starter notebook / sample submission の score、上位陣との gap、構造的な勝てない理由を 1-2 段落で >。

---

## 2. Phase 1〜2 統合所見

### 2.1 W3 (past-comps) からの戦略的 prior

- 構造類似度 maximum: **< past comp >** ([source](< writeup URL >))
  - 1 位手法: < technique 1 行 >
- 構造類似度 high: < N 件 > ([source](`docs/research/past-comps.references.json`))
- **歴史的 winning paradigm**: < heuristic / IL / DRL / GBDT-stack / hybrid のどれが多い >
- 段階的 recipe: < phase 1-4 の 1 行ずつ >

### 2.2 W2 (公開 kernels) からの戦術ヒント

- 公開 kernel 上位 N 件のうち technique 別:
  - heuristic: < N >, IL: < N >, DRL: < N >, GBDT: < N >, ...
- **横断的に見える技法 3 つ**:
  1. < trick 1 — 出典: `public_kernels/<slug>/content.md` >
  2. < trick 2 >
  3. < trick 3 >

### 2.3 W6 (first-principles) からの数理的 invariant

- < 不変条件 1 + 出典 file:line >
- < 不変条件 2 >
- < silent fail trap 1 >

### 2.4 W4 (host dataset) からの実測値

- host dataset: < 公開あり / なし >
- DL 済: < N > 件、合計 < N > 行
- **真の主因** (実測由来): < 1 文 + 出典 `docs/research/host_datasets.md:<section>` >

### 2.5 W1 (discussion) からの fresh insight

- 主催者開示: < 重要 announcement 1 行 >
- 上位 player 開示: < public hint 1 行 >
- 重大 caveat: < engine bug / metric 変更 / late shake-up 警告 — 該当する場合 >

### 2.6 W5 (domain 知識) からの SOTA & 教科書

- 2024-2025 SOTA: < 1 行 >
- 関連 paper: < URL >
- 教科書: < 該当 ref >

---

## 3. exp001 候補 — 構造原理が異なる 3+ 案

各候補は **異なるパラダイム**。同じ paradigm 内の param tuning は「バリエーション」なので除外。

### 候補 1: < 名前 (= paradigm 1) >

**Provenance** (= 着想の出典):

- < past_comp_id >: < 1 行 + URL >
- < public kernel slug >: < 1 行 + path >
- < first-principles 由来の根拠 >

**Mechanism** (= 構造原理):

- < step 1 >
- < step 2 >
- < step 3 >

**Phase 1 知見の活用**:

- W2 (kernels) の < trick > を < どう活かすか >
- W3 (past-comps) の < technique > を < どう活かすか >
- W6 (first-principles) の < invariant > を < どう守るか >

**Failure modes** (= 失敗シナリオ — Phase 4 critic に独立検証させる):

- < failure 1 + likelihood + impact >
- < failure 2 >

**Disqualifying conditions** (= この案を「やめる」べき条件):

- < 条件 1 (例: host dataset N < 1000 行で IL pretrain は不能) >
- < 条件 2 >

**コスト & 期待**:

- 開発: < 日数 >
- compute: < CPU / GPU >
- 期待 score lift: < +N% / +ELO >

---

### 候補 2: < 名前 (= paradigm 2) >

(同形式)

**Provenance**:

**Mechanism**:

**Phase 1 知見の活用**:

**Failure modes**:

**Disqualifying conditions**:

**コスト & 期待**:

---

### 候補 3: < 名前 (= paradigm 3) >

(同形式)

**Provenance**:

**Mechanism**:

**Phase 1 知見の活用**:

**Failure modes**:

**Disqualifying conditions**:

**コスト & 期待**:

---

## 4. 評価軸 (中立、開発者判断用)

| 軸 | 候補 1 | 候補 2 | 候補 3 |
|---|---|---|---|
| **実現可能性** | | | |
| **開発期間** | | | |
| **必要 compute** | | | |
| **期待 score lift** | | | |
| **メダル / Top X 確率** | | | |
| **Phase 1 トラック活用度** | | | |
| **失敗時の learning** | | | |
| **次段階への接続** | | | |

---

## 5. 段階的アプローチ案

```
Week 1:    候補 < N >  → < 期待結果 >
   ↓ baseline 確立 (= 候補 < M > の opponent / pretrain source)
Week 2:    候補 < M >  → < 期待結果 >
   ↓ < 中間成果物 >
Week 3-4:  候補 < L >  → < 最終 score 目標 >
```

ただし **段階を全て踏まずに途中で止める** のも合理的:

- 「短期で順位上昇」→ 候補 < N > のみで停止
- 「メダル必須 / 時間ある」→ 候補 < L > ベース、Week 1-3 で段階移行
- 「学びたい / 失敗してもいい」→ 候補 < M > のみ

---

## 6. 中立判断のために (中央は推奨しない)

| もし開発者が ... | 候補 |
|---|---|
| **時間限られる、確実な順位上昇が欲しい** | 候補 < ? > |
| **メダル取りに本気、時間あり、< compute > を使い倒せる** | 候補 < ? > |
| **新手法学習が目的、勝敗より試したい** | 候補 < ? > |
| **判断保留、まず 1 だけ試してから決める** | 候補 < ? > → 結果見て他を選ぶ |

---

## 7. Out of scope (このプランではやらない)

- 候補のうちどれを実装するかの**選択** (= 開発者の判断、次プランで `/plan kaggle-nvidia-nemotron-model-reasoning-challenge-baseline` を起動する際に決定)
- 候補 < L >/< M > の場合の RL / NN ハイパーパラメタ詳細チューニング (実装プランで)
- LB top 1 を抜く戦略 (= top-X 圏 < score > を目標とする)
- < 必要に応じて他の out of scope >

---

## 8. 参照ファイル一覧 (Phase 1 deliverable map)

### Research (Track A)
- `docs/research/past-comps.dense.md`
- `docs/research/past-comps.kids.md`
- `docs/research/past-comps.references.json`
- `docs/research/public_kernels/INDEX.md`
- `docs/research/public_kernels/<slug>/content.md` × < N > 件

### Observation (Track B)
- `docs/research/lb-observations.dense.md`
- `docs/research/lb-observations.kids.md`
- `docs/research/lb_snapshot_2026-05-12.csv`

### Strategy (Track C)
- `docs/strategy/first-principles.dense.md`
- `docs/strategy/first-principles.kids.md`

### Host Data (Track D)
- `docs/research/host_datasets.md`
- `data/external/<dataset>/`
- `data/processed/<derived>.parquet` (該当する場合)

### Discussion (Track E)
- `docs/discussion/insights.md`
- `docs/discussion/<date>.md` × < N > 件
- `docs/discussion/topics.json`

### Synthesis (中央)
- `docs/strategy/exp001-design.md` (= this file)

---

## 9. 次プランへの引き継ぎ事項

開発者は次のセッションで **候補 1/2/3 のいずれか (or その組み合わせ) を選択** し、`/plan kaggle-nvidia-nemotron-model-reasoning-challenge-baseline` で実装プランを立てる。
その際:

- このファイル (`exp001-design.md`) を Read で全文読み直す
- 該当候補の "Phase 1 知見の活用" 列に書かれた dense.md セクションを Read
- `docs/research/host_datasets.md` の EDA 結果を Read で再確認
- `make tournament` / `make baseline` 等の使い方を `docs/dev/experiment-mgmt.md` (該当する場合) で確認

実装中に新しく見えてくる事 (公開 notebook の追加情報、新規 LB 観察等) は **新しい research dense.md** として追記し、本ファイルは synthesis スナップショットとして固定する。

---

<!-- 以下、Phase 4 で critic agent 出力が append される -->
