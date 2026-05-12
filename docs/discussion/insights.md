# nvidia-nemotron-model-reasoning-challenge 戦略 insights

> 作成日: 2026-05-12
> 情報源: discussion ({{DISCUSSION_FILES}}, kaggle CLI で取得した topic {{TOPIC_IDS}}) + host dataset ({{HOST_DATASETS_USED}}) 自家分析

<!--
このファイルは W1 (discussion-miner) と W4 (host-dataset-analyzer) の合流地点。
W1 は discussion から「事実」を抽出し、W4 は host dataset から「実測値」を出す。
このファイルは両者を統合した「真の主因」を一段抽象化して書く。
すべての claim には [source](URL) または `discussion/<date>.md:line` 引用、または
`docs/research/host_datasets.md:<section>` への file 引用を必ず付ける。
-->

---

## 1. データ駆動で判明した「真の主因」

<!-- W4 の host dataset 実測 + W1 の discussion 仮説をクロス検証して書く。
     orbit-wars 例 (insights.md:8-22): bovard 2.82M row 分析で
     真因が「Capacity gap」でなく「Expansion gap」(= step 0-100 の planet 占領数 100 倍差)
     と判明した、のような quantitative claim を最低 1 つ書く。 -->

(空 — 埋めろ)

### 1.1 < 主因の定量的な記述 >

| < 軸 > | < player A > | < player B > | < ... > | **< 我々 (推定) >** |
|---|---|---|---|---|
| | | | | |

**所見**: < quantitative claim を 1-3 行 >。

**真の主因は「< 通説で考えがちな主因 >」ではなく「< 実測でわかった主因 >」**。

出典: < `docs/research/host_datasets.md:<section>` または `discussion/<date>.md:<line>` >

### 1.2 < 行動 / アクションの分布 >

<!-- 上位プレイヤーの actions / launch-size / submit-frequency / class-balance などの
     具体的分布を p10/p25/p50/p75/p90/p99 などの quantile で並べる -->

| < 軸 > | p10 | p25 | **p50** | p75 | p90 | **p99** |
|---|---|---|---|---|---|---|
| | | | | | | |

### 1.3 戦略の多様性

<!-- 単一最適戦略があるのか / 複数戦略が共存するのかを実測から判断 -->

---

## 2. ディスカッション横断で繰り返し言及されている事項

<!-- W1 worker 出力。topic_id ごとに `discussion/<date>.md` に raw dump があるはず。
     ここでは「複数 topic に共通する論点」を抽出する。 -->

### 2.1 主催者 / Host が明示的に開示した制約・データ

- < bullet 1 — 主催 announcement / pinned topic 由来。出典必須 >
- < bullet 2 >

### 2.2 上位プレイヤー / 公開 notebook 著者が明かしている技法

- < bullet — 公開 kernel author の forum コメント等。出典必須 >

### 2.3 重大な caveat (engine bug / metric 変更 / late shake-up 警告 / data leak)

<!-- orbit-wars 例 (insights.md:69-73): sweep_fleets overshoot bug, 2026-04 後半に修正済 →
     BC 訓練は 2026-04-30 以降の dataset に絞るべき -->

- < bullet — 該当する場合 >

### 2.4 重要な観測可能性 / 副次情報の指摘

<!-- orbit-wars 例 (insights.md:76-83): 敵 fleet の angle が完全 observable -->

---

## 3. 戦略への直接含意

| # | 改善案 | 解決する gap | 出典 |
|---|---|---|---|
| | | | |

---

## 4. 重要な未検証事項

<!-- W2-W6 で確認できなかったが、確認しないと plan に進めない事項を列挙 -->

- [ ] < 未検証事項 1 >
- [ ] < 未検証事項 2 >

これらは Phase 1 統合所見 → exp001 設計の前に追加調査する。

---

## 5. データソース管理

| ソース | パス | 規模 | 用途 |
|---|---|---|---|
| host dataset (top) | `data/external/<dataset>/` | < N row > | 戦略分析 |
| 自分の試合 / submission | `data/replays/*.json` (rl) または `submissions/.history.csv` (他) | < N > | baseline / debug |
| extracted features | `data/processed/<derived>.parquet` | < N row > | 統計分析 |
| user dump discussion | `docs/discussion/<date>.md` | manual | 主情報源 |

---

## 6. discussion 自動巡回の限界

<!-- orbit-wars insights.md:152-158 で確立された fallback 順序を継承 -->

- Kaggle discussion ページは React SPA で **WebFetch / curl で content 取れない**
- `kaggle competitions topic-messages <topic_id>` は動くが topic_id 発見が手動
- Meta-Kaggle ForumTopics は新コンペ topic を網羅していない (snapshot が古い)
- → ユーザーによる `docs/discussion/<date>.md` への手動 dump が引き続き主情報源
- 取得 topic id がわかれば `kaggle competitions topic-messages <slug> <id>` で content 取得可能
