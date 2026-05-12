# nvidia-nemotron-model-reasoning-challenge — リーダーボード & 上位 submission 観察 (dense)

> Worker B 調査、スナップショット 2026-05-12。
> 我々の現状: < submission ID / team name / 現 score / 現 rank >。
> < engine 参照 / 公式 docs URL >

<!--
W3 + main の合流地点。
- agent comp (rl): top 10 bot の replay を逆算して戦術 classify
- tabular: top kernel の technique / feature engineering / model class を抽出
- vision/nlp: top notebook の augmentation / pretrain / loss を抽出
- timeseries: top kernel の splits / target encoding / horizon handling を抽出
すべての claim に [source](URL) または `lb_snapshot_<date>.csv:<row>` 引用。
-->

---

## 1. リーダーボード Top N (スナップショット 2026-05-12)

ソース: `kaggle competitions leaderboard nvidia-nemotron-model-reasoning-challenge -s --csv --page-size 200` →
`docs/research/lb_snapshot_2026-05-12.csv`。

| Rank | Team | Score | Last Submission |
|---|---|---|---|
| 1 | < team > | < score > | < timestamp > |
| 2 | | | |
| 3 | | | |
| ... | | | |
| 30 | | | |

分布の観察:

- < score の分布 — top 10 の集中度、長い裾の有無 >
- < submission frequency — 上位は提出を絞る vs 大量提出 >
- < timing — 締切前のシャッフル予兆 >

---

## 2. 上位 N (公開 sub があれば) の戦術逆算

<!-- agent comp (rl):
     - replay 取得: `kaggle competitions episodes <sub_id>` → `data/replays/`
     - 逆算: action 分布 / role 分担 / 開幕戦略 / 終盤戦略
     tabular:
     - 公開 kernel との一致: technique, feature set, CV strategy
     - LB / private の overfitting 兆候
     vision/nlp:
     - augmentation / model size / train data 量 -->

### 2.1 < team / kernel 1 > の特徴

- **手法 (推定)**: < technique 1 行 >
- **特徴的な数値**: < quantitative claim + 出典 >
- **失敗するパターン**: < どんな相手 / 入力に弱いか >

### 2.2 < team / kernel 2 >

(同形式)

---

## 3. 失敗モード (= 我々の sub が上位に負ける構造的理由)

<!-- orbit-wars 例 (lb-observations.dense.md): Capacity gap / 500-step starvation /
     Swarm-overrun / No sun-rejection / No fleet aggregation の 5 つ -->

| # | 失敗モード | 兆候 | 対策 (Phase 1 へのインプット) |
|---|---|---|---|
| 1 | < mode > | < observable signal > | < strategy hint > |
| 2 | | | |

---

## 4. 公開 leak / 公開 notebook からの戦略ヒント

<!-- 公開 kernel author が discussion や code comment で明かしている技法を集約 -->

- **< author 1 (kernel slug) >**: < trick > (出典: `docs/research/public_kernels/<slug>/content.md:<line>`)
- **< author 2 >**: < trick >

---

## 5. submission strategy の含意

- < 1 日あたり提出枠 N、検証コスト見積 >
- < private LB shake-up 想定の対策 >
- < 締切直前の提出選択の方針 >
