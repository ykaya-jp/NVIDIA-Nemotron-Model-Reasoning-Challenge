# 過去の Kaggle 類似コンペ — nvidia-nemotron-model-reasoning-challenge 向け密度高めの統合分析

> Track A の調査成果物。nvidia-nemotron-model-reasoning-challenge ({{COMP_TAGLINE}}, {{COMP_PRIZE}}, {{COMP_DEADLINE}}, {{COMP_NUM_TEAMS}} チーム) 向けに 2026-05-12 にまとめた。
> すべての主張は `file:line` (ローカル) または完全な URL (Web) で出典を明示。直近のコンペほど重みを高めている。**< 構造類似度 maximum と判断したコンペ >** は構造的に最も類似した過去コンペとして特筆。

<!--
W3 (past-comp-researcher) の主出力。orbit-wars `past-comps.dense.md` を雛形に使う。
原則:
- 構造類似度 maximum を必ず 1 件特定する。完全新規 domain なら non-Kaggle 競技 / 学術 paper を載せる
- 各 sect: writeup URL / repo URL / 技法 / compute / non-trivial tricks / 当該コンペへの転用
- 全 URL は WebFetch で 200 OK 確認してから記載 (404/403/503 は status 列に書く)
- 本文は日本語、コード識別子・数式・URL・file:line は英数字のまま
-->

---

## TL;DR マトリクス

| コンペ | 年 | チーム数 | 優勝手法 | 計算資源 | 当該コンペとの構造的一致度 |
|---|---|---|---|---|---|
| < 候補 1 > | < year > | < N > | < technique > | < compute > | < low/medium/high/maximum > |
| < 候補 2 > | | | | | |
| < 候補 3 > | | | | | |

**要点となる横断的観察**: < heuristic vs IL vs RL の歴史的優位性、metric 系統別の傾向、当該コンペでの推定 winning paradigm を 1 段落で >

---

## 1. < 構造類似度 maximum コンペ >

- コンペページ: < URL >
- < 主催者 paper / engine repo URL >

**1 位 — < team name >:**

- Writeup: < URL >
- Repo: < URL >
- 手法: < 技法を 1-2 文で >。引用: *"< writeup quote >"* ([source](< URL >))
- アーキテクチャ: < 詳細 >
- 計算資源: < compute >

**非自明なテクニック** (当該コンペ転用候補):

1. **< trick 1 >** — < 説明 + writeup quote >
2. **< trick 2 >**
3. **< trick 3 >**

**当該コンペへの転用 (約 150 word)**:

- < 直接移植可能な点 >
- < 修正が必要な点 (= ドメイン固有制約) >
- < ROI 高い順に推奨実装順 >

---

## 2. < コンペ 2 >

(以下、同フォーマットで構造類似度高めの順に)

---

## N. 横断的シンセシス

### N.1 heuristic が勝った事例

- < bullet — コンペ名 + 年 + 1 行根拠 >

### N.2 IL / 模倣学習が勝った事例

- < bullet >

### N.3 DRL が勝った事例

- < bullet >

### N.4 トレンドと当該コンペへの含意

< 1-2 段落: 当該コンペは partial-observability / 完全観測 / 静的 tabular / image / time-series のいずれか、metric は何か、histortical pattern と照らしてどの paradigm が勝ちそうか。**結論を断定しない**、選択肢を提示。 >

### N.5 当該コンペでの transferable recipe (段階的アプローチ案)

```
Phase 1 (week 1-2): < heuristic / baseline GBDT / starter notebook 改良 >
Phase 2 (week 3-4): < NN-based 強化 or feature engineering 拡張 >
Phase 3 (week 5-6): < ensemble / stacking / pretrain finetune >
Phase 4 (week 7-8): < self-play / pseudo-label / TTA / submission 戦略 >
```

**Mandatory tricks (どの comp も上位がやってる)**:

- < trick 1 — 出典コンペ >
- < trick 2 >

---

## 出典・参照

- 本ファイルの構造化版: `docs/research/past-comps.references.json`
- TL;DR: `docs/research/past-comps.kids.md`
