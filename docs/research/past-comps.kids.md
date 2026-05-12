# 過去の類似コンペ — TL;DR 版

> nvidia-nemotron-model-reasoning-challenge 向け、2026-05-12 編集。
> 詳細は `docs/research/past-comps.dense.md`、構造化 DB は `past-comps.references.json`。

<!--
W3 が dense.md を書いた後に主要点だけ抜粋する短縮版。
高校生でも読める文体・図解多め。出典は dense.md の同 sect への内部リンクで OK。
-->

---

## どんなコンペが「似てる」か?

- **構造類似度 maximum**: < コンペ名 > — < なぜ似てるか 1 行 >
- **構造類似度 high**: < コンペ名 > × < 件数 >
- **構造類似度 medium**: < コンペ名 > × < 件数 >

---

## 過去の優勝者は何をしたか? (1 段落 × 3 件)

### < 構造類似度 maximum コンペ > の 1 位

- **手法**: < technique 1 行 >
- **すごい工夫**: < trick 1 + trick 2 を 1 行ずつ >
- **当該コンペでも使える?**: < はい / 部分的に / そのままは無理 — 1 行根拠 >

### < 構造類似度 high コンペ > の 1 位

(同形式)

### < 構造類似度 high or medium > の 1 位

(同形式)

---

## 「これは外せない」共通テクニック

過去の上位陣がほぼ全員やっていることリスト:

1. < trick 1 — 出典コンペ + なぜ重要か >
2. < trick 2 >
3. < trick 3 >

---

## 当該コンペでの推奨実装順 (= 段階的 recipe)

```
Week 1-2: < baseline >
Week 3-4: < 強化 >
Week 5-6: < ensemble / 高度化 >
Week 7-8: < submission 戦略 >
```

---

## 補足: 注意すべき過去事例

- **< 教訓的失敗例 >**: < 1 文 — 例: 「IL のはずが host が 10 億 frame 提供してたのに、リーダーボードは pure Python heuristic に勝たれた」 >
