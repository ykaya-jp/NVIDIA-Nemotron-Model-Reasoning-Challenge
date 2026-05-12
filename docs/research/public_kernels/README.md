# 公開 kernel カタログ

> 2026-05-12 編集。
> Pull スクリプト: `~/.claude/skills/kaggle-onboard/tools/pull_kernels.sh`
> 元 manifest: `kaggle kernels list -c nvidia-nemotron-model-reasoning-challenge --csv` (vote 上位 30 ∪ score 上位 30)

<!--
W2 (kernel-taxonomist) の主出力。
- 各 kernel ごとに `<kernel-slug>/content.md` (要点化) + `<kernel-slug>/kernel-metadata.json` (raw)
- このファイル (= INDEX.md) は全 kernel の「technique 別索引」
- 上位選手は public kernel を出さないことが多い (= LB top 1 の解法が含まれるとは限らない)
-->

## INDEX (technique 別)

### A. Heuristic / Rule-based

| kernel | author | vote | score (LB) | 1 行特徴 |
|---|---|---|---|---|
| `<slug>` | `<author>` | `<N>` | `<score>` | < trick の 1 行要約 > |

### B. Imitation Learning / 模倣学習

(同形式)

### C. Deep RL (PPO / DQN / IMPALA / etc.)

(同形式)

### D. GBDT (LightGBM / XGBoost / CatBoost) Stack

(同形式)

### E. Transformer / Sequence model

(同形式)

### F. CNN / Vision

(同形式)

### G. EDA-only / 未提出

(同形式)

### H. その他 / 分類保留

(同形式)

---

## 横断的な技法発見

<!-- W2 が pull した kernel から繰り返し見える technique を抽出 -->

- **< 技法 1 >**: < kernel A, B, C で共通。出典必須 >
- **< 技法 2 >**: < ... >

---

## 注意事項

- 公開 kernel の score は LB の「公開 baseline 帯」を反映するが、private LB top の解法は通常 public 化されない
- 著作権は各 kernel author に帰属。当該コンペでの利用は Kaggle Terms of Service に従う
- pull した raw notebook は `<kernel-slug>/<title>.ipynb` または `.py` として残す (再現性のため)
