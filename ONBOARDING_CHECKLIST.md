# nvidia-nemotron-model-reasoning-challenge — Onboarding Checklist

> kaggle-onboard skill の進捗 ledger。2026-05-12 編集。
> このファイルは skill (Phase 0〜4) と quality gate が更新する。手動編集は最後の「人手 review」セクションのみ。

---

## 6 軸 × 各 worker の達成状況

| # | 軸 | 担当 | 主出力 | 状態 | blocker |
|---|---|---|---|---|---|
| 1 | Discussion mining | W1 | `docs/discussion/insights.md`, `<date>.md`, `topics.json` | ⬜ pending | - |
| 2 | Public kernel taxonomy | W2 | `docs/research/public_kernels/INDEX.md` + `<slug>/content.md` × N | ⬜ pending | - |
| 3 | Past similar comps | W3 | `docs/research/past-comps.{dense,kids}.md`, `references.json` | ⬜ pending | - |
| 4 | Host dataset analysis | W4 | `docs/research/host_datasets.md` の EDA 部 | ⬜ pending | - |
| 5 | Domain knowledge synthesis | W5 | `docs/strategy/first-principles.dense.md` の domain 知識 sect | ⬜ pending | - |
| 6 | First-principles derivation | W6 | `docs/strategy/first-principles.{dense,kids}.md` の数式部 | ⬜ pending | - |

状態: ⬜ pending → 🔄 in_progress → ✅ done → ❌ failed (再走 1 回後も fail)

---

## Quality gate (QG-1〜9) 結果

`~/.claude/skills/kaggle-onboard/tools/quality_gate.py` 出力。

| QG | 検査内容 | 結果 | 失敗時の gap |
|---|---|---|---|
| QG-1 | host_datasets.md に CLI 出力あり | ⬜ | - |
| QG-2 | host dataset 公開ありなら 1 day DL + EDA N≥1000 | ⬜ | - |
| QG-3 | past-comps.references.json 最低 3 件 | ⬜ | - |
| QG-4 | public_kernels/<slug>/content.md 各々に source URL | ⬜ | - |
| QG-5 | discussion/insights.md 章ごとに source 引用 | ⬜ | - |
| QG-6 | 各 markdown の本文行 日本語比率 ≥ 70% | ⬜ | - |
| QG-7 | exp001-design.md に 3+ 案、各案 4 sect | ⬜ | - |
| QG-8 | critic 反論 ≥ 6 objection (likelihood/impact/detectability tag 付) | ⬜ | - |
| QG-9 | このファイルの 6 軸全部 ✅ | ⬜ | - |

---

## 自己検証 checklist (Phase 0 と Phase 4 で 2 回確認)

- [ ] `kaggle datasets list -s nvidia-nemotron-model-reasoning-challenge` を最初に叩いた (出力 host_datasets.md 冒頭に保存)
- [ ] host dataset 公開あり場合、最低 1 day 分の actual download を完了
- [ ] その data で **最低 1 軸の分布** を実測 (host_datasets.md § 3.2 に明記)
- [ ] `host_dataset_analyzer` worker 出力が「N=<実測値> 件」のように具体的サンプル数を明示
- [ ] `past-comps.references.json` に「現コンペと最も構造が近い過去コンペ 1 件」を `structural_relevance: maximum` でマーク
- [ ] `public_kernels/INDEX.md` で kernels を技法別に分類
- [ ] `first-principles.dense.md` で評価指標の数式定義 + 引用元コミット / 公式 docs URL を記載
- [ ] critic 反論 ≥ 6 objection
- [ ] uncommitted 変更なしで skill 開始

---

## 人手 review (任意)

skill 完了後にユーザーが見つけた gap や追加依頼:

- [ ] < 例: 「W3 が Halite II 入れてるのに reCurs3 writeup の Coulomb-style flee の記述が薄い、追記」 >
- [ ] < ... >

---

## handoff log

skill 完了時の transition 記録:

| 時刻 | from | to | 移行物 |
|---|---|---|---|
| 2026-05-12 {{TIMESTAMP}} | kaggle-onboard | {{NEXT_STEP}} | exp001-design.md, ONBOARDING_CHECKLIST.md |
