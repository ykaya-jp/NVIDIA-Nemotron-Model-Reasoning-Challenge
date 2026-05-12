# Day 0 Checklist — Nemotron 2026-05-13

> 親 `~/projects/kaggle/CLAUDE.md` §2 mandatory checklist 適用。

## 1. 公式 docs 全 Read

- [x] Overview: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- [x] Data: train.csv (9498 puzzle、 6 types balanced) / test.csv (3 puzzle dummy)
- [x] Evaluation metric: accuracy (= `\boxed{<answer>}` exact-match)
- [x] Rules: code competition、 model artifact (= LoRA adapter) を submit、 Internet off
- [x] Timeline: 2026-03-14 start、 **2026-06-15 23:59 UTC** deadline (= 残 33 日)
- [x] Prize: $106,388 total (Top 4 split: $50K / $25K / $20K / $11K 推定)
- [x] Late submission: 不可 (= deadline 後の submit 無効)

## 2. 公式 module / starter code Read

- [x] `ryanholbrook/nvidia-nemotron-submission-demo` (= 2170 votes、 公式 submit テンプレ): 中身は `predictions.parquet` 出力する notebook
- [x] Metric notebook: `metric/nvidia-nemotron-metric` で `\boxed{}` regex 抽出ロジック確認、 5/8 fix 済 (= topic 698106)
- [ ] Contributors 欄に GM 名が居るか → 後で確認 (Day 1)

## 3. Leaderboard top 20 確認

LB snapshot `docs/research/lb-snapshot-2026-05-13.csv` 取得済 (300 teams):

| 順位 | チーム | LB | 備考 |
|---|---|---|---|
| 1-3 | MOONMOON / Researcher 7919 / Lora is all you need | **0.87** | top 0.5% Gold 候補、 手法 非公開 |
| 4-300 | (= 297 teams、 0.86 で footrace) | 0.86 | dgxchen 公開 kernel で 多数到達 |

公開 kernel 著者で LB 高い: huikang (= 「Lora is all you need」 著者可能性高)、 konbu17 (= 484 vote 公開)。

## 4. Discussion 全 topic 概観

vote 上位 12 件取得済 (`docs/discussion/raw/topic-*.csv`):

| Topic ID | votes | replies | 重要度 | 要約 |
|---|---|---|---|---|
| 681745 | 56 | 33 | 高 | Nemotron 公式 resources 一覧 (NeMo / HF / RLVR) |
| 698293 | 30 | 3 | 高 | **97.2% Symbolic Solver**: gold-conditioned oracle で equation_symbolic の latent structure 確認 |
| 694556 | 21 | 14 | 高 | **symbol_transformation で 57% は公開 DSL fit せず**、 induction 限界 |
| 698106 | 20 | 13 | 中 | **Metric update**: `\boxed{}` の `}` 多重ネスト fix (= 5/8 deploy) |
| 687798 | 16 | 19 | 中 | Rescore after metric update |
| 697416 | 15 | (sticky) | 中 | host welcome |
| 697491 | 12 | 10 | 高 | **Better dataset Scored Worse**: 95.8% accurate CoT が LB 下がる罠 |
| 697139 | 6 | 4 | 低 | Brain rot submissions 批判 |
| 696059 | 11 | 11 | 中 | square brackets 抽出問題 |
| 698857 | 0 | 1 | 高 | **proprietary teacher CoT 使用許可** 確認中 (= 5/12 開示、 host 回答待ち) |
| 698649 | -2 | 0 | 低 | Stuck at 0.56 |

## 5. Public kernel vote 上位 30 確認

`docs/research_kernels/` に 6 本 fork 済:
- `dgxchen_*` (337 vote): **0.85 LB レシピ確定**
- `huikang_*` (276 vote): 同上 別ルート
- `kalyankkr_*` (79 vote): 6 puzzle types decode + SFT data
- `mohankrishnathalla_*` (69 vote): **3 types deterministic solver (Roman / Physics / Unit、 99%+)**
- `konbu17_*` (484 vote): LoRA + CoT 標準
- `kienngx_*` (642 vote): CoT label 生成

## 6. CV strategy 設計 (= 親 §1.1 「good CV is half of success」)

| 軸 | 採用 |
|---|---|
| Split | **Stratified 5-Fold by puzzle type** (= 親 §3 tabular generic) |
| ratio | 80/20 train/val per fold |
| Metric | **exact-match accuracy** (= 公式 metric 同一)、 \boxed{} 抽出は公式 regex を local replicate |
| ratio drift 監視 | 5 submit 毎の calibration check (= 親 §8.1 #9 #10) |
| 注意 | Searchformer 流評価で「backtracking 込み」 trajectory の generalization を別 holdout で確認 |

## 7. 計算 budget 計画

- Colab Pro+ A100 24h × 33 日 = **理論 792h、 実用 600-700h**
- Phase 3 (LoRA SFT 5 試行): 50h
- Phase 4 (GRPO 2 試行): 100h
- Phase 5 (inference test): 20h
- 合計 ~170h、 余裕で予算内
- Kaggle GPU 週 30h、 月 120h (= submit 評価用)
- Submission quota: 5/day × 33 = 165 submit max

## 8. Prior 競技優勝解法 確認

過去類似 (= LLM reasoning + LoRA + ARC-like puzzle):
- ARC-Prize-2024 ARChitects (53.5% accuracy)
- ARC-AGI-2 (ongoing)
- 一般 LLM math reasoning: DeepSeek-R1 (= GRPO + RLVR)
- Code golf NN: NeurIPS 2025 Google Code Golf (= neurogolf-2026 と類似、 別案件)

→ 後日 `docs/research/2026-05-14-past-comps-deepdive.dense.md` 起票予定 (= Day 1)

## 9. 不明領域

- top 3 (MOONMOON / Researcher 7919 / Lora is all you need) の非公開手法
- proprietary teacher 許可 (topic 698857)
- private 30 / public 70 LB split の効果

## 10. Next action (Day 1、 = 5/14)

1. `docs/research/2026-05-13-mathematical-foundations.md` 起票
2. `docs/research/2026-05-13-local-vs-lb-correlation.md` 雛形作成
3. dgxchen kernel を fork 化して自前 submission v0 として Kaggle push → 0.85 LB 再現確認
4. /codex:adversarial-review で本 plan 全体 challenge
5. Phase 1 着手 (= Roman / Physics / Unit 3 deterministic solver 移植)
