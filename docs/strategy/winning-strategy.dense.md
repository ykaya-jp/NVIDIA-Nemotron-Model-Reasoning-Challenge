# Winning Strategy v2 — Nemotron 2026-05-13 (Codex review 後)

> 親 plan: `~/.claude/plans/iterative-dancing-pearl.md` v2
> Codex review: `docs/strategy/codex-review-2026-05-13.md`
> 締切: 2026-06-15 (残 33 日)、 賞金 $106K、 2959 teams

## 1. 全体方針

**目的を「public LB max」 から 「private LB top percentile + 1 safety + 1 upside」 に reframe** (= Codex 推奨)。

- 銅 90% / 銀 70-80% / 金 20-25% / **優勝 1-3%** (= Codex calibrated)

## 2. 6 Stage (= S0 → S4)

| S | LB target | 施策 |
|---|---|---|
| S0 | 0.85 + **deployability proven** | 公開 kernel fork + T4×2 で 9h 完走実測 |
| S1 | 0.86 | + 3 deterministic + hybrid routing + cleaning |
| S2 | 0.865-0.87 | + Searchformer CoT (= **gated ablation 必須**) |
| S3 | 0.87 | + GRPO post-SFT (= **optional**、 Day 22 gate) |
| S4 | +0.005 | balanced-brace fix + self-consistency K=3 (= **offline timing pass のみ**) |

## 3. 6 puzzle types 戦略

| Type | Strategy | confidence | time cap |
|---|---|---|---|
| Roman / Physics / Unit | Python solver 移植、 99%+ | 100% | 0.5 日 each |
| Cipher | dictionary + frequency、 高 abstention | 70% | 1 日 |
| **Bit manipulation** | per-bit boolean enumerate、 **48h hard cap** | 60% | **48h MAX** |
| **Symbol** | high-precision only、 broad DSL 構築せず | 50% | 1 日 |

**v2 重要原則**: solver が confident wrong 返すより `None` 返して LLM fallback (= wrong abstention 防止)

## 4. Phase 構造 v2

```
P0+ Deployability (Day 0-2)  → 0.85 baseline + 9h 内 完走実測
P1  Solvers (Day 3-7)         → 6 types、 LB 0.86 ターゲット (Day 7 gate)
P2  Cleaning + 4-way ablation (Day 8-12) → Searchformer 効果 evidence
P3  SFT focused (Day 13-19)   → grid search、 LB 0.87 ターゲット (Day 19 gate)
P4  GRPO optional (Day 20-24) → Day 22 dry-run gate pass のみ
P5  Inference robustness (Day 25-29) → silent failure 防衛
P6  Final freeze (Day 30-33)  → safe + risky 2 sub
```

## 5. Codex 7 finding 防衛策

| Risk | 防衛策 |
|---|---|
| P1 time budget 楽観 | Bit 48h hard cap、 Symbol high-precision only |
| GRPO Kaggle 非互換 | Day 22 dry-run gate (= fail で廃棄、 浮いた 3 日 P5) |
| Searchformer +0.01 unverified | P2 4-way ablation、 LB lift 確証必須 |
| 4 comp 並走崩壊 | 既存 3 comp explicit maintenance mode until Day 25 |
| 5-8% 楽観 | calibrated 1-3% に修正 |
| Silent failure 6 種 | P5 Day 25-28 で全 audit |
| 公衆 LB overfit | private LB robust 優先、 safe sub = high abstention |

## 6. 既存 3 コンペの分担

私 (pane 1) = Nemotron 90%+ commit、 残 10% は emergency intervention only。 ROGII / orbit-wars / neurogolf は Day 25 (= 6/7) まで各 pane 中央が自走。

## 7. 5 つの critical gate

| Gate | Day | 判定 | Fail 対応 |
|---|---|---|---|
| Deployability | Day 2 | 30B + LoRA で 9h 内 完走 | 異なる base model に reset |
| LB 0.86 | Day 7 | hybrid routing で 0.86 達成 | Pivot or Abort |
| Ablation winner | Day 12 | 4-way で明確な勝者 | CoT 戦略 reset |
| SFT plateau | Day 19 | LB ≥ 0.86 | GRPO skip、 P5 集中 |
| GRPO dry-run | Day 22 | Kaggle 9h 内 完走 | GRPO 廃棄 |

## 8. Final submission (Day 30)

- **Safe**: P3 SFT + hybrid routing + balanced-brace fix + 高 abstention
- **Risky**: P4 GRPO + self-consistency K=3 (= offline timing pass のみ)

Risky は upper bound 狙い、 Safe は private LB shake-up 防御。

## 9. 私 (Claude) の核心 5 貢献

1. 6 types deterministic solver (= 3 移植 + 3 独自、 abstention 優先)
2. Searchformer CoT generator (= gated ablation 経由のみ採用)
3. GRPO orchestration (= optional)
4. Hybrid routing inference kernel
5. balanced-brace parser fix + silent failure 防衛
