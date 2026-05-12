# nvidia-nemotron-model-reasoning-challenge project-local instructions

> Kaggle competition: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
> Deadline: 2026-06-15 (= 残 33 日、 2026-05-13 起算)
> Prize: $106,388 USD
> Teams: 2,959

## CRITICAL: Session start protocol

1. 本ファイル read
2. **Latest HANDOFF: `docs/dev/HANDOFF-2026-05-14.md`** ★ ← 必ず first read
3. git log + `kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge` で最新確認
4. handoff doc "Next session 即 action" を実行

## Project context (1 行)

NVIDIA Nemotron 3 Nano 30B を LoRA fine-tune して 6 種類の reasoning puzzle (Roman / Physics / Unit / Cipher / Bit / Symbol) を解かせる。 LB top 0.87 (3 teams 占有)、 0.86 = 297 teams 飽和。 我々の戦略: deterministic Python solver (6 types) + Searchformer trajectory CoT + LoRA SFT + GRPO + hybrid routing inference で 0.87+ 突破狙い。

## Strategy doc (= 親 plan の dense 版)

- 親 plan: `~/.claude/plans/iterative-dancing-pearl.md` (= 2026-05-13 起票、 6 Phase 戦略)
- dense: `docs/strategy/winning-strategy.dense.md` (= Phase 0 完了後に起票)

## Session end protocol

親 `~/projects/kaggle/CLAUDE.md` §0.2 参照、 新 HANDOFF doc 作成 + pointer 更新 + commit + push。

## 関連

- 親: `~/projects/kaggle/CLAUDE.md` (= §0 session protocol、 §11 「優勝本質性」 criterion)
- 親親: `~/.claude/CLAUDE.md` (= 全プロジェクト原則 + 主道フレームワーク)
- 親 plan: `~/.claude/plans/iterative-dancing-pearl.md` (= 33 日 day-by-day)

## Codex review 必須化 (= 親 §0.5)

- 方針段階 (= 設計判断 commit 前): `/codex:adversarial-review "<focus>"` × 1
- 実装段階 (= submission build / training script / inference adapter commit 前): `/codex:review --base main` × 1
- BLOCK 出たら必ず address (= 「軽微」 で押し切らない)
