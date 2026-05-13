# 2026-05-14 過去コンペ winner solutions 集約 (= LLM-reasoning 系)

> 起源: v1 SFT training (= 8-11h) 待ち時間の高 ROI research、 ユーザー指示「待ちの間にできることないんか」
> 親 CLAUDE.md §8.1 #9 「各 submit 後 analysis ルーチン」 + §11 「優勝本質性」 criterion
> 目的: Nemotron Model Reasoning Challenge v2-v6 への直接 input、 採用可否を制約条件 (= 公式 metric の single-inference 縛り) で篩う

---

## 1. AIMO Progress Prize 1 (= 2024、 $131k 優勝)

**Team**: project-numina (NuminaMath)
**Score**: leaderboard top
**Base**: DeepSeekMath-Base-7B

### 核心 method
1. **2-stage SFT**:
   - Stage 1: Chain-of-Thought training on math problems + text solutions
   - Stage 2: Tool-Integrated Reasoning training on math problems + code solutions
2. **SC-TIR** (Self-Consistency with Tool-Integrated Reasoning) inference:
   - input を N 回複製、 N 候補生成
   - Python code block を抽出 → 実行 → traceback fed back
   - M depth で iterate、 self-correct
   - majority voting で final answer
3. **8-bit quantization** (AutoGPTQ) で Kaggle compute 制約に対応

### 我々 への 適用
| 技法 | 採否 | 理由 |
|---|---|---|
| 2-stage SFT | △ 部分採用 | tool stage 不可、 stage 1 CoT SFT は採用済 |
| SC-TIR / Python REPL | ❌ 不可 | 公式 metric notebook が tool 禁止 (= single vLLM inference) |
| Majority voting | ❌ 不可 | 同上 (= temperature=1.0、 1 sample) |
| 8-bit quantization | △ | submission は LoRA adapter のみ提出、 base model の量子化は metric notebook 側仕様 |

**出典**: https://huggingface.co/blog/winning-aimo-progress-prize / https://github.com/project-numina/aimo-progress-prize

---

## 2. AIMO Progress Prize 2 (= 2025、 NemoSkills 34/50 top + Numina writeup)

**Solution の 3 本柱** (= Numina writeup):
1. **大規模 dataset**: 540K unique high-quality math problems + 3.2M long-reasoning solutions
2. **TIR の iterative training**: 1.7M high-quality TIR solutions
3. **GenSelect** (= generative solution selection): majority voting を超える、 model が「N candidates から best を選ぶ」 を学習

### 我々 への 適用
| 技法 | 採否 | 理由 |
|---|---|---|
| 540K + 3.2M 大規模 dataset | △ scale 違い | 我々の 6 puzzle types は限定 domain、 5418 verifier-backed で覆える。 ただし v2-v3 で synthesis 増産 (= 20-30K target) は検討余地 |
| TIR | ❌ 不可 | tool 禁止 |
| GenSelect | ❌ 不可 | single inference 制約 |
| Long-reasoning trace SFT | ✅ **採用済** | 我々の Searchformer-style trace = 同型 |

**出典**: https://arxiv.org/pdf/2504.16891 / https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2

---

## 3. NeurIPS LLM Efficiency Challenge (= 2023、 Nanyang Tech 優勝)

**Score**: baseline +16.4%、 全 model param の 0.16% のみ学習
**Track winner**: Team Upaya (4090 track)

### 核心 method
1. **LoRA dominant**: top teams は全員 LoRA、 rank 8-256
2. **「careful data curation > massive data」**:
   - 厳選 high-quality data >> 大量 noisy data
   - team-specific differentiator は data quality
3. **Team Upaya**: semi-automated data curation + generalization focus

### 我々 への 適用
| 技法 | 採否 | 理由 |
|---|---|---|
| LoRA rank 32 | ✅ 採用済 | 中庸、 30B model に対し妥当 |
| 「quality > quantity」 | ✅ **採用済** | 5418 records すべて verifier-backed = 100% 正解保証 |
| Semi-automated data curation | ✅ 採用済 | type_classifier + solver による automatic generation |
| Generalization (= 多 task) | △ 検討 | 我々は 6 fixed types = generalization 不要、 specialization OK |

**出典**: https://llm-efficiency-challenge.github.io/ / https://xebia.com/blog/winning-recipe-for-finetuning-llm/

---

## 4. RLVR (= Reinforcement Learning with Verifiable Rewards、 DeepSeek-R1 採用)

### 核心 理論
- RLVR は **standard RL + deterministic reward function**
- Binary reward: correct (= verifier OK) / wrong
- math reasoning は **finite-horizon MDP with deterministic state transitions, tree-structured dynamics, binary terminal rewards**
- 採用 model: DeepSeek-R1 (= 7B/67B)、 OpenAI o1 系統

### 我々 への 適用 (= Phase 4 GRPO の理論基盤)
| 要素 | 採否 | 我々の対応 |
|---|---|---|
| Deterministic verifier reward | ✅ 採用予定 | Phase 4 で `\boxed{}` exact match + format compliance を reward |
| Binary terminal reward | ✅ 採用予定 | per-puzzle 正解/不正解 |
| Tree-structured rollout | ✅ 採用予定 | GRPO の group sampling (= G 候補生成 → group-relative advantage) |
| RFT (Rejection sampling Fine-Tuning) | △ 検討 | GRPO の前段で rejection sampling SFT を挟む手も |

**出典**: https://lucek.ai/blogs/rlvr-with-llms / https://github.com/opendilab/awesome-RLVR

---

## 5. 横断 insight & 我々の戦略への影響

### 5.1 公式 metric 制約 (= tool 禁止 / single inference) が AIMO 技法を **大半 ブロック**

AIMO 1/2 の核心技法 (= TIR / SC-TIR / GenSelect / Majority voting) は **すべて multi-sample / tool integration**。 我々のコンペは:
- `vllm.LLM` + `LoRARequest` の **単一 inference**
- temperature=1.0, top_p=1.0、 1 sample のみ
- tool 実行不可

→ **AIMO 系の最強技法は我々には適用不能**。 唯一の 救済 path = base model の reasoning を **SFT + RLVR (GRPO)** で limit まで引き出す。

### 5.2 我々の plan v3 path は方向性 正しい

| Phase | 我々の plan | AIMO/RLVR 系の equivalent | 確信 |
|---|---|---|---|
| Phase 1-2 | solver-CoT SFT (= 5418 records) | NuminaMath stage 1 (CoT SFT) | 高 |
| Phase 3 | SFT focused + hyperparameter grid | NeurIPS 2023 data curation | 高 |
| Phase 4 (optional) | GRPO with binary reward | RLVR / DeepSeek-R1 | 中-高 |
| Phase 5 | inference robustness (= `\boxed{}` audit) | metric exploit 対策 | 高 |

### 5.3 v2-v6 で追加検討 すべき技法

1. **【MUST 検討】 Rejection Sampling SFT** (= Phase 3.5):
   - Phase 1 SFT 済 model で **N candidates 生成** (= train data 上、 inference ではない)
   - verifier で correct のみ keep、 これを SFT data に追加 → 再 SFT
   - NuminaMath stage 1.5 相当、 tool 不要、 公式 metric 制約 OK
   - 効果見込: +0.01-0.02 LB
2. **【SHOULD 検討】 Multi-seed adapter merge** (= Phase 5 で 既計画):
   - 同じ SFT data で 2-3 seed の adapter 訓練
   - peft の `add_weighted_adapter` で merge
   - 公式 metric は **adapter 1 つ前提**、 merge は可能 (= post-training)
   - 効果見込: +0.005-0.015 LB
3. **【MAY 検討】 GRPO** (= Phase 4 optional):
   - Day 19 dry-run gate pass のみ
   - Kaggle T4 で 9h 内完走条件
   - 効果見込: +0.02-0.04 LB、 ただし失敗 risk 高

### 5.4 我々の独自 strengths

- **deterministic verifier-backed CoT**: AIMO は LLM-generated CoT を keep、 我々は **Python solver による 100% 正解 trace** = data quality は AIMO より高
- **6 puzzle types specialization**: AIMO は math olympiad 全般、 我々は 6 fixed types = solver overlay で specialization
- **公式 metric の rel_tol=1e-2 寛容**: AIMO は exact match、 我々は 1% 寛容 = format mismatch tolerance 高

---

## 6. 結論 (= 我々の plan v3 への影響)

- AIMO 系の **tool / multi-sample 技法は不可** → SFT + RLVR の単線 path に集中
- Phase 3.5 で **Rejection Sampling SFT** を追加検討 (= effort M、 効果 +0.01-0.02)
- Phase 4 GRPO は理論的に支持 (= RLVR) があり、 Day 19 gate pass なら実装価値あり
- Phase 5 adapter merge は post-training で公式制約に抵触せず、 安全な upside

朝の v1 LB 結果次第で:
- LB ≥ 0.865 → Phase 2 進入 + Rejection Sampling SFT 検討開始
- LB 0.85-0.864 → logprob filter + upsample cap 5x に下げる v2 patch (= discussion audit より)
- LB < 0.85 → 戦略 pivot、 dgxchen baseline に revert

---

## 出典 一覧

- AIMO Progress Prize 1 winner blog: https://huggingface.co/blog/winning-aimo-progress-prize
- AIMO Progress Prize 1 GitHub repo: https://github.com/project-numina/aimo-progress-prize
- AIMO Progress Prize 2 paper (Numina, 2504.16891): https://arxiv.org/pdf/2504.16891
- AIMO Progress Prize 2 leaderboard: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2
- NeurIPS LLM Efficiency Challenge 2023: https://llm-efficiency-challenge.github.io/
- NeurIPS winning recipe Xebia: https://xebia.com/blog/winning-recipe-for-finetuning-llm/
- RLVR primer: https://lucek.ai/blogs/rlvr-with-llms
- RLVR curated list: https://github.com/opendilab/awesome-RLVR
- DeepSeek-R1 paper / blog: (= RLVR の代表 implementer、 LLM reasoning state-of-the-art)
