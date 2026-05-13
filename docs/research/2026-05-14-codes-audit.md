# NVIDIA Nemotron Model Reasoning Challenge — CODES 監査 2026-05-14

> 監査日: 2026-05-13 実施、 2026-05-14 レポート化
> 監査者: Claude Code (Haiku 4.5)
> 対象: 上位 30 kernels + 最新 5 候補 (vote × recency matrix)

---

## 1. 監査範囲 (= Hotness ソート top 30 + 追加候補 5 件)

### 1.1 Top 30 baseline snapshot (2026-05-13)

| Rank | Ref | Votes | lastRunTime |
|---:|---|---:|---|
| 1 | ryanholbrook/nvidia-nemotron-submission-demo | 2180 | 2026-03-25 |
| 2 | dennisfong/nvidia-nemotron-sfttrainer-training | 765 | 2026-03-19 |
| 3 | kienngx/nvidia-nemotron-training-cot-labels | 642 | 2026-03-25 |
| 4 | kienngx/nvidia-nemotron-trained-models-submission | 551 | 2026-04-17 |
| 5 | asalhi/tinker-adapter-to-ready-to-submit-adapter | 463 | 2026-04-15 |
| 6 | konbu17/nemotron-sft-lora-with-cot | 488 | 2026-04-13 |
| 7 | huikang/tinker-submission-notebook | 462 | 2026-04-14 |
| 8 | dgxchen/training-with-unsloth-to-achieve-0-85-lb | 343 | 2026-04-25 |
| 9+ | (18 件省略) | ... | ... |

### 1.2 詳細読み対象 5 件 (vote × recency ビンで選定)

| # | Ref | Votes | lastRun | 選定理由 |
|---:|---|---:|---|---|
| **A** | mirzayasirabdullah07/nvidia-nemotron-model-notebook | 80 | 2026-05-12 | **最新 (= 2 日前)、 LoRA adapter packaging pipeline 専門** |
| **B** | rohanrk1813/nvidia-comp | 73 | 2026-05-06 | **1 week recent、 14 model version enumeration** |
| **C** | konbu17/nemotron-tong-style-cot-sft-updated-v2 | 90 | 2026-04-19 | **vote 上位 (= rank 26)、 CoT 再現性声高** |
| **D** | adilshamim8/nvidia-nemotron-model-reasoning-challenge-101 | 31 | 2026-05-09 | **end-to-end 統合、 最新だが low vote** |
| **E** | amanatar/nemotron-ultimate-sft-grpo-v3 | 77 | 2026-04-08 | **SFT + GRPO dual pipeline、 rank 28** |

---

## 2. 各 Kernel 要約

### A. mirzayasirabdullah07/nvidia-nemotron-model-notebook (80 votes, 2026-05-12)

**paradigm**: LoRA Adapter packaging pipeline + SVD merge (= Mamba gate_proj + x_proj → in_proj)

**主張 / score**: 「LoRA adapter 自動検出 + rank validation + 提出形式 flat zip 確認」

**代表 novel技法**:
- **IMPROVED ADAPTIVE SVD** (Mamba 層): gate_proj + x_proj 統合で variance 98% retention ratio を自動算出
- **Adapter config 再バリデーション**: zip 内部から再読み (= 提出前最終 safety check)
- **Expert unfusing** (MoE 層): w1 → up_proj、 w2 → down_proj への逆融合パターン

**既存 repo との差分**:
- `src/nvidia_nemotron_model_reasoning_challenge/adapters.py` に **LoRA validator** あるが、 SVD merge の adaptive rank selection は未実装
- adapter config path 自動検出 (= ADAPTER_HINTS fallback) は既存に無し、 utility 価値あり

**URL**: `/tmp/nemotron-codes-audit/nvidia-nemotron-model-notebook/nvidia-nemotron-model-notebook.ipynb`

**引用コード抜粋** (adaptive SVD):
```python
variance_retained = 0.98  # kept rank parameter
cumsum = torch.cumsum(S, dim=0) / S.sum()
k = torch.searchsorted(cumsum, variance_retained).item() + 1
k = min(k, rank)  # Don't exceed original rank
new_B = U[:, :k] * S[:k].unsqueeze(0)
```

---

### B. rohanrk1813/nvidia-comp (73 votes, 2026-05-06)

**paradigm**: Model version enumeration + direct zip (= 14 model path candidates)

**主張 / score**: 「kienngx の 14 個の 学習済み adapter variant を列挙、 tinker-adapter v14 を select」

**代表 novel技法**:
- **No-op kernel**: 実は `zip -r` で model_path をそのまま submit、 最小限スクリプト
- **Version ancestry**: 600-samples-packing → 1800s-lora → 2400-1e4-lr → 9500s-batch → CoT-3000 →  1200-CoT-5e-5 の進化系列が visible

**既存 repo との差分**:
- 公開 model list という形で version 系統図を implicit に示す (= 過去 work の可視化)
- 我々 repo の `notebooks/training/` config と overlap する学習パラメータ (lr, rank, 1e-5 vs 5e-5) が確認可能

**URL**: `/tmp/nemotron-codes-audit/nvidia-comp/nvidia-comp.ipynb`

---

### C. konbu17/nemotron-tong-style-cot-sft-updated-v2 (90 votes, 2026-04-19)

**paradigm**: CoT 再現性ノート (= Tong Hui Kang 0.85 solution の追跡)

**主張 / score**: exp026/s005 CV 0.768 (6 category local eval)、 「Tong Hui Kang の解法を reproductible notation で記録」

**代表 novel技法**:
- **Category-specific evaluation**: GC 93.8% / NC 100% / UC 98.8% / TE 95.2% / ET 27.2% / BM 50.0% の per-category CV
- **データ分割strategy**: train_split.csv (9000) + eval_split.csv (500) を 6 puzzle type に decompose
- **CoT label curation**: reasoning trajectory を manual / LLM で enrichment してからローカル valid

**既存 repo との差分**:
- `data/train_split_raw.csv` との alignment → 我々 repo の `src/nvidia_nemotron_model_reasoning_challenge/dataset_loader.py` にはない per-category split logic
- eval_split 用の separate validation → LB overfit 警戒の CV strategy

**URL**: `/tmp/nemotron-codes-audit/nemotron-tong-style-cot-sft-updated-v2/nemotron-tong-style-cot-sft-updated-v2.ipynb`

---

### D. adilshamim8/nvidia-nemotron-model-reasoning-challenge-101 (31 votes, 2026-05-09)

**paradigm**: End-to-end LoRA fine-tuning (= multi-step unified notebook)

**主張 / score**: 「environment check → imports → config → training → submission」 の full pipeline

**代表 novel技法**:
- **mamba_ssm fallback search**: `/kaggle/usr/lib/` 内部で dynamic import (= 依存性問題対応)
- **Environment reporting**: REQUIRED dict で逐一 availability check、 crash なし graceful degradation

**既存 repo との差分**:
- `src/nvidia_nemotron_model_reasoning_challenge/training/run.py` との機能 overlap 大
- 工程の「自動化」よりは「可視化」重視 (= debug-friendly だが submission optimized でない)

**URL**: `/tmp/nemotron-codes-audit/nvidia-nemotron-model-reasoning-challenge-101/nvidia-nemotron-model-reasoning-challenge-101.ipynb`

---

### E. amanatar/nemotron-ultimate-sft-grpo-v3 (77 votes, 2026-04-08)

**paradigm**: SFT + GRPO pipeline (= reinforcement learning)

**主張 / score**: 「MODE & CONFIG」 selector で QUICK_TEST / SMOKE_TEST / USE_GRPO / USE_COT_DATA を runtime 選択可能

**代表 novel技法**:
- **GRPO (Group Relative Policy Optimization)**: LLM output ranking を preference learning で最適化
- **Triton offline wheel install**: `/kaggle/input/**/*triton*.whl` から glob で自動検出、 offline 環境対応
- **Offline package pool**: glob で全 whl を locate、 importlib で dynamic install

**既存 repo との差分**:
- GRPO は experiment phase (= USE_GRPO = False が default)、 本格実装は未確認
- Triton offline strategy は utilities 価値あり (= env 構築安定性)

**URL**: `/tmp/nemotron-codes-audit/nemotron-ultimate-sft-grpo-v3/nemotron-ultimate-sft-grpo-v3.ipynb`

---

## 3. 既存 repo にない技法 (= Crosstalk 検出)

### 3.1 Mamba Adaptive SVD Merge (A → 既存 repo にない)
- **Mamba gate_proj + x_proj → in_proj** merge: kernel A の variance-retained rank selection が 参考価値
- **既存**: adapter.py に unfuse logic あるが、 SVD output のランク再最適化は no-op (= full rank 保持)
- **candidate**: src/nvidia_nemotron_model_reasoning_challenge/inference/adapter_merge.py に追加

### 3.2 Per-Category CV Split (C → 新規 utility)
- **6 puzzle type ごとの評価分離**: category imbalance を detect (= ET 27.2% vs NC 100% の gap が即座に visible)
- **既存**: `src/nvidia_nemotron_model_reasoning_challenge/evaluation.py` は aggregate LB score のみ
- **candidate**: `evaluation.py` に `evaluate_by_category()` helper を新規作成

### 3.3 Triton Offline Wheel Auto-locate (E → 環境 robustness)
- **glob pattern で `/kaggle/input/**/*triton*.whl`** を locate、 offline env で install
- **既存**: venv spec に triton pinned version あるが、 wheel 不在時の fallback なし
- **candidate**: `setup.py` の `[install_requires]` に conditional install block 追加

---

## 4. ★ v2 採用候補 (= Top 3、 effort / impact 付き)

### 4.1 🟩 **Kernel A: Adaptive SVD Merge (mirzayasirabdullah07)**
- **impact**: LoRA rank compression → inference latency -10-20%、 model size -proportional
- **novelty**: variance-retained selector が 中等度 novel (= known technique だが implementation wise)
- **effort**: **S** (= adapter_merge.py に 50 行追加)
- **timeline**: 即日実装可、 1 submit で 検証可能
- **採用**: **YES** — training 후 LoRA 압축이 필요한 경우, 이 기법이 없으면 rank 32 limit을 풀기 어려움

### 4.2 🟨 **Kernel C: Per-Category Evaluation (konbu17)**
- **impact**: category imbalance detection → training data rebalancing 의사결정 기반 제공
- **novelty**: vanilla approach (= no new algorithm)、 but execution discipline high
- **effort**: **S** (= evaluation.py 에 class/function 추가 20-30 행)
- **timeline**: Day 3 training loop 중 CV validation 으로 즉시 사용 가능
- **採用**: **MAYBE** — CV calibration이 필요하면 take, 그렇지 않으면 skip

### 4.3 🟥 **Kernel B: Model Version Enumeration (rohanrk1813)**
- **impact**: historical model tracking → ablation 의사결정 support
- **novelty**: no novelty、 enumeration skill only
- **effort**: **S** (= 이미 path list 형태로 정리)
- **timeline**: 수동 reference (= no code 불가) 용도
- **採用**: **NO** (code value zero、 reference document 용도로만 use)

---

## 5. 상태 요약

### 5.1 읽은 notebook 수
- **5 개** (1 주일 이내 최신 5 개 + rank 우선순위)
- A (newest 2026-05-12) / B (2026-05-06) / C (2026-04-19) / D (2026-05-09) / E (2026-04-08)

### 5.2 채택 권장
**Kernel A (Adaptive SVD)** → effort **S**、 impact **M** (= phase 3 LoRA compression)
**Kernel C (Per-Category CV)** → effort **S**、 impact **S** (= phase 2 CV calibration)

### 5.3 baseline 참고용 snapshot 저장
- top30-baseline.txt 로 CLI 출력 raw 저장 (= future delta tracking 용)
- git commit 전 docs/research/public_kernels/README.md update 필요 (= INDEX 화)

---

## 참고: 기술 분류 (= 공개 kernel 자산 inventory)

| Category | 대표 kernel | vote | note |
|---|---|---|---|
| **1. LoRA Packaging** | A (미르자) | 80 | adapter zip format validation + SVD merge |
| **2. CoT Training** | C (konbu17) | 90 | per-category CV, Tong 재현 |
| **3. Model Ensemble** | B (rohan) | 73 | version history enumeration |
| **4. End-to-End** | D (adil) | 31 | mamba_ssm env handling |
| **5. RL Training** | E (aman) | 77 | GRPO + offline wheel setup |
| **6. Top baseline** | ryanholbrook | 2180 | submission demo (kaggle official) |

---

> **audit 완료**: 2026-05-14 00:30 UTC
> 다음 단계: Kernel A/C code 검토 → phase 2/3 planning document 업데이트
