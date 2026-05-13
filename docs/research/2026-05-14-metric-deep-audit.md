# 2026-05-14 公式メトリック・ノートブック 深掘り監査

## 1. ファイル一覧 + 役割

| ファイル | サイズ | 役割 |
|---|---|---|
| `nvidia-nemotron-metric.ipynb` | 21 KB | 唯一の metric NB。 LoRA load + vLLM inference + answer extraction + accuracy scoring |

**構成**: setup (torch/triton) → `cache_model()` → `extract_final_answer()` → `verify()` → `generate_predictions()` → `score()` (= entry point)

---

## 2. Per-type evaluator edge cases (完全度: HIGH — notebook に evaluator function は 1 個のみ)

### 2.1 統一評価関数: `verify(stored_answer, predicted)`

**CRITICAL**: per-type evaluator は **存在しない**。全問に単一 `verify()` 関数で評価。

#### 規則:

```python
# L201-217 (nvidia-nemotron-metric.ipynb)
def verify(stored_answer: str, predicted: str) -> bool:
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # 規則1: Binary 判定
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    # 規則2: Float 試行
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # 規則3: Case-insensitive string
        return predicted.lower() == stored_answer.lower()
```

#### Edge cases 検査:

| Type | Edge case | 挙動 | Risk |
|---|---|---|---|
| **binary** | `"00011"` vs `"11"` | ❌ **FAIL** — binary match は full string比較、 leading-zero 無視されない | **HIGH** |
| **binary** | `"1"` vs `"1.0"` | ❌ FAIL — `"1.0"` は float に cast、 `math.isclose(1.0, 1.0)` = True だが predicted が binary pattern でない場合は 規則1 skip → 規則3 (case-insensitive) で比較 | **MEDIUM** |
| **roman** | `"XLVII"` vs `"xlvii"` | ✅ PASS — 規則3 (case-insensitive 文字列) で hit | **LOW** |
| **roman** | `"XLVII"` vs `"47"` | ❌ FAIL — numeric cast 失敗 → 文字列比較で fail | **HIGH** |
| **physics** | `"9.81"` vs `"9.8100"` | ✅ PASS — `rel_tol=1e-2` (1%) で `math.isclose(9.81, 9.8100)` = True | **LOW** |
| **physics** | `"1e-5"` vs `"0.00001"` | ✅ PASS — float cast で両者とも 1e-5、 `math.isclose` = True | **LOW** |
| **unit** | `"3.5 m"` vs `"3.5"` | ❌ FAIL — `float("3.5 m")` exception → 規則3 → `"3.5 m".lower() != "3.5".lower()` | **HIGH** |
| **cipher** | `"HELLO"` vs `"hello"` | ✅ PASS — 規則3 で case-insensitive | **LOW** |
| **symbol** | `"\\frac{1}{2}"` vs `"0.5"` | ❌ FAIL — float("\\frac{1}{2}") exception → 規則3 → 文字列不一致 | **HIGH** |

### 2.2 数値判定の境界値

- **`rel_tol=1e-2`** (1%) : **広い許容度** — 4.9 と 5.0 は許可 (= 2%)、 49 と 50 も許可 (= 2%)
- **`abs_tol=1e-5`** : 0 に近い数値で効く (e.g., 1e-6 vs 0 なら `abs_tol` で OK だが rel_tol は 無限倍)
- **実際**: `predicted = "0.0001"`, `stored = "0"` → `math.isclose(0.0001, 0, rel_tol=1e-2, abs_tol=1e-5)` = **False** (abs diff 1e-4 > 1e-5)

---

## 3. 公式 prompt format (完全再現)

### 3.1 System prompt

**未設定** — vLLM chat template で決定 (host default)。

### 3.2 User prompt template (L405-415)

```python
user_content = (
    item.prompt
    + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
)

prompt = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': user_content}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
```

**CRITICAL**:
1. user content = **test.csv の `prompt` 列 + suffix**
2. suffix = **`\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`**
3. tokenizer **chat template 適用** (= Nemotron default は LLaMA-style `<|user|>...<|assistant|>`)
4. **`enable_thinking=True`** — thinking tokens 有効 (重要)

**我々の SFT data との mismatch risk**: test.csv prompt が何か未知 → render() で同一フォーマットを手動再現必須。

---

## 4. 答え抽出 logic (完全仕様)

### 4.1 `extract_final_answer(text)` L126-197

```python
# Step 1: \boxed{} 優先
boxed_starts = list(re.finditer(r'\\boxed\{', text))
if boxed_starts:
    # 複数 boxed{} の場合、 各々について最後の } までを scope
    # → 最後の boxed{} の content を返す
    matches = [...]
    if non_empty matches: return non_empty[-1]

# Step 2: 別 format (「Final answer is:」 pattern)
patterns = [r'Final answer is:\s*([^\n]+)', ...]
for pattern in patterns:
    matches = re.findall(pattern, text, re.IGNORECASE)
    if matches: return matches[-1].strip()

# Step 3: 最後の数値
matches = re.findall(r'-?\d+(?:\.\d+)?', text)
if matches: return matches[-1]

# Step 4: 最後の行
return lines[-1] if lines else 'NOT_FOUND'
```

### 4.2 Edge cases

| Case | Input | Output | Comment |
|---|---|---|---|
| Empty boxed | `r"\boxed{}"` | `""` (empty) → 規則3 (最後の数値 など fallback) | **HIGH RISK** |
| Multiple boxed | `r"\boxed{1}\boxed{2}"` | `"2"` (最後を pick) | ✅ EXPECTED |
| Nested braces | `r"\boxed{\frac{1}{2}}"` | `"\frac{1}{2}"` (last } まで) | ✅ OK (LaTeX math) |
| Mismatched | `r"\boxed{1"` (no closing }) | segment[:rfind('}')] = `""` (empty) → fallback | **HIGH RISK** |
| No boxed | `"The answer is 42"` | `"42"` (step 3 で最後の数値) | ✅ OK |
| Only text | `"hello world"` | `"world"` (step 4 で最後の行) | ⚠️ 危険 — 数値ない場合 plain text return |

**CRITICAL MISMATCH**: empty / mismatched `\boxed{}` では **fallback chain が走る**。 最後の数値や最後の行を返すため、 model output の後続テキストが answer として誤抽出される可能性。

---

## 5. SamplingParams + LoRARequest 詳細

### 5.1 SamplingParams (L403)

```python
sampling_params = SamplingParams(
    temperature=temperature,  # default 1.0
    top_p=top_p,               # default 1.0
    max_tokens=max_tokens,      # default 3584
)
```

**特性**:
- **deterministic ではない** (temperature 1.0、 top_p 1.0 でも stochastic)
- **max_tokens=3584** + **max_model_len=4096** → 入力 token 最大 512

### 5.2 LoRARequest (L418)

```python
lora_request=LoRARequest('adapter', 1, lora_path)
```

**要件**: submission.csv の 'prediction' 列に **adapter ディレクトリパス** を記入。 metric NB がそのパスから `adapter_config.json` 探索 (L341-352)。

---

## 6. ★ 我々の SFT data / inference の mismatch risk 一覧 (severity 付き)

| # | Risk | Severity | Description | Mitigation |
|---|---|---|---|---|
| **A1** | Binary leading-zero | **CRITICAL** | `"0011"` vs `"11"` は string full-match で FAIL。 SFT で leading-zero 生成なら LB 黙 -0.5%+ | train data で leading-zero 統一、 inference で去除 or padding 일관성 확인 |
| **A2** | Unit/symbol non-numeric | **CRITICAL** | `"3.5 m"` or `"\\frac{1}{2}"` は float cast exception → string case-insensitive で判定。 metric は numeric 期待なら SFT で numeric-only output enforce | loss に unit-stripping penalty、 or post-processing で unit 去除 |
| **A3** | Empty `\boxed{}` fallback | **HIGH** | model が `\boxed{}` 出力しても empty なら step 3-4 fallback → 後続テキストが誤抽出。 テスト時だけでなく SFT 중에도 可能性 | train data に empty boxed 패턴 절대금지、 inference で generation length 강제 (min_tokens > 0) |
| **A4** | `enable_thinking=True` 의존 | **HIGH** | metric NB가 thinking tokens 활성화。 우리 SFT data / tokenizer가 thinking format support 안 하면 silent mismatch | Nemotron tokenizer docs 확인、 SFT vocab에 thinking token 포함 필수 |
| **A5** | `tokenizer.apply_chat_template` format | **HIGH** | Nemotron chat template (LLaMA-style) 依存。 우리 adapter가 다른 tokenizer 가정하면 prompt format 불일치 → score 올라가지 않음 | adapter build 시점에 Nemotron official tokenizer 명시 사용、 chat_template 일치 확인 |
| **A6** | `rel_tol=1e-2` rounding | **MEDIUM** | 1% 허용도。 우리가 "정확한 수치" 기대하면 실패。 physics/unit에서 rounding error 누적되면 1% 초과 가능 | est-LB calibration doc에서 1e-2 허용도 명시、 test-time precision 체크 |
| **A7** | Fallback chain order | **MEDIUM** | boxed ❌ → Final answer pattern ❌ → last number ❌ → last line。 우리가 다른 order 기대하면 surprise | metric NB 읽으며 확인했으니 step 순서 고정、 SFT에서 expected format hierarchy match |
| **A8** | test.csv prompt unknown | **HIGH** | test.csv 'prompt' 열 내용이 unknown。 우리 SFT data의 render()가 다르면 silent distribution shift | test.csv head 확인 + render() byte-exact match test (dummy inference로 검증) |

---

## 7. SFT data 구성 요구사항 (역산)

### 7.1 Example format (우리가 준수해야 할 형식)

```json
{
  "prompt": "<test.csv의 prompt 열 그대로>",
  "thinking": "...",  // enable_thinking=True 대응
  "assistant_response": "... \\boxed{<최종답>} ...",
  "answer": "<ground_truth>"
}
```

### 7.2 inference.py 검증 체크리스트

- [ ] submission.csv 'prediction' = adapter path (절대경로 확인)
- [ ] adapter_config.json 존재여부 (metric NB가 locate 할 수 있도록)
- [ ] LoRA rank ≤ 32 (default)
- [ ] max_tokens ≤ 3584
- [ ] max_model_len ≤ 4096

---

## 8. 우리의 rendering() 필수 검증

우리 SFT data의 render() 함수는 다음을 보장해야:

```python
# metric NB와 byte-identical 재현
user_content = prompt + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
# → tokenizer.apply_chat_template(..., enable_thinking=True) 결과와 일치해야 함
```

**검증방법**: dummy test set (10개) 로 inference 돌려서 extract_final_answer() 거쳐 답 나오는지 확인。
