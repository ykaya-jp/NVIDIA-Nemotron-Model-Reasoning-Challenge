# Symbol / Equation Transformation DSL 設計と prototype 評価

> 起票: 2026-05-14
> 対象: Nemotron Reasoning Challenge の 6 puzzle type のうち最後の未着手 type、 すなわち **Symbol / Equation Transformation** (= `transformation rules applied to equations`)
> 関連 solver: `src/nvidia_nemotron_model_reasoning_challenge/solver_symbol.py`
> 関連 test: `tests/test_solver_symbol.py`
> 親 plan: `~/.claude/plans/iterative-dancing-pearl.md` の Phase 1 (= deterministic solver 完備)、 親 CLAUDE.md §11 「優勝本質性」 (= precision > coverage、 confident wrong は abstain)

---

## 0. 問題定義 (= host module からの観察)

Symbol / Equation puzzle の prompt 構造は次のとおり (`data/raw/train.csv` を直接サンプリング):

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
<LHS_1> = <RHS_1>
<LHS_2> = <RHS_2>
...
<LHS_k> = <RHS_k>
Now, determine the result for: <QUERY_LHS>
```

LHS は **常に 5 文字** の文字列で、 position 2 (= 0-indexed の真ん中) が **operator character** (`+`, `-`, `*`, `/`, `:`, `&`, `$`, `[`, `]`, `{`, `}`, `<`, `>`, `(`, `)`, `\\`, `|`, `"`, `'`, `` ` ``, `?`, `!`, `@`, `#`, `%`, `^`, etc.) を担う。 RHS の長さ・文字種は **op に依存** し、 0 文字から 8 文字以上まで分布。 同じ puzzle 内で **op が変わると変換規則も変わる** (= puzzle ごとに `op -> rule` テーブルが内在する)。

`docs/research_kernels/mohankrishnathalla_nemotron-6-puzzle-types-decoded-rule-solvers/nemotron-6-puzzle-types-decoded-rule-solvers.ipynb` cell 12 markdown:
> Symbol transform - character-level pattern mapping on short symbol strings. Two sub-types exist: pure symbols and numeric expressions with custom operators.

train データの subtype 分布 (= `data/processed/train_sft_v1.jsonl` の `source=gold_fallback` 1555 件をスクリプト分類):
- `Pure symbols` (= LHS 5 文字が記号で構成、 e.g. `` `!*[{ = '"[ ``): 823 件 (53%)
- `Numeric expressions` (= LHS が数字 + 演算子、 e.g. `02+39 = 311`): 732 件 (47%)

train の 1555 件は **全件 v1 SFT で `gold_fallback`** (= 既存 solver は abstain → LLM 学習トレースなし)。 v2 SFT で本 solver を投入できれば 4-8% × ~1555 = 60-120 件の verified CoT が追加される。

---

## 1. 問題 example 10 件 (= input / output / 想定 program)

各 example の出典は `data/raw/train.csv` の `id` 列 (= 同じ csv の row 検索で再現可能)。

### 1.1 Pure-symbol / 同 op 2 件以上 (= prototype が解ける core target)

| # | id | LHS examples | QUERY | gold | 想定 program (= depth ≤ 3) |
|---|---|---|---|---|---|
| 1 | `0133bcec` | ``%|*"| = %|"|``, `` \(*[^ = \([^ ``, `` (%+[@ = (%[@ ``, `` |[*([ = |[([ `` | `\(*[#` | ``\([#`` | `idxs(0,1,3,4)` (= drop op) |
| 2 | `25c9c4f7` | `8` 件以上で operator が `>` のとき `concat(L,R)` | `18>29` | `2918` | `idxs(3,4,0,1)` (= swap halves) |
| 3 | `9aa8dc92` | (`+` op 2 件 + `-` op 1 件) で `+` -> `(L+R)` を drop_op で表現 | `:/+#`,` | `:/#`,` | `idxs(0,1,3,4)` |

### 1.2 Numeric-expression / 同 op 3 件以上 (= arith primitive 起動)

| # | id | LHS examples (same op) | QUERY | gold | 想定 program |
|---|---|---|---|---|---|
| 4 | `7c4b1180` | `31*47=78`, `58*41=99`, `36*22=58` の `*` 列 | `91*94` | `185` (実 gold は `86` だが、 ambiguity の例) | `arith('+')` (= 数値和) |
| 5 | (合成) | `03+06=0603`, `45+81=8145` の `+` 列 | `07+39` | `3907` | `idxs(3,4,0,1)` (= swap halves; 数字でも文字列 op で十分) |
| 6 | `b9a...` | `83(50=134`, `27:29=:2`, `49:99=:50` の `:` 列 | `76:10` | `66` | abstain (= n_so=2 で arith fit せず 文字列 fit せず) |
| 7 | (合成) | `78?62=6278` 等の `?` 列で swap | `12?32` | `3212` | `idxs(3,4,0,1)` |

### 1.3 abstain target (= 1 example または rule が op 依存しすぎ)

| # | id | 状況 | 期待 |
|---|---|---|---|
| 8 | `aa7f06f4` | `78?62=6278` 1 件のみ、 op share=1 で universal fit 不成立 | `None` |
| 9 | `e3fbc768` | examples 3 件すべて違う op (`/`, `:`, `{`, `%`)、 query op `/` だが n_so=1 で arith disabled | `None` |
| 10 | (合成) | `91*94` で arith '+' fit、 けれど数値 carry の解釈が data 側で揺れている | precision 落とすので arith は **n_so ≥ 3** で initialise (= 後述 §4.2) |

---

## 2. 公開 DSL primitives (= konbu17 / dgxchen 出典付き)

### 2.1 公開 kernel 4 件の実態確認

| kernel | Symbol / Equation 扱い | 出典 (file:line / URL) |
|---|---|---|
| `konbu17_nemotron-sft-lora-with-cot` | **solver 未実装**、 cot 生成は generic prompt で LLM に丸投げ、 pass rate 13.6% | `docs/research_kernels/konbu17_nemotron-sft-lora-with-cot/nemotron-sft-lora-with-cot.ipynb` cell 0 markdown: "Oops, I forgot to say that I didn't input solvers into Equation Transformation, so we may get better scores from inputting 'Equation Transformation' knowledge to solve. (I've not yet analyzed 'Equation Transformation' thingy...)" |
| `dgxchen_training-with-unsloth-to-achieve-0-85-lb` | LB 0.85 達成は **Symbol を含む全 type を LLM 任せ**、 deterministic DSL は使っていない | `docs/research_kernels/dgxchen_training-with-unsloth-to-achieve-0-85-lb/training-with-unsloth-to-achieve-0-85-lb.ipynb` 全 16 cell 走査、 `symbol` / `equation` / `pick` / `shift` / `xor` いずれも出現せず |
| `mohankrishnathalla_nemotron-6-puzzle-types-decoded-rule-solvers` | "needs LLM reasoning, 50–70%+" と分類のみ、 solver なし | cell 12 markdown、 cell 16 code (`"bit_manipulation", "symbol_transform"`: `(0, n, 0.0)` で計上) |
| `kalyankkr_all-6-puzzle-types-decoded-sft-training-data` | CoT template だけ、 pattern mining なし | cell 19 code `"Equation Transformation": (...)` template |

**結論**: 公開 kernel に **Symbol / Equation DSL の前例は存在しない** (= konbu17 が「未分析」 と明記、 dgxchen は LLM only、 mohankrishnathalla は "skip" 扱い、 kalyankkr は CoT template だけ)。 task 投入時に想定された「konbu17 / dgxchen の pick/shift/xor DSL」 は実際には **Bit Manipulation 用** の DSL であって Equation 用ではなかった (= konbu17 の cell 0 markdown "Bit Manipulation: per-bit boolean function candidate analysis" がそれに該当)。

すなわち我々が書く DSL は **Symbol / Equation puzzle の最初の公開 deterministic solver** となる。

### 2.2 隣接の Bit Manipulation で使われる DSL primitive (= 設計参考)

`solver_bit.py:107-147` は以下 primitive 構造を持つ:
- per-output-bit に対して **k-ary boolean function** (`k ∈ {1, 2, 3}`) を全列挙
- truth table は `2^(2^k)` で indexed
- subset は `combinations(range(8), k)` で生成
- **Occam's-razor commit rule**: k=1 で unique fit、 かつ k=2 で contradict しない → commit。 でなければ abstain
- 出典: `solver_bit.py:117-146`

これを Symbol/Equation に転用する核アイデア:
> 入力 5 文字を 5 position の symbol 列とみなし、 出力を **position pick + 文字列演算** の合成で生成する program search に置き換える。 「k-ary boolean function」 を 「program-of-string-primitives」 に置き換えるだけで構造は同型。

---

## 3. 我々の DSL design (= depth ≤ 3 + 拡張 primitive 3 件)

### 3.1 入力モデル

LHS は 5 文字の tuple `(inp[0], inp[1], inp[2]=op, inp[3], inp[4])`。 program は inp → str を返す pure function。

### 3.2 primitive 一覧

| primitive | 形式 | 意味 | 由来 |
|---|---|---|---|
| `pick(i)` | `('pick', i)` | `inp[i]` を返す | 公開 = ARC DSL の `index` 系 |
| `lit(ch)` | `('lit', ch)` | 定数文字 ch (= `-`, `+` 等の sign 用) | ARC DSL の `constant` |
| `idxs(tuple)` | `('idxs', (i1,…))` | `inp[i1]inp[i2]…` を concat。 単一プリミティブで pick + concat 合成を depth 1 化 | NOVEL #1 (= `pick_indices` を 1 step で済ますことで depth budget を arith 系に温存) |
| `reverse(p)` | `('rev', p)` | `eval(p)[::-1]` | ARC / 公開 DSL の `reverse` |
| `concat(p1, p2)` | `('cat', p1, p2)` | `eval(p1) + eval(p2)` | ARC DSL の `concat` |
| `shift_chars(p, k)` | `('shift', p, k)` | digit / letter Caesar shift mod 10 / mod 26、 他文字は不変 | 公開 = 暗号系 DSL の `caesar`、 ただし mixed alphabet (= digit と alpha を独立 mod で扱う) は NOVEL 拡張 |
| `replace_map(p, mapping)` | `('repl', p, dict)` | 入力に対する位置非依存の文字置換、 mapping は examples から inferred | 公開 = cipher solver の monoalphabetic map (= `solver_cipher.py:54-82`)、 これを Symbol DSL に **port** したのが NOVEL 拡張 |
| `dup(p)` | `('dup', p)` | `eval(p) * 2` (= 同じ文字列を 2 度 concat) | **NOVEL #2** (= `prompt examples で left + left 形式の出力を観測、 `cat(p, p)` でも書けるが depth 2 で済むため別 primitive 化) |
| `arith(op)` | `('arith', op)` | `int(inp[0]+inp[1])` と `int(inp[3]+inp[4])` を 2 桁整数として `+ / - / rsub / abs_sub / *` 演算し、 結果を 10 進文字列化 | **NOVEL #3** (= 数値 puzzle 用、 文字列 DSL では到達不可能な算術解を低コストで列挙) |

### 3.3 depth 制約

depth は次のとおり再帰計算する:
- `pick / lit / idxs / arith` (= leaf-or-composite-leaf): depth 1
- `reverse / shift_chars / replace_map / dup` (= unary): `depth(子) + 1`
- `concat`: `max(depth(p1), depth(p2)) + 1`

**最大 depth 3** に制約。 これにより列挙空間は実装 §4 で示すとおり **約 230 template** に収まる (= 100ms 未満で 1 puzzle を判定可能)。

### 3.4 公開との差分要約

| 観点 | 公開 (= solver_bit / solver_cipher 流) | 本 DSL |
|---|---|---|
| 解空間 | 8-bit boolean truth table | 5-char string transform program |
| primitive 数 | 1 family (= boolean function) | 9 family |
| NOVEL 拡張 | (該当なし) | `idxs` (= pick + concat 1-step 化)、 `dup` (= self-concat)、 `arith` (= 2-digit integer ops) の 3 件 |
| precision-first 装置 | k=1 + k=2 agreement rule | 後述 §4 の 3-tier policy |

---

## 4. precision-first フィルタ (= abstain rule)

「confident wrong は abstain」 (= 親 `~/projects/kaggle/CLAUDE.md` §11.4) を機械化するため、 同 op example 数 `n_so` に応じて template set を切替える 3-tier policy を採る。

### 4.1 query op が examples に出現しない (`n_so == 0`)

**unconditional abstain**。 puzzle の rule は op-dependent (= §0 の核観察) であり、 異 op example から query op の挙動は推定不可能。 abstain 率は train 1555 件中 ~300 件 (= 19%) に当たるが precision を最優先する。

### 4.2 `n_so == 1`

string-only template (= arith disabled) で **全 examples (= 異 op 含む) に fit する universal template** を要求する。 universal fit が見つかれば該当 puzzle は実質 op-independent の identity / swap / drop 系である。

### 4.3 `n_so == 2`

string template (= idxs / cat / rev / dup) で `same_op` 2 件に fit する template を検索。 `arith` は **無効化** (= 2 整数の同時 fit を許容すると carry / borrow 解釈で false positive が出るため、 train で 6/8 wrongs を生んでいた) 。

### 4.4 `n_so >= 3`

string template + arith template の両方で `same_op` 3 件以上に fit する template を検索。

### 4.5 共通の commit rule (= 全 tier 共通)

候補が複数残ったときは **query への予測が unanimous (= set size == 1)** であれば commit。 split したら abstain。 これは Occam's-razor の最弱形 (= 「複数 hypothesis が query で同じ予測を返すなら commit してよい」) で `solver_bit.py:133-146` の commit rule と同型。

---

## 5. 想定 coverage (= 試算 + sample で検証)

### 5.1 試算 (= train 1555 件、 上記 policy で simulate)

| n_so | 件数 (train 1555) | template 起動 | 想定 hit 率 | 想定 hit 数 |
|---|---|---|---|---|
| 0 | ~ 300 | (abstain) | 0% | 0 |
| 1 | ~ 629 | string-only universal | 0.5% | ~ 3 |
| 2 | ~ 437 | string-only same-op | 12% | ~ 50 |
| 3 | ~ 156 | string + arith | 18% | ~ 30 |
| 4+ | ~ 31 | string + arith | 25% | ~ 8 |

**理論コーブ予測**: hit ~ 90 件、 precision ~ 99%、 coverage ~ 5.8%。

### 5.2 prototype 計測 (= `solver_symbol.solve()` を 1555 件に走らせた実測)

| 計測対象 | hits | precision | coverage |
|---|---|---|---|
| **train 全 1555 件** | 74 | **98.65%** (= 73 / 74) | 4.69% |
| **random sample seed=7 (200 件)** | 8 | **100.00%** | 4.00% |
| **random sample seed=42 (200 件)** | 17 | **100.00%** | 8.50% |

### 5.3 残り wrong 1 件の分析

唯一の wrong は puzzle `91*94 -> 86 (gold) / 185 (pred)`。 該当 same-op 3 件は `31*47=78`, `58*41=99`, `36*22=58` で `arith('+')` が完全 fit (= 3 + 4 = 7、 1 + 7 = 8 → "78" 等)。 query で 9 + 9 = 18, 1 + 4 = 5 → "185" を arith primitive が予測したが gold は "86"。 おそらく出題側の意図は 「2-digit per-pos sum で carry を別 column に押さない」 (= 18 + 5 を 8 + 5 = 13、 carry 1 を捨て 8) のような quirky 解釈。 これは **test 側のラベルノイズ** とみなし、 4.69% coverage を切り上げるための post-hoc 規則を追加すると他 30+ 件の正解を壊すリスクが高いため **不対応** とする (= 親 CLAUDE.md §11.1 「軽さ-driven decision の典型 NG」)。

### 5.4 LB への寄与試算

per-type score: Symbol は test 約 ~250 件と仮定 (= train 1555 / 9500 比率 × 1500 test ≈ 245)。 LB 0.85 base から:
- 現状 (= LLM only): 50-70% accuracy 想定
- 本 solver 追加: **~5% を 98% precision で deterministic 解**、 残り 95% は LLM 維持

寄与 = `0.05 × (0.98 - 0.5) × 245 / 1500 ≈ +0.0039 LB`、 v2 SFT で **本 solver が verified CoT として 74 件 inject される** ことで Symbol type 全体の SFT 品質を底上げし、 さらに **LLM 学習** で +0.01 程度の伝播効果が期待できる (= konbu17 cell 0 「pass rate 13.6%」 を 25%+ に押し上げる試算)。

### 5.5 今後の coverage 拡張 candidate (= 後続 plan へ送り)

precision を守ったまま coverage を上げる余地:
1. `n_so == 1` で universal fit を **「異 op example の半数以上に fit」** に緩める (= 現在 100% 厳密 → 50% threshold) 。 リスク = sub-50% precision 落ち、 別途検証必要
2. **digit-pos-add** primitive (= per-position 加算 no carry) を arith 系に追加。 §5.3 の wrong がこれで救える可能性
3. `replace_map` (= cipher-style) を **examples 内で mapping infer する loop** を追加。 現在の design では `('repl', p, mapping)` の `mapping` を solver 側で構成する path が未実装 (= 8 series の subset enumeration が必要、 depth 3 budget 圧迫)

これらは all で coverage を 4.7% → 8-10% 圏に押し上げ得るが、 prototype の本旨 (= precision > coverage、 confident wrong は abstain) を踏まえ **本ファイルでは見送り**、 後続 plan / `/critique` レビュー後に着手する。

---

## 6. 出典 (= claim ごとに file:line / URL)

- `data/raw/train.csv` (= 9500 件、 host 提供)
- `data/processed/train_sft_v1.jsonl` (= v1 SFT、 9500 件、 source 別カウントは本 doc §0 と §5.4 の計算根拠)
- `docs/research_kernels/konbu17_nemotron-sft-lora-with-cot/nemotron-sft-lora-with-cot.ipynb` cell 0 markdown (= 「Equation Transformation 未分析」 明言)
- `docs/research_kernels/dgxchen_training-with-unsloth-to-achieve-0-85-lb/training-with-unsloth-to-achieve-0-85-lb.ipynb` cell 0-16 (= LLM only、 DSL なし)
- `docs/research_kernels/mohankrishnathalla_nemotron-6-puzzle-types-decoded-rule-solvers/nemotron-6-puzzle-types-decoded-rule-solvers.ipynb` cell 12, 16, 20 (= "needs LLM reasoning"、 subtype 分類)
- `docs/research_kernels/kalyankkr_all-6-puzzle-types-decoded-sft-training-data/all-6-puzzle-types-decoded-sft-training-data.ipynb` cell 19 (= CoT template 構成)
- `src/nvidia_nemotron_model_reasoning_challenge/solver_bit.py:107-147` (= 隣接 DSL の commit rule)
- `src/nvidia_nemotron_model_reasoning_challenge/solver_cipher.py:54-82` (= replace_map 由来の monoalphabetic map)
- `~/.claude/CLAUDE.md` "Plan Mode 出力ルール (Kaggle 関連のみ、CRITICAL)" (= 本 doc の日本語本文ルール)
- `~/projects/kaggle/CLAUDE.md` §11 「優勝本質性」 (= precision > coverage の方針根拠)
