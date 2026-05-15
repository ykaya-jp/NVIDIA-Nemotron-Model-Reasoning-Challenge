# 2026-05-15 致命的 postmortem: submission format 完全誤解

> 起源: Colab quota 切れで公開 adapter baseline submit を試みた際、 我々の既往 submission kernel (= submissions/v1_inference/inference.py、 dc77aee で push) が submission.csv 形式 を採用していたが、 公式は **adapter zip 形式** だったと判明
> 親 CLAUDE.md §0.5 「silent except 禁止 / Codex review 必須化」 + lessons.md 「smoke PASS だけで真の動作と誤判定」

## 1. 真の submission 仕様

公式 reference: `docs/research_kernels/ryanholbrook_submission_demo/nvidia-nemotron-submission-demo.ipynb`

```python
model.save_pretrained(OUTPUT_DIR="/kaggle/working")  # adapter_config.json + adapter_model.safetensors
subprocess.run("zip -m submission.zip *", shell=True, check=True)
```

→ `/kaggle/working/submission.zip` に LoRA adapter (= adapter_config.json + adapter_model.safetensors) を入れて kernel COMPLETE。 公式 metric notebook が この zip を unzip → vLLM + LoRARequest で inference。

## 2. 我々がやっていたこと (= 完全な誤解)

`submissions/v1_inference/inference.py` (= dc77aee):

```python
LORA_DIR = "/kaggle/input/nemotron-v1-adapter"
submission = pd.DataFrame({row_id_col: test[row_id_col], "prediction": LORA_DIR})
submission.to_csv("/kaggle/working/submission.csv", index=False)
```

= test.csv の各 row に対して LORA_DIR の文字列を「prediction」 として書く CSV。 これは **完全に間違った format**。 仮に v1 adapter の training が完走していても、 この CSV を提出していたら LB は 0 か reject か、 submit-fail 扱いだった。

## 3. なぜ Day 0 で見逃したか

Day 0 の plan v3 §0 で公式 metric notebook を解析、 「rel_tol=1e-2、 hybrid 不可、 max_model_len=4096、 ...」 と finding 列挙。 ただし **submission の出力 format** (= CSV vs zip) は確認していなかった。 metric notebook の input 仕様 のみ見て output 仕様を見落とした。

ryanholbrook_submission_demo は Day 0 で repo に import 済 (= `docs/research_kernels/ryanholbrook_submission_demo/`)、 ただし read していなかった。

## 4. 影響

- **我々の v1 adapter training が完走していたとしても、 submission は fail**
- 16h Colab training が無駄になる潜在 risk があった (= 結果的に session 落ちで救われた皮肉)
- Day 0 の HANDOFF doc + plan v3 + winning-strategy.dense.md すべて この誤解前提で書かれている、 全部見直し必要

## 5. 修正

1. ✅ `submissions/baseline_konbu17/inference.py` を adapter zip 形式に書き直し (= 本 commit)
2. ⏳ `submissions/v1_inference/inference.py` (= 元の自前 v1 用) も同 format に書き直す
3. ⏳ `notebooks/train_lora_colab.py` の Cell 6 (= `model.save_pretrained(ADAPTER_DIR)` で Drive 保存) は OK、 ただし **Drive から Kaggle Dataset 化 → kernel で copy + zip → submission** の flow を README + HANDOFF に明記
4. ⏳ HANDOFF-2026-05-14-research-summary.md §7 の 3 scenario action path を再点検

## 6. 教訓

- **公式 submission demo notebook は必読** (= input + output 両 仕様)。 metric notebook は input 解析だが、 submission demo は output 仕様を示す
- 「動作確認できないものを Day 0 で push した」 = dc77aee 「GPU-free inference kernel」 を実走させずに push した、 deployability test を skip した
- 親 CLAUDE.md「動的スモーク必須ルール」 違反: 「Server Action の型が通っているから動くはず」 と同じ pattern (= submission.csv 出力できるから動くはず → 公式 metric の入力 format と不整合 silent fail)
- **lesson 追加**: Kaggle 案件で submission kernel を初めて push する時、 必ず「official submission demo / sample submission」 の 出力 format と byte-identical か確認

## 7. 親 CLAUDE.md への lesson 追加候補

```
- [2026-05-15] nemotron Day 0 で submission を CSV 形式で書いた → 公式は adapter zip 形式、 16h training 完走 してたら全 submission 仕様外で reject。 救いは Colab session 落ちで気づいた。 教訓: Kaggle 案件で submission kernel push 前に必ず公式 submission demo / sample submission の 出力 format を byte-identical 確認、 metric notebook (= input 仕様) だけ見るのは不十分
```

## 出典

- 公式 submission demo: `docs/research_kernels/ryanholbrook_submission_demo/nvidia-nemotron-submission-demo.ipynb`
- 我々の誤った v1 inference: `submissions/v1_inference/inference.py` (= dc77aee で push、 修正待ち)
- 本日修正済: `submissions/baseline_konbu17/inference.py` (= 本 commit)
