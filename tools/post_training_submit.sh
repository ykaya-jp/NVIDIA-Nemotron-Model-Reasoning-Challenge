#!/usr/bin/env bash
# Post-training one-shot: adapter dir -> Kaggle Dataset -> inference kernel -> submit
#
# Pre-condition (= user does this once after Colab finishes):
#   1. In Drive, navigate to /MyDrive/nemotron-2026/adapter_v1/
#   2. Right-click the folder -> Download (= ZIP). Unzip locally.
#   3. The unzipped folder must contain exactly:
#        - adapter_config.json
#        - adapter_model.safetensors
#        - (tokenizer files are optional, harmless)
#
# Usage:
#   tools/post_training_submit.sh /absolute/path/to/adapter_v1
#
# What this script does:
#   (a) Creates or updates Kaggle Dataset `ky7240/nemotron-v1-adapter`
#   (b) Pushes the robust inference kernel (= submissions/v1_inference/)
#       which attaches that dataset and produces submission.zip
#   (c) Polls until the kernel finishes, then prints the LB score
#
# Idempotent: re-running with a new adapter increments the Kaggle Dataset
# version + the kernel version. The kernel always grabs the latest dataset.

set -euo pipefail

ADAPTER_DIR="${1:-}"
if [[ -z "$ADAPTER_DIR" ]]; then
    echo "Usage: $0 /absolute/path/to/adapter_v1" >&2
    exit 1
fi
if [[ ! -d "$ADAPTER_DIR" ]]; then
    echo "ERROR: $ADAPTER_DIR is not a directory" >&2
    exit 1
fi
for f in adapter_config.json adapter_model.safetensors; do
    if [[ ! -f "$ADAPTER_DIR/$f" ]]; then
        echo "ERROR: missing $ADAPTER_DIR/$f (required by official metric vLLM LoRARequest)" >&2
        exit 1
    fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer uv-managed kaggle if available; fall back to PATH kaggle.
if command -v uv >/dev/null 2>&1 && [[ -f pyproject.toml ]]; then
    K() { uv run kaggle "$@"; }
else
    K() { kaggle "$@"; }
fi

DATASET_SLUG="ky7240/nemotron-v1-adapter"
KERNEL_SLUG="ky7240/nemotron-v1-inference"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

echo "==> Stage adapter files at $STAGE_DIR"
cp "$ADAPTER_DIR/adapter_config.json"     "$STAGE_DIR/"
cp "$ADAPTER_DIR/adapter_model.safetensors" "$STAGE_DIR/"
cat > "$STAGE_DIR/dataset-metadata.json" <<EOF
{
  "title": "Nemotron v1 LoRA adapter (verifier-backed CoT, r=32)",
  "id": "$DATASET_SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
ls -lh "$STAGE_DIR"

# Step (a): Kaggle Dataset create-or-version
echo
echo "==> Step (a): Kaggle Dataset upload"
# Probe whether the dataset already exists. `datasets status` returns
# non-zero (403/404) if it doesn't exist yet — then we create it.
if K datasets status "$DATASET_SLUG" >/dev/null 2>&1; then
    echo "    dataset $DATASET_SLUG exists; pushing new version"
    K datasets version -p "$STAGE_DIR" -m "v1 adapter $(date -u +%Y-%m-%dT%H:%MZ)"
else
    echo "    dataset $DATASET_SLUG does not exist; creating"
    K datasets create -p "$STAGE_DIR" --public=false
fi

# Kaggle's dataset processing is async; wait for it to finish before
# attaching to the kernel. ~30-60s for a 200-400 MB adapter.
echo "    waiting for dataset to finish processing..."
for _ in {1..60}; do
    if K datasets status "$DATASET_SLUG" 2>/dev/null | grep -qi "ready\|complete"; then
        echo "    dataset ready"
        break
    fi
    sleep 5
done

# Step (b): push the inference kernel (= robust os.walk version)
echo
echo "==> Step (b): push inference kernel"
K kernels push -p submissions/v1_inference

# Step (c): poll until the kernel finishes
echo
echo "==> Step (c): poll kernel"
for _ in {1..120}; do
    STATUS=$(K kernels status "$KERNEL_SLUG" 2>&1 || true)
    echo "    $STATUS"
    if echo "$STATUS" | grep -qi 'complete'; then
        echo "    kernel completed"
        break
    fi
    if echo "$STATUS" | grep -qi 'error'; then
        echo "    kernel ERROR; downloading log"
        K kernels output "$KERNEL_SLUG" -p /tmp/post_training_log || true
        ls /tmp/post_training_log/
        exit 2
    fi
    sleep 15
done

# Submit by name = the kernel's own submission.zip is auto-attached
# via the "Submit to Competition" button. The CLI equivalent uses the
# kernel's most recent output.
echo
echo "==> Step (d): submit kernel output to competition"
K competitions submit -c nvidia-nemotron-model-reasoning-challenge \
    -f "$KERNEL_SLUG" \
    -m "v1 LoRA adapter (verifier-backed CoT, r=32, lr=1e-4, bit-13x)"

echo
echo "==> Submitted. Check leaderboard with:"
echo "    kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge"
echo
echo "Public LB target gates (= HANDOFF):"
echo "  >= 0.865  -> Phase 1 success, proceed to Phase 2"
echo "  0.850-0.864 -> partial, 1 targeted patch + resubmit"
echo "  <  0.850  -> abort, revert to public dgxchen baseline"
