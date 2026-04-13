#!/bin/bash
# ── Run Nemotron ONNX Streaming ASR evaluation across all ESB datasets ──
#
# This script evaluates WER and RTFx. VAD is disabled by default.
# To enable VAD, set ENABLE_VAD=1 before invoking.
#
# Usage:
#   ./run_eval.sh                                                  # VAD disabled
#   ENABLE_VAD=1 ./run_eval.sh                                     # VAD enabled (default threshold 0.5)
#   ENABLE_VAD=1 VAD_THRESHOLD=0.3 ./run_eval.sh                   # VAD enabled with custom threshold
set -euo pipefail

export PYTHONPATH="..:${PYTHONPATH:-}"

MODEL_PATH="/datadisks/disk4/nebanfic/nemotron-cpu-int4"
MODEL_ID="nvidia/nemotron-speech-streaming-en-0.6b"

BATCH_SIZE=8
NUM_CORES=32

# VAD control — override via environment variables
ENABLE_VAD="${ENABLE_VAD:-0}"
VAD_THRESHOLD="${VAD_THRESHOLD:-0.5}"
VAD_MIN_SILENCE="${VAD_MIN_SILENCE_CHUNKS:-5}"

# Build VAD flags
VAD_FLAGS=()
if [[ "${ENABLE_VAD}" == "1" ]]; then
  VAD_FLAGS+=("--enable_vad")
  VAD_FLAGS+=("--vad_threshold=${VAD_THRESHOLD}")
  VAD_FLAGS+=("--vad_min_silence_chunks=${VAD_MIN_SILENCE}")
  echo ">> VAD: enabled (threshold=${VAD_THRESHOLD}, min_silence_chunks=${VAD_MIN_SILENCE})"
else
  echo ">> VAD: disabled"
fi

DATASETS=(
  "ami              test"
  "earnings22       test"
  "gigaspeech       test"
  "librispeech      test.clean"
  "librispeech      test.other"
  "tedlium          test"
  "voxpopuli        test"
  "spgispeech       test"
)

for entry in "${DATASETS[@]}"; do
  read -r DATASET SPLIT <<< "$entry"

  echo "============================================================"
  echo "  Dataset: ${DATASET}  Split: ${SPLIT}"
  echo "============================================================"

  taskset -c 0-$((NUM_CORES-1)) python run_eval.py \
    --model_path="${MODEL_PATH}" \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="${DATASET}" \
    --split="${SPLIT}" \
    --batch_size="${BATCH_SIZE}" \
    --max_eval_samples=-1 \
    --execution_provider="follow_config" \
    --num-cores="${NUM_CORES}" \
    "${VAD_FLAGS[@]+"${VAD_FLAGS[@]}"}"

  if [[ $? -ne 0 ]]; then
    echo "ERROR: Failed on dataset=${DATASET} split=${SPLIT}" >&2
    exit 1
  fi
done

# Score all results
RUNDIR="$(pwd)"
cd ../normalizer

if [[ "${ENABLE_VAD}" == "1" ]]; then
  SCORE_MODEL_ID="${MODEL_ID}_vad-t${VAD_THRESHOLD}"
else
  SCORE_MODEL_ID="${MODEL_ID}_novad"
fi

python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${SCORE_MODEL_ID}')"
cd "${RUNDIR}"
