#!/bin/bash
set -euo pipefail

export PYTHONPATH="..:${PYTHONPATH:-}"

MODEL_IDs=("Qwen/Qwen3-ASR-0.6B")

BATCH_SIZE=64
MAX_NEW_TOKENS=256

# Qwen-specific knobs (run_eval.py supports these)
MAX_INFERENCE_BATCH_SIZE=32
DTYPE="bf16"
LANG_DEFAULT=""          # empty => auto
LANG_ENGLISH="English"

# ---- real streaming-style stitching (time-based) ----
# Constraint: chunk_s <= 3. overlap must be < chunk_s.
STREAMING_CHUNKS=1
CHUNK_S=3
STRIDE_S=1

# ---- timestamps + forced aligner (required for time-based stitching) ----
RETURN_TIME_STAMPS=1
FORCED_ALIGNER="Qwen/Qwen3-ForcedAligner-0.6B"
FORCED_ALIGNER_DEVICE_MAP="cuda:0"
FORCED_ALIGNER_DTYPE="bf16"

# ---- stitching controls in the UPDATED run_eval.py ----
# Use strict timestamp authority (no invented time fallback by default)
STITCH_MODE="left"     # "left" or "half"
MAX_OVERLAP_K=12
TIME_SLACK_S=0.08
USE_TEXT_FALLBACK=0    # 0 => do NOT add --use_text_fallback

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ )); do
  MODEL_ID=${MODEL_IDs[$i]}

  chunk_flags=()
  if [[ "${STREAMING_CHUNKS}" == "1" ]]; then
    chunk_flags+=(--streaming_chunks --chunk_s "${CHUNK_S}" --stride_s "${STRIDE_S}")
    # only enable if you see missing time_stamps occasionally
    if [[ "${USE_TEXT_FALLBACK}" == "1" ]]; then
      chunk_flags+=(--use_text_fallback)
    fi

    # timestamp-based stitcher knobs
    chunk_flags+=(--max_overlap_k "${MAX_OVERLAP_K}")
  fi

  ts_flags=()
  if [[ "${RETURN_TIME_STAMPS}" == "1" ]]; then
    ts_flags+=(
      --forced_aligner "${FORCED_ALIGNER}"
      --forced_aligner_device_map "${FORCED_ALIGNER_DEVICE_MAP}"
      --forced_aligner_dtype "${FORCED_ALIGNER_DTYPE}"
    )
  fi

  echo "----------------------------------------------------------------"
  echo "Model: ${MODEL_ID}"
  echo "Streaming: ${STREAMING_CHUNKS}  chunk_s=${CHUNK_S}  stride_s=${STRIDE_S}  stitch_mode=${STITCH_MODE}"
  echo "Batch: batch_size=${BATCH_SIZE}  max_inference_batch_size=${MAX_INFERENCE_BATCH_SIZE}"
  echo "----------------------------------------------------------------"

  echo "Running VoxPopuli..."
  python run_eval.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="voxpopuli" \
    --split="test" \
    --device=0 \
    --batch_size=$((5 * BATCH_SIZE / 10)) \
    --max_eval_samples=-1 \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --max_inference_batch_size="${MAX_INFERENCE_BATCH_SIZE}" \
    --dtype="${DTYPE}" \
    --language="${LANG_DEFAULT}" \
    "${chunk_flags[@]}" \
    "${ts_flags[@]}"

  echo "Running AMI..."
  python run_eval.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="ami" \
    --split="test" \
    --device=0 \
    --batch_size="${BATCH_SIZE}" \
    --max_eval_samples=-1 \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --max_inference_batch_size="${MAX_INFERENCE_BATCH_SIZE}" \
    --dtype="${DTYPE}" \
    --language="${LANG_DEFAULT}" \
    "${chunk_flags[@]}" \
    "${ts_flags[@]}"

  echo "Running Earnings22..."
  python run_eval.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="earnings22" \
    --split="test" \
    --device=0 \
    --batch_size="${BATCH_SIZE}" \
    --max_eval_samples=-1 \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --max_inference_batch_size="${MAX_INFERENCE_BATCH_SIZE}" \
    --dtype="${DTYPE}" \
    --language="${LANG_ENGLISH}" \
    "${chunk_flags[@]}" \
    "${ts_flags[@]}"

  echo "Running GigaSpeech..."
  python run_eval.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="gigaspeech" \
    --split="test" \
    --device=0 \
    --batch_size=$((9 * BATCH_SIZE / 10)) \
    --max_eval_samples=-1 \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --max_inference_batch_size="${MAX_INFERENCE_BATCH_SIZE}" \
    --dtype="${DTYPE}" \
    --language="${LANG_DEFAULT}" \
    "${chunk_flags[@]}" \
    "${ts_flags[@]}"

  echo "Running LibriSpeech Clean..."
  python run_eval.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" \
    --split="test.clean" \
    --device=0 \
    --batch_size="${BATCH_SIZE}" \
    --max_eval_samples=-1 \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --max_inference_batch_size="${MAX_INFERENCE_BATCH_SIZE}" \
    --dtype="${DTYPE}" \
    --language="${LANG_DEFAULT}" \
    "${chunk_flags[@]}" \
    "${ts_flags[@]}"

  echo "Running LibriSpeech Other..."
  python run_eval.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="librispeech" \
    --split="test.other" \
    --device=0 \
    --batch_size="${BATCH_SIZE}" \
    --max_eval_samples=-1 \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --max_inference_batch_size="${MAX_INFERENCE_BATCH_SIZE}" \
    --dtype="${DTYPE}" \
    --language="${LANG_DEFAULT}" \
    "${chunk_flags[@]}" \
    "${ts_flags[@]}"

  echo "Running SPGISpeech..."
  python run_eval.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="spgispeech" \
    --split="test" \
    --device=0 \
    --batch_size=$((9 * BATCH_SIZE / 10)) \
    --max_eval_samples=-1 \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --max_inference_batch_size="${MAX_INFERENCE_BATCH_SIZE}" \
    --dtype="${DTYPE}" \
    --language="${LANG_DEFAULT}" \
    "${chunk_flags[@]}" \
    "${ts_flags[@]}"

  echo "Running TED-LIUM..."
  python run_eval.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="tedlium" \
    --split="test" \
    --device=0 \
    --batch_size="${BATCH_SIZE}" \
    --max_eval_samples=-1 \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --max_inference_batch_size="${MAX_INFERENCE_BATCH_SIZE}" \
    --dtype="${DTYPE}" \
    --language="${LANG_DEFAULT}" \
    "${chunk_flags[@]}" \
    "${ts_flags[@]}"

  # Evaluate results
  RUNDIR="$(pwd)"
  cd ../normalizer
  python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
  cd "${RUNDIR}"
done
