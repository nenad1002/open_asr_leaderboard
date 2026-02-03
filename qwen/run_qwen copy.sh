#!/bin/bash
set -euo pipefail

export PYTHONPATH="..:${PYTHONPATH:-}"

MODEL_IDs=("Qwen/Qwen3-ASR-1.7B")

BATCH_SIZE=64
MAX_NEW_TOKENS=256

# Qwen-specific knobs (your run_eval.py supports these)
MAX_INFERENCE_BATCH_SIZE=32
DTYPE="bf16"
LANG_DEFAULT=""          # empty => auto
LANG_ENGLISH="English"

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ )); do
  MODEL_ID=${MODEL_IDs[$i]}

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
    --language="${LANG_DEFAULT}"

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
    --language="${LANG_DEFAULT}"

  # Earnings22: force English (replaces Phi's "user_prompt to English")
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
    --language="${LANG_ENGLISH}"

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
    --language="${LANG_DEFAULT}"

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
    --language="${LANG_DEFAULT}"

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
    --language="${LANG_DEFAULT}"

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
    --language="${LANG_DEFAULT}"

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
    --language="${LANG_DEFAULT}"

  # Evaluate results (same style as Phi script)
  RUNDIR="$(pwd)" && \
  cd ../normalizer && \
  python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
  cd "${RUNDIR}"
done
