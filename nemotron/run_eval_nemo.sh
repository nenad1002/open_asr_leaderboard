#!/bin/bash
set -euo pipefail

export PYTHONPATH="..:${PYTHONPATH:-}"

MODEL_ID="nvidia/nemotron-speech-streaming-en-0.6b"
BATCH_SIZE=1
DEVICE_ID=0

CHUNK_SIZE=7
SHIFT_SIZE=7
LEFT_CHUNKS=10

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

  python3 run_eval_nemo.py \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="${DATASET}" \
    --split="${SPLIT}" \
    --device="${DEVICE_ID}" \
    --batch_size="${BATCH_SIZE}" \
    --chunk_size="${CHUNK_SIZE}" \
    --shift_size="${SHIFT_SIZE}" \
    --left_chunks="${LEFT_CHUNKS}" \
    --max_eval_samples=-1
done

# Score all results
RUNDIR="$(pwd)"
cd ../normalizer
python3 -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
cd "${RUNDIR}"
