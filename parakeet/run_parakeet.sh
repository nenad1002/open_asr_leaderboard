#!/bin/bash
set -euo pipefail

# Activate conda for NeMo
source ~/miniconda3/bin/activate

export PYTHONPATH="..:${PYTHONPATH:-}"

MODEL_ID="nvidia/parakeet-tdt-0.6b-v3-nemo-streaming"

BATCH_SIZE=1

DATASETS=(
  "librispeech    test.clean"
)

for entry in "${DATASETS[@]}"; do
  read -r DATASET SPLIT <<< "$entry"

  echo "============================================================"
  echo "  Dataset: ${DATASET}  Split: ${SPLIT}"
  echo "  Model: nvidia/parakeet-tdt-0.6b-v3 (NeMo official)"
  echo "============================================================"

  python run_parakeet.py \
    --pretrained_name="nvidia/parakeet-tdt-0.6b-v3" \
    --model_id="${MODEL_ID}" \
    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    --dataset="${DATASET}" \
    --split="${SPLIT}" \
    --batch_size="${BATCH_SIZE}" \
    --left_context_secs=9 \
    --chunk_secs=0.8 \
    --right_context_secs=1.6 \
    --max_eval_samples=-1
done

# Score all results
RUNDIR="$(pwd)"
cd ../normalizer
python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')"
cd "${RUNDIR}"
