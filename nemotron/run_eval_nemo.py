"""
Nemotron Speech Streaming ASR evaluation using NeMo's CacheAwareStreamingAudioBuffer
+ conformer_stream_step — the official NeMo streaming inference path.

Usage:
  python3 run_eval_nemo.py \
    --model_id nvidia/nemotron-speech-streaming-en-0.6b \
    --dataset_path hf-audio/esb-datasets-test-only-sorted \
    --dataset librispeech --split test.clean
"""

import argparse
import json
import os
import shutil
import tempfile
import time

import numpy as np
import evaluate
import soundfile as sf
import torch
from tqdm import tqdm

from normalizer import data_utils

wer_metric = evaluate.load("wer")

SAMPLE_RATE = 16000


def extract_transcriptions(transcribed_texts):
    """Extract text from transcription hypotheses."""
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
    if isinstance(transcribed_texts[0], Hypothesis):
        return [h.text for h in transcribed_texts]
    return transcribed_texts


def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
    """Calculate tokens to drop after downsampling."""
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded


def perform_streaming(asr_model, streaming_buffer, compute_dtype,
                      pad_and_drop_preencoded=False):
    """Run streaming inference on the audio loaded in streaming_buffer."""
    batch_size = len(streaming_buffer.streams_length)

    cache_last_channel, cache_last_time, cache_last_channel_len = \
        asr_model.encoder.get_initial_cache_state(batch_size=batch_size)

    previous_hypotheses = None
    pred_out_stream = None

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
        with torch.inference_mode():
            chunk_audio = chunk_audio.to(compute_dtype)
            with torch.no_grad():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = asr_model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=pred_out_stream,
                    drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                        asr_model, step_num, pad_and_drop_preencoded),
                    return_transcription=True,
                )

    final_streaming_tran = extract_transcriptions(transcribed_texts)
    return final_streaming_tran


def main(args):
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device >= 0:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    # Keep model in float32 — CacheAwareStreamingAudioBuffer produces float32 mel.
    # Use torch.amp.autocast for mixed precision if needed.
    compute_dtype = torch.float32

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading NeMo model (streaming via conformer_stream_step): {args.model_id} ...")
    if args.model_id.endswith(".nemo"):
        asr_model = ASRModel.restore_from(args.model_id, map_location=device)
    else:
        asr_model = ASRModel.from_pretrained(args.model_id, map_location=device)

    asr_model.to(compute_dtype)
    asr_model.eval()

    # Disable CUDA graph decoder to avoid cu_call unpacking error on newer PyTorch
    from omegaconf import open_dict
    with open_dict(asr_model.cfg):
        asr_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    asr_model.change_decoding_strategy(asr_model.cfg.decoding)

    # ── Setup streaming params ───────────────────────────────────────────────
    chunk_size = args.chunk_size
    shift_size = args.shift_size
    left_chunks = args.left_chunks

    if hasattr(asr_model.encoder, 'streaming_cfg'):
        if chunk_size < 0:
            chunk_size = asr_model.encoder.streaming_cfg.last_channel_cache_size
        if shift_size < 0:
            shift_size = chunk_size
        print(f"Streaming params: chunk_size={chunk_size}, shift_size={shift_size}, left_chunks={left_chunks}")
        if args.chunk_size >= 0:
            asr_model.encoder.setup_streaming_params(
                chunk_size=chunk_size, left_chunks=left_chunks, shift_size=shift_size
            )
    else:
        if chunk_size < 0:
            raise ValueError("chunk_size must be specified for models without streaming_cfg")
        asr_model.encoder.setup_streaming_params(
            chunk_size=chunk_size, left_chunks=left_chunks, shift_size=shift_size
        )

    # ── Online normalization ─────────────────────────────────────────────────
    online_normalization = args.online_normalization
    if hasattr(asr_model.encoder, 'streaming_cfg'):
        if hasattr(asr_model.encoder.streaming_cfg, 'norm_before_caching'):
            if asr_model.encoder.streaming_cfg.norm_before_caching:
                online_normalization = True

    # ── Create streaming buffer ──────────────────────────────────────────────
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=online_normalization,
        pad_and_drop_preencoded=args.pad_and_drop_preencoded,
    )

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples")
        dataset = dataset.take(args.max_eval_samples)

    dataset = data_utils.prepare_data(dataset)

    # ── Evaluation loop ──────────────────────────────────────────────────────
    temp_dir = tempfile.mkdtemp()
    all_preds_so_far = []
    all_refs_so_far = []
    all_predictions = []
    all_references = []
    all_audio_lengths = []
    all_transcription_times = []

    batch_files = []
    batch_refs = []
    batch_durations = []
    sample_counter = 0
    batch_size = args.batch_size

    print("Starting streaming evaluation...")
    for sample_idx, sample in enumerate(tqdm(dataset, desc=f"Evaluating {args.dataset}")):
        audio_array = np.asarray(sample["audio"]["array"], dtype=np.float32)
        sampling_rate = sample["audio"]["sampling_rate"]
        duration = len(audio_array) / sampling_rate

        ref_text = sample["norm_text"]

        # Save audio to temp file (streaming_buffer needs file paths)
        temp_file = os.path.join(temp_dir, f"temp_{sample_idx}.wav")
        sf.write(temp_file, audio_array, sampling_rate)

        streaming_buffer.append_audio_file(temp_file, stream_id=-1)

        batch_files.append(temp_file)
        batch_refs.append(ref_text)
        batch_durations.append(duration)

        # Process batch when full or at end
        if len(batch_files) >= batch_size:
            t0 = time.time()
            with torch.inference_mode():
                streaming_tran = perform_streaming(
                    asr_model, streaming_buffer, compute_dtype,
                    pad_and_drop_preencoded=args.pad_and_drop_preencoded,
                )
            elapsed = time.time() - t0
            per_sample_time = elapsed / len(streaming_tran)

            # Normalize and record
            for i, (pred_raw, ref_norm, dur) in enumerate(
                zip(streaming_tran, batch_refs, batch_durations)
            ):
                sample_counter += 1
                pred_norm = data_utils.normalizer(pred_raw)

                if ref_norm.strip():
                    sample_wer = wer_metric.compute(
                        references=[ref_norm], predictions=[pred_norm]
                    )
                    sample_wer_pct = round(100 * sample_wer, 1)
                else:
                    sample_wer_pct = 0.0 if not pred_norm.strip() else 100.0

                all_preds_so_far.append(pred_norm)
                all_refs_so_far.append(ref_norm)
                agg_wer = wer_metric.compute(
                    references=all_refs_so_far, predictions=all_preds_so_far
                )
                agg_wer_pct = round(100 * agg_wer, 2)

                print(
                    f"  [{sample_counter:>4d}] "
                    f"[{dur:.1f}s, WER={sample_wer_pct:5.1f}%, aggWER={agg_wer_pct:5.2f}%]\n"
                    f"    HYP: {pred_raw}\n"
                    f"    REF: {ref_norm}"
                )

                all_predictions.append(pred_norm)
                all_references.append(ref_norm)
                all_audio_lengths.append(dur)
                all_transcription_times.append(per_sample_time)

            # Clean up
            for f in batch_files:
                os.remove(f)
            streaming_buffer.reset_buffer()
            batch_files = []
            batch_refs = []
            batch_durations = []

    # ── Process remaining samples ────────────────────────────────────────────
    if len(batch_files) > 0:
        t0 = time.time()
        with torch.inference_mode():
            streaming_tran = perform_streaming(
                asr_model, streaming_buffer, compute_dtype,
                pad_and_drop_preencoded=args.pad_and_drop_preencoded,
            )
        elapsed = time.time() - t0
        per_sample_time = elapsed / len(streaming_tran)

        for i, (pred_raw, ref_norm, dur) in enumerate(
            zip(streaming_tran, batch_refs, batch_durations)
        ):
            sample_counter += 1
            pred_norm = data_utils.normalizer(pred_raw)

            if ref_norm.strip():
                sample_wer = wer_metric.compute(
                    references=[ref_norm], predictions=[pred_norm]
                )
                sample_wer_pct = round(100 * sample_wer, 1)
            else:
                sample_wer_pct = 0.0 if not pred_norm.strip() else 100.0

            all_preds_so_far.append(pred_norm)
            all_refs_so_far.append(ref_norm)
            agg_wer = wer_metric.compute(
                references=all_refs_so_far, predictions=all_preds_so_far
            )
            agg_wer_pct = round(100 * agg_wer, 2)

            print(
                f"  [{sample_counter:>4d}] "
                f"[{dur:.1f}s, WER={sample_wer_pct:5.1f}%, aggWER={agg_wer_pct:5.2f}%]\n"
                f"    HYP: {pred_raw}\n"
                f"    REF: {ref_norm}"
            )

            all_predictions.append(pred_norm)
            all_references.append(ref_norm)
            all_audio_lengths.append(dur)
            all_transcription_times.append(per_sample_time)

        for f in batch_files:
            os.remove(f)
        streaming_buffer.reset_buffer()

    shutil.rmtree(temp_dir, ignore_errors=True)

    # ── Write manifest + compute metrics ─────────────────────────────────────
    model_id = args.model_id

    manifest_path = data_utils.write_manifest(
        all_references,
        all_predictions,
        model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_audio_lengths,
        transcription_time=all_transcription_times,
    )
    print("Results saved at:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_references,
        predictions=all_predictions,
    )
    wer = round(100 * wer, 2)

    total_audio = sum(all_audio_lengths)
    total_time = sum(all_transcription_times)
    rtfx = round(total_audio / max(total_time, 1e-9), 2)

    print(f"\n{'=' * 60}")
    print(f"  Model:   {args.model_id} (STREAMING via conformer_stream_step)")
    print(f"  Dataset: {args.dataset} / {args.split}")
    print(f"  WER:     {wer} %")
    print(f"  RTFx:    {rtfx}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nemotron Speech Streaming — NeMo conformer_stream_step evaluation"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="NeMo model identifier (HF hub or .nemo path)",
    )
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=int, default=0, help="-1 for CPU, 0+ for GPU")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (number of streams processed in parallel)")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    # Streaming params
    parser.add_argument("--chunk_size", type=int, default=-1,
                        help="Chunk size for streaming (-1 to use model default)")
    parser.add_argument("--shift_size", type=int, default=-1,
                        help="Shift size for streaming (-1 to use chunk_size)")
    parser.add_argument("--left_chunks", type=int, default=2,
                        help="Left chunks for streaming context (default: 2)")
    parser.add_argument("--online_normalization", action="store_true",
                        help="Use online normalization for streaming")
    parser.add_argument("--pad_and_drop_preencoded", action="store_true",
                        help="Pad and drop pre-encoded for streaming")
    parser.add_argument(
        "--no-streaming",
        dest="streaming",
        action="store_false",
        help="Disable HF dataset streaming",
    )
    args = parser.parse_args()
    parser.set_defaults(streaming=True)

    main(args)
