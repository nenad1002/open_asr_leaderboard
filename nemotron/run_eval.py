"""
Nemotron Speech Streaming ASR evaluation for the Open ASR Leaderboard.

Uses onnxruntime-genai StreamingProcessor + Generator API to transcribe
audio samples from the ESB benchmark datasets and computes WER + RTFx.

Usage:
  python run_eval.py \
    --model_path /path/to/nemotron-speech-streaming-en-0.6b \
    --dataset_path hf-audio/esb-datasets-test-only-sorted \
    --dataset librispeech --split test.clean
"""

import argparse
import json
import os
import re
import time

import numpy as np
import evaluate
from tqdm import tqdm

import torch

from normalizer import data_utils

wer_metric = evaluate.load("wer")

SAMPLE_RATE = 16000

_vad_model = None
_vad_utils = None


def _get_vad_model():
    """Lazy-load and cache the Silero VAD model."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            force_reload=False, onnx=False,
        )
        _vad_model = model
        _vad_utils = utils
    return _vad_model, _vad_utils


def _apply_vad(audio: np.ndarray, sr: int, threshold: float, min_silence_chunks: int) -> np.ndarray:
    """Return only speech segments concatenated, using Silero VAD."""
    vad_model, utils = _get_vad_model()
    get_speech_timestamps = utils[0]

    audio_tensor = torch.from_numpy(audio)
    speech_timestamps = get_speech_timestamps(
        audio_tensor, vad_model,
        sampling_rate=sr,
        threshold=threshold,
        min_silence_duration_ms=min_silence_chunks * 30,  # ~30 ms per chunk
    )
    if not speech_timestamps:
        return audio  # no speech detected — return original

    segments = [audio[ts["start"]:ts["end"]] for ts in speech_timestamps]
    return np.concatenate(segments)


def _load_model_config(model_path: str):
    """Read sample_rate and chunk_samples from genai_config.json."""
    config_path = os.path.join(model_path, "genai_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    sample_rate = config["model"]["sample_rate"]
    chunk_samples = config["model"]["chunk_samples"]
    return sample_rate, chunk_samples


def _decode_tokens(generator, tokenizer_stream):
    """Decode all available tokens from the generator, returning the text."""
    text = ""
    while not generator.is_done():
        generator.generate_next_token()
        tokens = generator.get_next_tokens()
        if len(tokens) > 0:
            token_text = tokenizer_stream.decode(tokens[0])
            if token_text:
                text += token_text
    return text


def _transcribe_audio(processor, generator, tokenizer_stream, audio_array: np.ndarray, chunk_samples: int) -> str:
    """
    Run full streaming transcription on a single audio sample.
    Returns the final decoded text string.
    """
    audio = audio_array.astype(np.float32)
    full_transcript = ""

    # Process audio in streaming chunks
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start : start + chunk_samples].astype(np.float32)
        inputs = processor.process(chunk)
        if inputs is not None:
            generator.set_inputs(inputs)
            full_transcript += _decode_tokens(generator, tokenizer_stream)

    # Flush remaining audio
    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        full_transcript += _decode_tokens(generator, tokenizer_stream)

    return full_transcript.strip()


def main(args):
    import onnxruntime_genai as og

    num_cores = args.num_cores
    if num_cores:
        import onnxruntime as ort
        # Set ORT thread count to match pinned cores
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
        print(f"Pinned to {num_cores} cores (set OMP/MKL threads)")
    else:
        num_cores = os.cpu_count() or 1
        print(f"Using all {num_cores} cores")

    print(f"Loading model from {args.model_path} ...")

    sample_rate, chunk_samples = _load_model_config(args.model_path)
    print(f"  Sample rate: {sample_rate}, Chunk samples: {chunk_samples}")

    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            config.append_provider(args.execution_provider)
    model = og.Model(config)

    processor = og.StreamingProcessor(model)
    tokenizer = og.Tokenizer(model)
    params = og.GeneratorParams(model)

    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(
                range(min(args.max_eval_samples, len(dataset)))
            )

    dataset = data_utils.prepare_data(dataset)

    # Running accumulators for aggregated WER
    all_preds_so_far = []
    all_refs_so_far = []
    sample_counter = [0]

    if args.enable_vad:
        print(f"VAD: enabled (threshold={args.vad_threshold}, min_silence_chunks={args.vad_min_silence_chunks})")
    else:
        print("VAD: disabled")

    def benchmark(batch):
        audios = batch["audio"]

        predictions = []
        transcription_times = []

        for a in audios:
            audio_array = np.asarray(a["array"], dtype=np.float32)
            sr = a["sampling_rate"]
            if sr != sample_rate:
                import scipy.signal
                num_samples = int(len(audio_array) * sample_rate / sr)
                audio_array = scipy.signal.resample(audio_array, num_samples).astype(np.float32)

            if args.enable_vad:
                audio_array = _apply_vad(audio_array, sample_rate, args.vad_threshold, args.vad_min_silence_chunks)

            # Create fresh processor, generator, tokenizer_stream per utterance
            fresh_processor = og.StreamingProcessor(model)
            generator = og.Generator(model, params)
            tokenizer_stream = tokenizer.create_stream()

            # Time the full transcription (all chunks + flush)
            t0 = time.time()
            text = _transcribe_audio(fresh_processor, generator, tokenizer_stream, audio_array, chunk_samples)
            elapsed = time.time() - t0

            del generator
            del fresh_processor

            predictions.append(text)
            transcription_times.append(elapsed)

        # Normalize predictions the same way as the leaderboard harness
        norm_preds = [data_utils.normalizer(p) for p in predictions]
        norm_refs = batch["norm_text"]

        # Per-sample WER + running aggregate
        for i, (pred_raw, pred_norm, ref_norm, a, t_time) in enumerate(
            zip(predictions, norm_preds, norm_refs, audios, transcription_times)
        ):
            sample_counter[0] += 1
            audio_array = np.asarray(a["array"], dtype=np.float32)
            sr = a["sampling_rate"]
            audio_dur = len(audio_array) / sr

            # Per-sample RTFx
            sample_rtfx = round(audio_dur / max(t_time, 1e-9), 2)

            # Per-sample WER
            if ref_norm.strip():
                sample_wer = wer_metric.compute(
                    references=[ref_norm], predictions=[pred_norm]
                )
                sample_wer_pct = round(100 * sample_wer, 1)
            else:
                sample_wer_pct = 0.0 if not pred_norm.strip() else 100.0

            # Update running aggregates
            all_preds_so_far.append(pred_norm)
            all_refs_so_far.append(ref_norm)
            agg_wer = wer_metric.compute(
                references=all_refs_so_far, predictions=all_preds_so_far
            )
            agg_wer_pct = round(100 * agg_wer, 2)

            print(
                f"  [{sample_counter[0]:>4d}] "
                f"[{audio_dur:.1f}s, WER={sample_wer_pct:5.1f}%, aggWER={agg_wer_pct:5.2f}%, "
                f"RTFx={sample_rtfx:6.2f}]\n"
                f"    HYP: {pred_raw}\n"
                f"    REF: {ref_norm}"
            )

        batch["predictions"] = norm_preds
        batch["references"] = norm_refs
        batch["transcription_time_s"] = transcription_times

        return batch

    # eval
    dataset = dataset.map(
        benchmark,
        batch_size=args.batch_size,
        batched=True,
        remove_columns=["audio"],
    )

    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }

    for result in tqdm(dataset, desc="Samples"):
        for key in all_results:
            all_results[key].append(result[key])

    # Model ID for the manifest filename — use directory basename
    model_id = args.model_id or os.path.basename(os.path.normpath(args.model_path))

    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=all_results["references"],
        predictions=all_results["predictions"],
    )
    wer = round(100 * wer, 2)

    total_audio = sum(all_results["audio_length_s"])
    total_time = sum(all_results["transcription_time_s"])
    rtfx = round(total_audio / max(total_time, 1e-9), 2)
    rtfx_per_core = round(rtfx / num_cores, 2)

    print(f"WER: {wer} %  RTFx: {rtfx}  RTFx/core: {rtfx_per_core} ({num_cores} cores)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nemotron Speech Streaming ASR — Open ASR Leaderboard evaluation"
    )

    # Model
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the Nemotron ONNX model directory",
    )
    parser.add_argument(
        "--model_id", type=str, default=None,
        help="Model identifier for result filenames (default: model dir basename)",
    )
    parser.add_argument(
        "--execution_provider", type=str, default="follow_config",
        help="ORT execution provider (cpu, cuda, dml, follow_config)",
    )
    parser.add_argument(
        "--num-cores", type=int, default=None,
        help="Number of CPU cores to use. Sets OMP/MKL threads. Use with 'taskset -c 0-N' for pinning.",
    )

    # Dataset
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--no-streaming", dest="streaming", action="store_false",
        help="Disable HF streaming mode",
    )
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for dataset.map (samples processed sequentially inside)")

    # VAD
    parser.add_argument("--enable_vad", action="store_true",
                        help="Enable Voice Activity Detection to strip silence before transcription")
    parser.add_argument("--vad_threshold", type=float, default=0.5,
                        help="Silero-VAD speech probability threshold (default: 0.5)")
    parser.add_argument("--vad_min_silence_chunks", type=int, default=5,
                        help="Minimum number of consecutive silence chunks to split on (default: 5)")

    parser.set_defaults(streaming=True)
    args = parser.parse_args()

    main(args)
