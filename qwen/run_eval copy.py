import argparse
import os
import time
import tempfile
from typing import List

import numpy as np
import torch
import evaluate
from tqdm import tqdm

from normalizer import data_utils

# Qwen ASR wrapper
from qwen_asr import Qwen3ASRModel

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("medium")


def _device_from_arg(device: int) -> str:
    # Match your Phi script semantics: -1 => CPU, else cuda:<idx>
    if device is None or device < 0:
        return "cpu"
    return f"cuda:{device}"


def _write_wav_16k_mono(path: str, audio_array: np.ndarray, sampling_rate: int) -> None:
    """
    Write audio to WAV. We resample only if needed (simple path: assume already 16k, as ESB datasets usually are).
    If your dataset isn't 16k, install 'soxr' or use torchaudio for resampling.
    """
    # Lazy import so the script still starts even if soundfile isn't installed
    import soundfile as sf

    x = audio_array
    if isinstance(x, list):
        x = np.array(x)

    # Ensure float32
    x = x.astype(np.float32, copy=False)

    # If multi-channel, average to mono
    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32, copy=False)

    if sampling_rate != 16000:
        # Minimal, explicit error (don’t silently do a bad resample)
        raise ValueError(
            f"Expected 16kHz audio but got {sampling_rate}. "
            f"Either resample in data_utils.prepare_data(...) or add a resampler here."
        )

    sf.write(path, x, 16000, subtype="PCM_16")


def main(args):
    device_str = _device_from_arg(args.device)

    # Load Qwen ASR model
    # NOTE: Qwen3ASRModel uses device_map; pass "cuda" or "cpu".
    # For multi-GPU / sharding, you can pass device_map="auto".
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    model = Qwen3ASRModel.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="cuda" if device_str.startswith("cuda") else "cpu",
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        # attn_implementation="flash_attention_2",  # optional if you have it
    )

    # Load + optionally subsample dataset
    dataset = data_utils.load_data(args)

    print("[CUDA] available:", torch.cuda.is_available())
    print("[CUDA] device count:", torch.cuda.device_count())
    print("[CUDA] current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
    print("[CUDA] device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

    # Check where model weights live
    for name, param in model.model.named_parameters():
        print("[CUDA] first param device:", param.device)
        break

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = data_utils.prepare_data(dataset)

    # Temp directory for writing wav files (kept for debugging if requested)
    base_tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="qwen_asr_eval_")
    os.makedirs(base_tmp_dir, exist_ok=True)
    print("Temp wav dir:", base_tmp_dir)

    def benchmark(batch):
        """
        Phi-style benchmark() that:
        - receives a batch dict with "audio" and "norm_text"
        - produces "predictions", "references", and timing
        """
        audios = batch["audio"]  # list of {"array":..., "sampling_rate":...}
        minibatch_size = len(audios)

        # Write wav files for this batch
        wav_paths: List[str] = []
        for i, a in enumerate(audios):
            arr = a["array"]
            sr = a["sampling_rate"]
            wav_path = os.path.join(base_tmp_dir, f"sample_{time.time_ns()}_{i}.wav")
            _write_wav_16k_mono(wav_path, arr, sr)
            wav_paths.append(wav_path)

        # START TIMING (total batch runtime)
        start_time = time.time()

        # Qwen transcribe: pass list of paths
        # language: None => auto; or you can force "English"
        results = model.transcribe(
            audio=wav_paths,
            language=args.language if args.language else None,
        )

        runtime = time.time() - start_time

        # Gather predictions
        pred_text = [r.text if hasattr(r, "text") else str(r) for r in results]

        # after pred_text = [...]
        if os.environ.get("DEBUG_PRINT", "0") == "1":
            for j in range(min(2, len(pred_text))):
                print(f"\n[DEBUG] REF: {batch['norm_text'][j]}")
                print(f"[DEBUG] HYP: {pred_text[j]}\n")


        # Per-sample time (Phi script normalizes)
        batch["transcription_time_s"] = minibatch_size * [runtime / max(minibatch_size, 1)]

        # Normalize predictions the same way as the leaderboard harness
        batch["predictions"] = [data_utils.normalizer(p) for p in pred_text]
        batch["references"] = batch["norm_text"]

        # Optional cleanup
        if not args.keep_wavs:
            for p in wav_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        return batch

    # Warmup (optional)
    if args.warmup_steps is None and args.warmup_steps > 0:
        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = dataset.take(num_warmup_samples)
        else:
            warmup_dataset = dataset.select(range(min(num_warmup_samples, len(dataset))))

        warmup_iter = iter(
            warmup_dataset.map(
                benchmark,
                batch_size=max(1, args.batch_size // 2),
                batched=True,
                fn_kwargs={},
            )
        )
        for _ in tqdm(warmup_iter, desc="Warming up..."):
            continue

    # Timed evaluation
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

    result_iter = iter(dataset)
    for result in tqdm(result_iter, desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

    # Write manifest (WER + RTFx)
    manifest_path = data_utils.write_manifest(
        all_results["references"],
        all_results["predictions"],
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"],
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(references=all_results["references"], predictions=all_results["predictions"])
    wer = round(100 * wer, 2)
    rtfx = round(sum(all_results["audio_length_s"]) / max(sum(all_results["transcription_time_s"]), 1e-9), 2)

    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True, help="Model id, e.g. Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--dataset_path", type=str, default="esb/datasets", help="Dataset path, default esb/datasets")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. librispeech_asr")
    parser.add_argument("--split", type=str, default="test", help="Split name, e.g. test/validation")

    parser.add_argument("--device", type=int, default=0, help="-1 CPU, else GPU index")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataset.map")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Limit samples for quick tests")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable HF streaming mode")

    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for generation")
    parser.add_argument("--max_inference_batch_size", type=int, default=1, help="Qwen internal batch cap (-1 unlimited)")

    parser.add_argument("--warmup_steps", type=int, default=2, help="Warmup steps before timed runs")
    parser.add_argument("--language", type=str, default="", help='Force language, e.g. "English" (empty => auto)')
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Compute dtype")

    parser.add_argument("--tmp_dir", type=str, default="", help="Directory to store temporary wavs")
    parser.add_argument("--keep_wavs", action="store_true", help="Do not delete temp wavs (debugging)")

    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
