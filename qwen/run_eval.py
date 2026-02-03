import argparse
import os
import time
import tempfile
import re
import unicodedata
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
import evaluate
from tqdm import tqdm

from normalizer import data_utils
from qwen_asr import Qwen3ASRModel

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision("medium")


# ============================================================
# Device / audio I/O
# ============================================================

def _device_from_arg(device: int) -> str:
    if device is None or device < 0:
        return "cpu"
    return f"cuda:{device}"


def _write_wav_16k_mono(path: str, audio_array: np.ndarray, sampling_rate: int) -> None:
    import soundfile as sf

    x = audio_array
    if isinstance(x, list):
        x = np.array(x)
    x = x.astype(np.float32, copy=False)

    if x.ndim == 2:
        x = x.mean(axis=1).astype(np.float32, copy=False)

    if sampling_rate != 16000:
        raise ValueError(
            f"Expected 16kHz audio but got {sampling_rate}. "
            f"Either resample in data_utils.prepare_data(...) or add a resampler here."
        )

    sf.write(path, x, 16000, subtype="PCM_16")


# ============================================================
# Chunking (returns start time and is_last flag)
# ============================================================

def _chunk_audio_16k(
    x: np.ndarray,
    sr: int,
    chunk_s: float,
    stride_s: float,
) -> List[Tuple[int, int, float, bool, np.ndarray]]:
    """
    Returns list of (start_sample, end_sample, start_time_sec, is_last, chunk_array)
    """
    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio but got {sr}")

    if isinstance(x, list):
        x = np.array(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)

    chunk_len = int(round(chunk_s * sr))
    stride_len = int(round(stride_s * sr))
    if chunk_len <= 0:
        raise ValueError("chunk_s must be > 0")
    if stride_len < 0 or stride_len >= chunk_len:
        raise ValueError("stride_s must be >= 0 and < chunk_s")

    step = chunk_len - stride_len
    n = x.shape[0]
    out: List[Tuple[int, int, float, bool, np.ndarray]] = []

    start = 0
    while start < n:
        end = min(start + chunk_len, n)
        t0 = start / sr
        is_last = (end == n)
        out.append((start, end, t0, is_last, x[start:end]))

        if is_last:
            break
        start += step

    return out


# ============================================================
# Word helpers
# ============================================================

_ws = re.compile(r"\s+")
_strip_edges = re.compile(r"^[^\w']+|[^\w']+$")


def _split_words(text: str) -> List[str]:
    text = _ws.sub(" ", (text or "").strip())
    if not text:
        return []
    words = []
    for w in text.split(" "):
        w2 = _strip_edges.sub("", w)
        if w2:
            words.append(w2)
    return words


def _canon_word(w: str) -> str:
    w = unicodedata.normalize("NFKD", w)
    w = "".join(ch for ch in w if not unicodedata.combining(ch))
    w = w.lower()
    out = []
    for ch in w:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "'":
            out.append(ch)
    return "".join(out)


def _drop_prefix_overlap(committed_words: List[str], new_words: List[str], max_k: int = 12) -> List[str]:
    """
    Drops the largest prefix of new_words that matches a suffix of committed_words (canonized).
    Useful for repeats like "oppose the oppose the".
    """
    if not committed_words or not new_words:
        return new_words

    c = [_canon_word(x) for x in committed_words[-max_k:]]
    n = [_canon_word(x) for x in new_words[:max_k]]

    best = 0
    lim = min(len(c), len(n))
    for k in range(1, lim + 1):
        if c[-k:] == n[:k]:
            best = k
    return new_words[best:]


# ============================================================
# Timestamp extraction from ForcedAlignResult
# ============================================================

def _extract_forced_align_items(r: Any) -> List[Tuple[float, float, str]]:
    """
    Returns list of (start_time, end_time, text) in CHUNK-LOCAL seconds.
    Handles:
      - r.time_stamps = ForcedAlignResult(items=[ForcedAlignItem(...), ...])
      - r.time_stamps might already be a list of items
    """
    ts = getattr(r, "time_stamps", None)
    if ts is None:
        return []

    items = None
    if hasattr(ts, "items"):
        items = getattr(ts, "items", None)
    else:
        items = ts

    if not items:
        return []

    out: List[Tuple[float, float, str]] = []
    for it in items:
        # ForcedAlignItem(text='I', start_time=0.0, end_time=0.56)
        if hasattr(it, "start_time") and hasattr(it, "end_time") and hasattr(it, "text"):
            try:
                s = float(getattr(it, "start_time"))
                e = float(getattr(it, "end_time"))
                w = str(getattr(it, "text"))
            except Exception:
                continue
            if w:
                out.append((s, e, w))
            continue

        # fallback dict/tuple forms
        if isinstance(it, dict):
            if "start_time" in it and "end_time" in it and "text" in it:
                try:
                    s = float(it["start_time"])
                    e = float(it["end_time"])
                    w = str(it["text"])
                except Exception:
                    continue
                if w:
                    out.append((s, e, w))
        elif isinstance(it, (tuple, list)) and len(it) >= 3:
            try:
                s = float(it[0])
                e = float(it[1])
                w = str(it[2])
            except Exception:
                continue
            if w:
                out.append((s, e, w))

    return out


def _to_global_words(meta: Dict, r: Any) -> List[Tuple[float, float, str]]:
    """
    Convert forced-align items for a chunk into GLOBAL times:
      returns list of (start_global, end_global, word) sorted by start.
    """
    t0 = float(meta["start_t"])
    items = _extract_forced_align_items(r)
    out: List[Tuple[float, float, str]] = []
    for s_local, e_local, w in items:
        s = t0 + float(s_local)
        e = t0 + float(e_local)
        out.append((s, e, w))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


# ============================================================
# Stitcher implementing EXACT policy you described
# ============================================================

class ExactBoundaryStitcher:
    """
    EXACT policy:

    Boundary is at t1 (end of left chunk i): t1 = start_t + len_s for chunk i.

    Left chunk:
      - commit words that end <= t1
      - let s1 be end time of last committed word (in global time)

    Right chunk:
      - commit words that start >= t1
      - let s2 be start time of first such word (in global time; if none, +inf)

    Bridge words (boundary-split words):
      - commit words (from the RIGHT chunk) whose start >= s1 and end <= s2
        (these lie between last safe-left and first safe-right)
    """

    def __init__(self, max_overlap_k: int = 12, eps_s: float = 1e-6):
        self.max_overlap_k = int(max_overlap_k)
        self.eps_s = float(eps_s)

    def stitch(self, chunks: List[Tuple[Dict, Any]]) -> str:
        if not chunks:
            return ""

        # Precompute global words per chunk
        gw: List[List[Tuple[float, float, str]]] = []
        metas: List[Dict] = []
        for meta, r in chunks:
            metas.append(meta)
            gw.append(_to_global_words(meta, r))

        out_words: List[str] = []

        # If only one chunk, just emit all words from it
        if len(chunks) == 1:
            out_words = [w for _, _, w in gw[0]]
            return " ".join(out_words).strip()

        # Helper: append with small dedup for immediate repeats / overlap prefix
        def _append_words(words: List[str]):
            nonlocal out_words
            if not words:
                return
            # n-gram style prefix overlap drop to combat duplication
            words2 = _drop_prefix_overlap(out_words, words, max_k=self.max_overlap_k)
            for w in words2:
                if out_words and _canon_word(out_words[-1]) == _canon_word(w):
                    continue
                out_words.append(w)

        # Process boundary-by-boundary
        for i in range(len(chunks) - 1):
            meta_L = metas[i]
            meta_R = metas[i + 1]
            words_L = gw[i]
            words_R = gw[i + 1]

            t0_L = float(meta_L["start_t"])
            t1_L = t0_L + float(meta_L["len_s"])  # EXACT boundary at end of left chunk

            # ---- Left commit: end <= t1 ----
            left_commit = [w for (s, e, w) in words_L if e <= t1_L + self.eps_s]
            # s1 = end time of last committed word (or t0_L if none)
            s1 = (max((e for (s, e, w) in words_L if e <= t1_L + self.eps_s), default=t0_L))

            # Append left commit only once (for i==0). For i>0, left_commit would re-emit already emitted words,
            # so we do NOT re-append all of it. We append only "new" words by time.
            if i == 0:
                _append_words(left_commit)
            else:
                # only append words in left chunk whose start is after the last boundary t1 of previous chunk.
                # We can approximate using the last emitted time by s1_prev stored as last_end_out.
                # We'll instead append nothing here; bridging + right side will cover continuity.
                pass

            # ---- Right commit: start >= t1 ----
            right_post = [(s, e, w) for (s, e, w) in words_R if s >= t1_L - self.eps_s]
            s2 = right_post[0][0] if right_post else float("inf")

            # ---- Bridge words: start >= s1 AND end <= s2 (from RIGHT chunk) ----
            # This is exactly your "split words live between s1 and s2".
            bridge = [w for (s, e, w) in words_R if (s >= s1 - self.eps_s) and (e <= s2 + self.eps_s)]

            # Append bridge first (fills any gap between last left word and first right-post word)
            _append_words(bridge)

            # Then append right-post (everything after boundary)
            _append_words([w for (_, _, w) in right_post])

        return " ".join(out_words).strip()


# ============================================================
# Main
# ============================================================

def main(args):
    device_str = _device_from_arg(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    forced_aligner = args.forced_aligner if args.forced_aligner else None
    forced_aligner_kwargs = None
    if forced_aligner:
        fa_dtype = torch.bfloat16 if args.forced_aligner_dtype == "bf16" else torch.float16
        forced_aligner_kwargs = dict(
            dtype=fa_dtype,
            device_map=args.forced_aligner_device_map,
        )

    model = Qwen3ASRModel.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="cuda" if device_str.startswith("cuda") else "cpu",
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
        forced_aligner=forced_aligner,
        forced_aligner_kwargs=forced_aligner_kwargs,
    )

    dataset = data_utils.load_data(args)

    base_tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="qwen_asr_eval_")
    os.makedirs(base_tmp_dir, exist_ok=True)
    print("Temp wav dir:", base_tmp_dir)

    def _transcribe(wav_paths: List[str]):
        # IMPORTANT: request timestamps; CLI doesn't need a flag.
        return model.transcribe(
            audio=wav_paths,
            language=args.language if args.language else None,
            return_time_stamps=True,
        )

    def _get_refs_and_key(batch: Dict) -> Tuple[List[str], str]:
        for k in ["norm_text", "text", "sentence"]:
            if k in batch:
                return batch[k], k
        return [""] * len(batch.get("audio", [])), ""

    def benchmark(batch: Dict):
        audios = batch["audio"]
        n_samples = len(audios)

        # audio length for RTFx
        audio_lengths_s: List[float] = []
        for a in audios:
            arr = a["array"]
            sr = a["sampling_rate"]
            audio_lengths_s.append(float(len(arr)) / float(sr))
        batch["audio_length_s"] = audio_lengths_s

        wav_paths: List[str] = []
        chunk_metadata: List[Dict] = []
        per_owner_chunk_audio_s = [0.0] * n_samples

        if args.streaming_chunks:
            for i, a in enumerate(audios):
                arr = a["array"]
                sr = a["sampling_rate"]
                chunks = _chunk_audio_16k(arr, sr, args.chunk_s, args.stride_s)

                for start_sample, end_sample, t0, is_last, chunk_arr in chunks:
                    wav_path = os.path.join(base_tmp_dir, f"chunk_{time.time_ns()}_{i}.wav")
                    _write_wav_16k_mono(wav_path, chunk_arr, 16000)
                    wav_paths.append(wav_path)

                    chunk_len_s = (end_sample - start_sample) / 16000.0
                    per_owner_chunk_audio_s[i] += chunk_len_s

                    chunk_metadata.append({
                        "owner_idx": i,
                        "start_t": float(t0),
                        "len_s": float(chunk_len_s),
                        "is_last": bool(is_last),
                    })
        else:
            for i, a in enumerate(audios):
                arr = a["array"]
                sr = a["sampling_rate"]
                wav_path = os.path.join(base_tmp_dir, f"sample_{time.time_ns()}_{i}.wav")
                _write_wav_16k_mono(wav_path, arr, sr)
                wav_paths.append(wav_path)

                per_owner_chunk_audio_s[i] = float(len(arr)) / float(sr)

                chunk_metadata.append({
                    "owner_idx": i,
                    "start_t": 0.0,
                    "len_s": float(len(arr)) / float(sr),
                    "is_last": True,
                })

        # Transcribe
        start_time = time.time()
        results = _transcribe(wav_paths)
        runtime = time.time() - start_time

        # Group by sample
        by_sample: List[List[Tuple[Dict, Any]]] = [[] for _ in range(n_samples)]
        for meta, res in zip(chunk_metadata, results):
            by_sample[meta["owner_idx"]].append((meta, res))

        pred_texts: List[str] = [""] * n_samples
        for i in range(n_samples):
            by_sample[i].sort(key=lambda x: x[0]["start_t"])

            if args.streaming_chunks:
                stitcher = ExactBoundaryStitcher(
                    max_overlap_k=args.max_overlap_k,
                    eps_s=1e-6,
                )
                pred_texts[i] = stitcher.stitch(by_sample[i])
            else:
                pred_texts[i] = getattr(by_sample[i][0][1], "text", "") if by_sample[i] else ""

        # Time attribution: weight by audio seconds processed (not chunk count)
        total_weight = sum(per_owner_chunk_audio_s) or 1e-9
        per_sample_times = [runtime * (w / total_weight) for w in per_owner_chunk_audio_s]
        batch["transcription_time_s"] = per_sample_times

        # References (normalize symmetrically)
        ref_texts_raw, ref_key = _get_refs_and_key(batch)
        if ref_key == "norm_text":
            ref_texts = ref_texts_raw
        else:
            ref_texts = [data_utils.normalizer(r) for r in ref_texts_raw]

        preds_norm = [data_utils.normalizer(p) for p in pred_texts]

        if os.environ.get("DEBUG_PRINT", "0") == "1":
            for j in range(min(8, n_samples)):
                print(f"\n[DEBUG] REF: {ref_texts[j]}")
                print(f"[DEBUG] HYP: {preds_norm[j]}\n")

        batch["predictions"] = preds_norm
        batch["references"] = ref_texts

        if not args.keep_wavs:
            for p in wav_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        return batch

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

    for result in tqdm(iter(dataset), desc="Samples..."):
        for key in all_results:
            all_results[key].append(result[key])

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

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_inference_batch_size", type=int, default=1)

    parser.add_argument("--language", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--tmp_dir", type=str, default="")
    parser.add_argument("--keep_wavs", action="store_true")

    # chunked streaming-like feeding
    parser.add_argument("--streaming_chunks", action="store_true")
    parser.add_argument("--chunk_s", type=float, default=3.0)
    parser.add_argument("--stride_s", type=float, default=0.75)

    # forced aligner (required for word timestamps)
    parser.add_argument("--forced_aligner", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--forced_aligner_device_map", type=str, default="cuda:0")
    parser.add_argument("--forced_aligner_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # stitch controls (kept for your bash script compatibility)
    parser.add_argument("--max_overlap_k", type=int, default=12)

    args = parser.parse_args()

    # IMPORTANT FIX: normalizer/data_utils.py expects args.streaming to exist.
    args.streaming = bool(getattr(args, "streaming_chunks", False))

    main(args)
