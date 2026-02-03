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
    Useful for occasional repeats like "oppose the oppose the".
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


# ============================================================
# Timestamp-based stitcher (strict authority intervals)
# ============================================================

class BoundaryCommitStitcher:
    """
    Timestamp stitching with an explicit boundary T between consecutive chunks.

    For each boundary between chunk i and i+1:
      - commit words from chunk i with end <= T
      - commit words from chunk i+1 with start >= T
      - resolve words that straddle T (start < T < end) using best candidate

    Requires forced aligner word timestamps (start_time/end_time).
    """

    def __init__(self, stride_s: float, boundary_mode: str = "half", collide_window_s: float = 0.18):
        """
        boundary_mode:
          - "left":  T = t1 - stride
          - "half":  T = t1 - stride/2   (recommended)
        collide_window_s:
          - for resolving straddlers across chunks by time proximity
        """
        self.stride_s = float(stride_s)
        self.boundary_mode = boundary_mode
        self.collide_window_s = float(collide_window_s)
        self._out_words: List[str] = []

    @staticmethod
    def _to_global(meta: Dict, items: List[Tuple[float, float, str]]) -> List[Tuple[float, float, float, str]]:
        """Return list of (center, start, end, word) in GLOBAL time."""
        t0 = float(meta["start_t"])
        out = []
        for s_local, e_local, w in items:
            s = t0 + float(s_local)
            e = t0 + float(e_local)
            c = 0.5 * (s + e)
            out.append((c, s, e, w))
        out.sort(key=lambda x: x[1])
        return out

    def stitch(self, metas_and_results: List[Tuple[Dict, Any]]) -> str:
        # Convert each chunk to global-word list
        chunks: List[Tuple[Dict, List[Tuple[float, float, float, str]]]] = []
        for meta, r in metas_and_results:
            items = _extract_forced_align_items(r)  # (s_local, e_local, word)
            words = self._to_global(meta, items)
            chunks.append((meta, words))

        if not chunks:
            return ""

        # If only 1 chunk: keep all its words
        if len(chunks) == 1:
            self._out_words = [w for _, _, _, w in chunks[0][1]]
            return " ".join(self._out_words).strip()

        out: List[str] = []

        # Process boundaries between consecutive chunks
        for i in range(len(chunks) - 1):
            meta_i, wi = chunks[i]
            meta_j, wj = chunks[i + 1]

            t0_i = float(meta_i["start_t"])
            len_i = float(meta_i["len_s"])
            t1_i = t0_i + len_i

            # boundary time inside overlap
            if self.boundary_mode == "left":
                T = t1_i - self.stride_s
            else:
                T = t1_i - 0.5 * self.stride_s  # "half" default

            # Safe-left: end <= T
            safe_left = [(c, s, e, w) for (c, s, e, w) in wi if e <= T]
            # Candidates around boundary from both sides: straddlers + near-boundary words
            amb_left = [(c, s, e, w) for (c, s, e, w) in wi if (s < T < e) or abs(c - T) <= self.collide_window_s]
            amb_right = [(c, s, e, w) for (c, s, e, w) in wj if (s < T < e) or abs(c - T) <= self.collide_window_s]

            # Emit safe-left now (but avoid re-emitting words already emitted from previous step)
            if i == 0:
                out.extend([w for _, _, _, w in safe_left])
            else:
                # for i>0, safe-left may include words already emitted from previous boundary,
                # so only emit words whose center is after the previous boundary T_prev.
                # simplest: just emit all safe_left; later dedup will handle
                out.extend([w for _, _, _, w in safe_left])

            # Resolve ambiguous region by merging amb candidates by time and deduping collisions,
            # preferring right chunk (more right-context).
            merged = amb_left + amb_right
            merged.sort(key=lambda x: x[0])  # by center

            resolved: List[Tuple[float, float, float, str]] = []
            for cand in merged:
                c, s, e, w = cand
                if not resolved:
                    resolved.append(cand)
                    continue
                c_prev, s_prev, e_prev, w_prev = resolved[-1]
                if abs(c - c_prev) <= self.collide_window_s:
                    # prefer right-chunk candidate: heuristic = if its start is >= T OR it comes from amb_right timing
                    # We can approximate: prefer the one whose center is >= T
                    if c >= T:
                        resolved[-1] = cand
                    continue
                resolved.append(cand)

            # Emit resolved words whose center is near boundary (to fill the gap)
            out.extend([w for (c, _, _, w) in resolved if abs(c - T) <= self.collide_window_s])

            # For the last boundary, also emit safe-right tail from final chunk after T
            if i == len(chunks) - 2:
                safe_right = [(c, s, e, w) for (c, s, e, w) in wj if s >= T]
                out.extend([w for _, _, _, w in safe_right])

        # Final cleanup: remove immediate duplicates
        cleaned: List[str] = []
        for w in out:
            if cleaned and _canon_word(cleaned[-1]) == _canon_word(w):
                continue
            cleaned.append(w)

        return " ".join(cleaned).strip()


class TimestampAuthorityStitcher:
    """
    Stitching that strictly uses word start/end timestamps (from ForcedAligner).

    Each chunk is given a global "authority interval". A word is committed iff
    its center time falls in that interval.

    Modes:
      - left: chunk owns everything except its full overlapped tail
              auth = [t0, t1 - stride_s] for non-final chunks
      - half: chunk owns everything except half the overlap
              auth = [t0, t1 - stride_s/2] for non-final chunks
      - final chunk always owns [t0, t1]
    """

    def __init__(
        self,
        stride_s: float,
        mode: str = "left",
        max_overlap_k: int = 12,
        time_slack_s: float = 0.08,
    ):
        self.stride_s = float(stride_s)
        self.mode = mode
        self.max_overlap_k = int(max_overlap_k)
        self.time_slack_s = float(time_slack_s)

        self._words: List[str] = []
        self._last_end_t: float = -1e9

    def _authority_bounds(self, t0: float, chunk_len_s: float, is_final: bool) -> Tuple[float, float]:
        t1 = t0 + chunk_len_s
        if is_final or self.stride_s <= 0:
            return t0, t1
        if self.mode == "half":
            return t0, t1 - 0.5 * self.stride_s
        return t0, t1 - self.stride_s

    def push_chunk(self, r: Any, chunk_start_t: float, chunk_len_s: float, is_final: bool) -> None:
        items = _extract_forced_align_items(r)

        if not items:
            # Fallback: text-only overlap (avoid dropping everything)
            txt = getattr(r, "text", "") or ""
            ws = _split_words(txt)
            ws = _drop_prefix_overlap(self._words, ws, max_k=self.max_overlap_k)
            if ws:
                self._words.extend(ws)
            return

        auth_lo, auth_hi = self._authority_bounds(chunk_start_t, chunk_len_s, is_final)

        cand: List[Tuple[float, float, str]] = []
        for s_local, e_local, w in items:
            s = chunk_start_t + s_local
            e = chunk_start_t + e_local
            c = 0.5 * (s + e)

            if c < auth_lo or c >= auth_hi:
                continue

            if s < self._last_end_t - self.time_slack_s:
                continue

            cand.append((s, e, w))

        if not cand:
            return

        cand.sort(key=lambda x: (x[0], x[1]))
        new_words = [w for _, _, w in cand]

        # Extra dedup for within-chunk repeats
        new_words = _drop_prefix_overlap(self._words, new_words, max_k=self.max_overlap_k)

        if new_words:
            self._words.extend(new_words)
            self._last_end_t = max(self._last_end_t, cand[-1][1])

    def final_text(self) -> str:
        return " ".join(self._words).strip()


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
        # IMPORTANT: request timestamps. Your CLI does NOT expose a flag for this.
        # If you pass a forced aligner, this yields ForcedAlignItems with start/end times.
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
                stitch = TimestampAuthorityStitcher(
                    stride_s=args.stride_s,
                    mode=args.stitch_mode,
                    max_overlap_k=args.max_overlap_k,
                    time_slack_s=args.time_slack_s,
                )
                for meta, r in by_sample[i]:
                    stitch.push_chunk(
                        r,
                        chunk_start_t=float(meta["start_t"]),
                        chunk_len_s=float(meta["len_s"]),
                        is_final=bool(meta["is_last"]),
                    )
                pred_texts[i] = stitch.final_text()
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
            for j in range(min(4, n_samples)):
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

    # forced aligner (must be enabled for word times)
    parser.add_argument("--forced_aligner", type=str, default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--forced_aligner_device_map", type=str, default="cuda:0")
    parser.add_argument("--forced_aligner_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    # timestamp-based stitching controls
    parser.add_argument("--stitch_mode", type=str, default="left", choices=["left", "half"])
    parser.add_argument("--max_overlap_k", type=int, default=12)
    parser.add_argument("--time_slack_s", type=float, default=0.08)

    args = parser.parse_args()

    # IMPORTANT FIX: data_utils.load_data(args) expects args.streaming to exist.
    # Make it equivalent to "chunked streaming is enabled".
    args.streaming = bool(getattr(args, "streaming_chunks", False))

    main(args)
