"""
Official NeMo parakeet-tdt-0.6b-v3 streaming evaluation for the Open ASR Leaderboard.

Uses NeMo's speech_to_text_streaming_infer_rnnt.py logic directly (preprocessor + encoder + TDT decoder).
Requires: conda activate (with NeMo installed)

Usage:
  python run_eval_nemo.py \
    --dataset_path hf-audio/esb-datasets-test-only-sorted \
    --dataset librispeech --split test.clean
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import evaluate
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from normalizer import data_utils

wer_metric = evaluate.load("wer")
SAMPLE_RATE = 16000


class NeMoStreamingASR:
    """
    NeMo parakeet-tdt-0.6b-v3 streaming ASR using the official algorithm
    from speech_to_text_streaming_infer_rnnt.py.
    """

    def __init__(self, pretrained_name="nvidia/parakeet-tdt-0.6b-v3",
                 left_context_secs=9.0, chunk_secs=0.8, right_context_secs=1.6):
        from nemo.collections.asr.models import EncDecRNNTModel
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
        from nemo.collections.asr.parts.utils.streaming_utils import ContextSize, StreamingBatchedAudioBuffer

        self.ContextSize = ContextSize
        self.StreamingBatchedAudioBuffer = StreamingBatchedAudioBuffer

        torch.set_grad_enabled(False)

        print(f"Loading {pretrained_name}...")
        self.model = EncDecRNNTModel.from_pretrained(pretrained_name, map_location='cpu')
        self.model.eval()
        self.model.freeze()
        self.model.preprocessor.featurizer.dither = 0.0
        self.model.preprocessor.featurizer.pad_to = 0

        model_cfg = copy.deepcopy(self.model._cfg)
        self.esf = self.model.encoder.subsampling_factor
        fss = model_cfg.preprocessor['window_stride']
        self.fps = 1.0 / fss
        self.ff2a = (int(SAMPLE_RATE * fss) // self.esf) * self.esf
        self.ef2a = self.ff2a * self.esf

        # Setup TDT decoding
        decoding_cfg = OmegaConf.structured(RNNTDecodingConfig())
        decoding_cfg.strategy = "greedy_batch"
        decoding_cfg.greedy.loop_labels = True
        decoding_cfg.greedy.preserve_alignments = False
        decoding_cfg.greedy.use_cuda_graph_decoder = False
        decoding_cfg.fused_batch_size = -1
        self.model.change_decoding_strategy(decoding_cfg)
        self.dc = self.model.decoding.decoding.decoding_computer

        # Context sizes
        cef = ContextSize(
            left=int(left_context_secs * self.fps / self.esf),
            chunk=int(chunk_secs * self.fps / self.esf),
            right=int(right_context_secs * self.fps / self.esf),
        )
        self.context_samples = ContextSize(
            left=cef.left * self.esf * self.ff2a,
            chunk=cef.chunk * self.esf * self.ff2a,
            right=cef.right * self.esf * self.ff2a,
        )
        print(f"  Context: L={self.context_samples.left/SAMPLE_RATE:.1f}s "
              f"C={self.context_samples.chunk/SAMPLE_RATE:.1f}s "
              f"R={self.context_samples.right/SAMPLE_RATE:.1f}s")

    def transcribe_streaming(self, audio_np):
        """Streaming chunked inference using NeMo's official algorithm."""
        from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
        audio_len = torch.tensor([len(audio_np)], dtype=torch.long)
        cs = self.context_samples

        cbh = None; state = None
        ls = 0; rs = min(cs.chunk + cs.right, audio_tensor.shape[1])
        buf = self.StreamingBatchedAudioBuffer(
            batch_size=1, context_samples=cs,
            dtype=audio_tensor.dtype, device=audio_tensor.device)
        ral = audio_len.clone()

        while ls < audio_tensor.shape[1]:
            cl = min(rs, audio_tensor.shape[1]) - ls
            ilcb = cl >= ral; ilc = rs >= audio_tensor.shape[1]
            clb = torch.where(ilcb, ral, torch.full_like(ral, fill_value=cl))
            buf.add_audio_batch_(
                audio_tensor[:, ls:rs], audio_lengths=clb,
                is_last_chunk=ilc, is_last_chunk_batch=ilcb)

            eo, eol = self.model(
                input_signal=buf.samples,
                input_signal_length=buf.context_size_batch.total())
            eo = eo.transpose(1, 2)
            ec = buf.context_size.subsample(factor=self.ef2a)
            ecb = buf.context_size_batch.subsample(factor=self.ef2a)
            eo = eo[:, ec.left:]

            ch, _, state = self.dc(
                x=eo,
                out_len=torch.where(ilcb, eol - ecb.left, ecb.chunk),
                prev_batched_state=state)
            if cbh is None: cbh = ch
            else: cbh.merge_(ch)

            ral -= clb; ls = rs
            rs = min(rs + cs.chunk, audio_tensor.shape[1])

        hyps = batched_hyps_to_hypotheses(cbh, None, batch_size=1)
        return self.model.tokenizer.ids_to_text(hyps[0].y_sequence.tolist())

    def transcribe_batch(self, audio_np):
        """Full-sequence (offline) inference using NeMo model forward."""
        from nemo.collections.asr.parts.utils.rnnt_utils import batched_hyps_to_hypotheses

        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
        audio_len = torch.tensor([len(audio_np)], dtype=torch.long)

        eo, eol = self.model(input_signal=audio_tensor, input_signal_length=audio_len)
        eo = eo.transpose(1, 2)

        dec_out, h, c = None, None, None
        state = None
        ch, _, _ = self.dc(x=eo, out_len=eol, prev_batched_state=None)
        hyps = batched_hyps_to_hypotheses(ch, None, batch_size=1)
        return self.model.tokenizer.ids_to_text(hyps[0].y_sequence.tolist())


def main(args):
    asr = NeMoStreamingASR(
        pretrained_name=args.pretrained_name,
        left_context_secs=args.left_context_secs,
        chunk_secs=args.chunk_secs,
        right_context_secs=args.right_context_secs,
    )

    dataset = data_utils.load_data(args)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling to first {args.max_eval_samples} samples")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    dataset = data_utils.prepare_data(dataset)

    all_stream_preds = []
    all_batch_preds = []
    all_refs_so_far = []
    sample_counter = [0]

    def benchmark(batch):
        audios = batch["audio"]
        stream_predictions = []
        batch_predictions = []
        transcription_times = []

        for a in audios:
            audio_array = np.asarray(a["array"], dtype=np.float32)
            sr = a["sampling_rate"]
            if sr != SAMPLE_RATE:
                raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz.")

            t0 = time.time()
            stream_text = asr.transcribe_streaming(audio_array)
            elapsed = time.time() - t0

            batch_text = asr.transcribe_batch(audio_array)

            stream_predictions.append(stream_text)
            batch_predictions.append(batch_text)
            transcription_times.append(elapsed)

        norm_stream = [data_utils.normalizer(p) for p in stream_predictions]
        norm_batch = [data_utils.normalizer(p) for p in batch_predictions]
        norm_refs = batch["norm_text"]

        for i, (s_raw, b_raw, s_norm, b_norm, ref_norm, a, t_time) in enumerate(
            zip(stream_predictions, batch_predictions,
                norm_stream, norm_batch, norm_refs, audios, transcription_times)):
            sample_counter[0] += 1
            audio_array = np.asarray(a["array"], dtype=np.float32)
            audio_dur = len(audio_array) / a["sampling_rate"]
            sample_rtfx = round(audio_dur / max(t_time, 1e-9), 2)

            if ref_norm.strip():
                s_wer = round(100 * wer_metric.compute(references=[ref_norm], predictions=[s_norm]), 1)
                b_wer = round(100 * wer_metric.compute(references=[ref_norm], predictions=[b_norm]), 1)
            else:
                s_wer = 0.0 if not s_norm.strip() else 100.0
                b_wer = 0.0 if not b_norm.strip() else 100.0

            all_stream_preds.append(s_norm)
            all_batch_preds.append(b_norm)
            all_refs_so_far.append(ref_norm)
            agg_s = round(100 * wer_metric.compute(references=all_refs_so_far, predictions=all_stream_preds), 2)
            agg_b = round(100 * wer_metric.compute(references=all_refs_so_far, predictions=all_batch_preds), 2)

            print(f"  [{sample_counter[0]:>4d}] [{audio_dur:.1f}s, sWER={s_wer:5.1f}%, bWER={b_wer:5.1f}%, "
                  f"aggSWER={agg_s:5.2f}%, aggBWER={agg_b:5.2f}%, RTFx={sample_rtfx:6.2f}]\n"
                  f"    STREAM: {s_raw}\n    BATCH:  {b_raw}\n    REF:    {ref_norm}")

        batch["predictions"] = norm_stream
        batch["batch_predictions"] = norm_batch
        batch["references"] = norm_refs
        batch["transcription_time_s"] = transcription_times
        return batch

    dataset = dataset.map(benchmark, batch_size=args.batch_size, batched=True, remove_columns=["audio"])

    all_results = {"audio_length_s": [], "transcription_time_s": [], "predictions": [], "batch_predictions": [], "references": []}
    for result in tqdm(dataset, desc="Samples"):
        for key in all_results:
            all_results[key].append(result[key])

    model_id = args.model_id or args.pretrained_name.replace("/", "_")
    manifest_path = data_utils.write_manifest(
        all_results["references"], all_results["predictions"], model_id,
        args.dataset_path, args.dataset, args.split,
        audio_length=all_results["audio_length_s"],
        transcription_time=all_results["transcription_time_s"])
    print("Results saved at:", os.path.abspath(manifest_path))

    s_wer = round(100 * wer_metric.compute(references=all_results["references"], predictions=all_results["predictions"]), 2)
    b_wer = round(100 * wer_metric.compute(references=all_results["references"], predictions=all_results["batch_predictions"]), 2)
    total_audio = sum(all_results["audio_length_s"])
    total_time = sum(all_results["transcription_time_s"])
    rtfx = round(total_audio / max(total_time, 1e-9), 2)

    print(f"\n{'='*70}")
    print(f"Streaming WER: {s_wer} %")
    print(f"Batch WER:     {b_wer} %")
    print(f"RTFx:          {rtfx}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeMo parakeet-tdt-0.6b-v3 Official Streaming Eval")
    parser.add_argument("--pretrained_name", type=str, default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--left_context_secs", type=float, default=10.0)
    parser.add_argument("--chunk_secs", type=float, default=2.0)
    parser.add_argument("--right_context_secs", type=float, default=2.0)
    parser.set_defaults(streaming=True)
    args = parser.parse_args()
    main(args)
