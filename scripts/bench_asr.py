"""ASR benchmark for whole-file input with streamed text output.

Default target:
  vLLM OpenAI-compatible server at http://localhost:8000/v1
  model name: mega

Measured stages:
  load        Read and decode the audio file.
  normalize   Convert to mono 16 kHz int16 PCM.
  encode      Encode as WAV bytes and base64 data URL.
  connect     Create the streaming API response object.
  ttft        Time to first non-empty streamed text delta.
  stream      Time from first text delta to stream completion.
  infer       Full API streaming time, including connect and generation.
  total       End-to-end time for load + normalize + encode + infer.

Usage:
  python3 scripts/bench_asr.py --audio assets/input/test.wav
  python3 scripts/bench_asr.py --audio assets/input/test.wav --host localhost --port 8000 --model mega
  python3 scripts/bench_asr.py --audio assets/input/ --rounds 3
  python3 scripts/bench_asr.py --audio test.wav --gt "你好世界"
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
import wave
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


TARGET_SAMPLE_RATE = 16000


@dataclass
class AudioPayload:
    data_url: str
    duration_s: float
    source_sample_rate: int
    num_samples: int


@dataclass
class StreamResult:
    text: str
    timings: Dict[str, float]
    chunks: int


def collect_audio_files(path: str) -> List[str]:
    """Collect audio files under a directory or return a single file."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        print(f"[ERROR] 路径不存在: {path}")
        return []

    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    files = sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.splitext(name)[1].lower() in exts
    )
    if not files:
        print(f"[WARN] 目录下无音频文件: {path}")
    return files


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Read an audio file as a numpy array and sample rate."""
    try:
        import soundfile as sf

        data, sample_rate = sf.read(path, always_2d=False)
        return np.asarray(data), int(sample_rate)
    except Exception as sf_error:
        ext = os.path.splitext(path)[1].lower()
        if ext != ".wav":
            raise RuntimeError(
                f"soundfile 读取失败，且非 WAV 文件无法回退 scipy: {sf_error}"
            ) from sf_error

        try:
            import scipy.io.wavfile as wavfile

            sample_rate, data = wavfile.read(path)
            return np.asarray(data), int(sample_rate)
        except Exception as wav_error:
            raise RuntimeError(f"读取音频失败: {wav_error}") from wav_error


def normalize_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert audio to mono 16 kHz int16 PCM."""
    if audio.size == 0:
        return np.array([], dtype=np.int16)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio)
    if sample_rate != TARGET_SAMPLE_RATE:
        import scipy.signal

        audio = scipy.signal.resample_poly(audio, TARGET_SAMPLE_RATE, sample_rate)

    if np.issubdtype(audio.dtype, np.floating):
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16)

    if audio.dtype != np.int16:
        info = np.iinfo(audio.dtype) if np.issubdtype(audio.dtype, np.integer) else None
        if info is not None and info.bits > 16:
            audio = (audio.astype(np.float32) / max(abs(info.min), info.max)) * 32767.0
        return np.clip(audio, -32768, 32767).astype(np.int16)

    return audio


def encode_wav_data_url(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
    """Encode int16 PCM as a WAV data URL accepted by vLLM audio_url input."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    b64_audio = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64_audio}"


def prepare_audio_payload(path: str) -> tuple[AudioPayload, Dict[str, float]]:
    """Read, normalize, and encode the complete input audio file."""
    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    audio, source_sample_rate = load_audio(path)
    timings["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    normalized = normalize_audio(audio, source_sample_rate)
    timings["normalize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    data_url = encode_wav_data_url(normalized)
    timings["encode"] = time.perf_counter() - t0

    payload = AudioPayload(
        data_url=data_url,
        duration_s=len(normalized) / TARGET_SAMPLE_RATE,
        source_sample_rate=source_sample_rate,
        num_samples=len(normalized),
    )
    return payload, timings


def extract_delta_text(chunk) -> str:
    """Extract streamed text from OpenAI-compatible chat completion chunks."""
    try:
        delta = chunk.choices[0].delta
    except Exception:
        return ""

    content = getattr(delta, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or getattr(item, "content", "")))
        return "".join(parts)
    return ""


def stream_transcribe(
    client,
    model: str,
    payload: AudioPayload,
    prompt: str,
    temperature: float,
    print_stream: bool,
) -> StreamResult:
    """Send one complete audio file and consume streamed text output."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": payload.data_url}},
            {"type": "text", "text": prompt},
        ],
    }]

    timings: Dict[str, float] = {}
    text_parts: List[str] = []
    chunks = 0
    first_text_at: Optional[float] = None

    infer_start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    timings["connect"] = time.perf_counter() - infer_start

    for chunk in response:
        delta_text = extract_delta_text(chunk)
        if not delta_text:
            continue
        now = time.perf_counter()
        if first_text_at is None:
            first_text_at = now
        chunks += 1
        text_parts.append(delta_text)
        if print_stream:
            print(delta_text, end="", flush=True)

    infer_end = time.perf_counter()
    if print_stream:
        print()

    timings["ttft"] = (first_text_at - infer_start) if first_text_at else float("nan")
    timings["stream"] = (infer_end - first_text_at) if first_text_at else 0.0
    timings["infer"] = infer_end - infer_start
    return StreamResult(text="".join(text_parts).strip(), timings=timings, chunks=chunks)


def calc_wer(hypothesis: str, reference: str) -> float:
    """Calculate WER; for Chinese references this is character error rate."""
    if any("\u4e00" <= c <= "\u9fff" for c in reference):
        hyp_units = list(hypothesis)
        ref_units = list(reference)
    else:
        hyp_units = hypothesis.split()
        ref_units = reference.split()

    n = len(ref_units)
    m = len(hyp_units)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_units[i - 1] == hyp_units[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / n


def print_timing(prefix: str, timings: Dict[str, float]) -> None:
    keys = ["load", "normalize", "encode", "connect", "ttft", "stream", "infer", "total"]
    parts = []
    for key in keys:
        value = timings.get(key)
        if value is None:
            continue
        if np.isnan(value):
            parts.append(f"{key}=N/A")
        else:
            parts.append(f"{key}={value:.3f}s")
    print(f"{prefix}{' | '.join(parts)}")


def avg(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def run_benchmark(args: argparse.Namespace) -> int:
    from openai import OpenAI

    audio_files = collect_audio_files(args.audio)
    if not audio_files:
        return 1

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    print(f"[初始化] vLLM API: {base_url}")
    print(f"[初始化] model: {args.model}")
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        print(f"[初始化] 服务在线，可用模型: {model_ids}")
        if args.model not in model_ids:
            print(f"[WARN] 指定模型名 '{args.model}' 不在 /v1/models 返回列表中")
    except Exception as exc:
        print(f"[ERROR] 无法连接 vLLM 服务: {type(exc).__name__}: {exc}")
        return 2

    gt_map: Dict[str, str] = {}
    if args.gt and len(audio_files) == 1:
        gt_map[audio_files[0]] = args.gt

    all_timings: List[Dict[str, float]] = []
    results: List[Dict[str, object]] = []
    print(f"[测试] {len(audio_files)} 个文件 x {args.rounds} 轮\n")

    for filepath in audio_files:
        fname = os.path.basename(filepath)
        print(f"--- {fname} ---")
        file_timings: List[Dict[str, float]] = []
        texts: List[str] = []
        duration_s = 0.0

        for round_idx in range(args.rounds):
            total_start = time.perf_counter()
            try:
                payload, timings = prepare_audio_payload(filepath)
                duration_s = payload.duration_s
            except Exception as exc:
                print(f"  [ERROR] 音频预处理失败: {type(exc).__name__}: {exc}")
                break

            print(f"  第 {round_idx + 1}/{args.rounds} 轮流式输出: ", end="", flush=True)
            try:
                stream_result = stream_transcribe(
                    client=client,
                    model=args.model,
                    payload=payload,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    print_stream=not args.no_print_stream,
                )
            except Exception as exc:
                print()
                print(f"  [ERROR] API 调用失败: {type(exc).__name__}: {exc}")
                break

            timings.update(stream_result.timings)
            timings["total"] = time.perf_counter() - total_start
            timings["rtf_infer"] = timings["infer"] / duration_s if duration_s > 0 else float("inf")
            timings["rtf_total"] = timings["total"] / duration_s if duration_s > 0 else float("inf")

            texts.append(stream_result.text)
            file_timings.append(timings)
            all_timings.append(timings)

            print_timing("  耗时: ", timings)
            print(
                f"  音频: {duration_s:.2f}s | chunks={stream_result.chunks} "
                f"| RTF(infer)={timings['rtf_infer']:.3f} | RTF(total)={timings['rtf_total']:.3f}"
            )

        if not file_timings:
            continue

        file_result: Dict[str, object] = {
            "file": fname,
            "duration_s": duration_s,
            "text": texts[0] if texts else "",
        }
        for key in ["load", "normalize", "encode", "connect", "ttft", "stream", "infer", "total"]:
            file_result[f"avg_{key}"] = avg(t[key] for t in file_timings if key in t and not np.isnan(t[key]))
        file_result["rtf_infer"] = float(file_result["avg_infer"]) / duration_s if duration_s > 0 else float("inf")
        file_result["rtf_total"] = float(file_result["avg_total"]) / duration_s if duration_s > 0 else float("inf")

        gt = gt_map.get(filepath) or gt_map.get(fname)
        if gt:
            wer = calc_wer(str(file_result["text"]), gt)
            file_result["wer"] = wer
            print(f"  WER/CER: {wer:.2%}  (参考: {gt})")

        if args.rounds > 1 and texts:
            same = sum(1 for text in texts if text == texts[0])
            consistency = same / len(texts)
            file_result["consistency"] = consistency
            print(f"  多轮一致性: {consistency:.0%} ({same}/{len(texts)})")

        print_timing("  平均: ", {key.replace("avg_", ""): value for key, value in file_result.items() if key.startswith("avg_")})
        print(
            f"  平均 RTF(infer)={file_result['rtf_infer']:.3f} "
            f"| 平均 RTF(total)={file_result['rtf_total']:.3f}\n"
        )
        results.append(file_result)

    if not results:
        return 3

    print("=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"  文件数: {len(results)}")
    print(f"  总测试轮数: {len(all_timings)}")
    for key in ["load", "normalize", "encode", "connect", "ttft", "stream", "infer", "total"]:
        values = [timing[key] for timing in all_timings if key in timing and not np.isnan(timing[key])]
        print(f"  平均 {key:<9}: {avg(values):.3f}s")

    hdr = (
        f"{'文件':<26} {'音频(s)':<8} {'load':<8} {'norm':<8} {'encode':<8} "
        f"{'ttft':<8} {'infer':<8} {'total':<8} {'RTF_i':<7} {'RTF_t':<7} {'WER':<7}"
    )
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for result in results:
        wer = result.get("wer")
        wer_str = f"{wer:.2%}" if isinstance(wer, float) else "-"
        print(
            f"{str(result['file']):<26} {float(result['duration_s']):<8.1f} "
            f"{float(result['avg_load']):<8.3f} {float(result['avg_normalize']):<8.3f} "
            f"{float(result['avg_encode']):<8.3f} {float(result['avg_ttft']):<8.3f} "
            f"{float(result['avg_infer']):<8.3f} {float(result['avg_total']):<8.3f} "
            f"{float(result['rtf_infer']):<7.3f} {float(result['rtf_total']):<7.3f} {wer_str:<7}"
        )

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM ASR whole-file input with streamed text output")
    parser.add_argument("--audio", required=True, help="音频文件路径或音频目录")
    parser.add_argument("--host", default="localhost", help="vLLM host，默认 localhost")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port，默认 8000")
    parser.add_argument("--model", default="mega", help="vLLM served model name，默认 mega")
    parser.add_argument("--api-key", default="EMPTY", help="OpenAI compatible API key，默认 EMPTY")
    parser.add_argument("--rounds", type=int, default=1, help="每个文件测试轮数")
    parser.add_argument("--gt", default=None, help="ground truth 文本，单文件时用于计算 WER/CER")
    parser.add_argument(
        "--prompt",
        default="请将这段音频完整转写为文本，只输出转写结果。",
        help="发送给 ASR 模型的文本提示",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度，默认 0")
    parser.add_argument("--no-print-stream", action="store_true", help="不在接收时逐块打印流式文本")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run_benchmark(parse_args()))
