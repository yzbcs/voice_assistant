"""ASR 性能单测脚本 — 分阶段计时 + WER。

计时阶段:
  load    读取音频文件 + 重采样到 16kHz
  preprocess  音频标准化 + 编码为 WAV bytes（仅 api 模式）
  infer   API 调用 / 模型推理（纯推理耗时，不含预处理）
  total   端到端总耗时（load + preprocess + infer）

用法:
    python3 scripts/bench_asr.py --mode api --audio assets/input/test.wav
    python3 scripts/bench_asr.py --mode local --audio assets/input/test.wav
    python3 scripts/bench_asr.py --mode api --audio assets/input/
    python3 scripts/bench_asr.py --mode api --audio test.wav --gt "你好世界"
    python3 scripts/bench_asr.py --mode api --audio test.wav --rounds 5
"""

from __future__ import annotations

import argparse
import os
import time
import base64

import numpy as np

# 项目根目录加入 sys.path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from step2_asr_module import ASRModule


def load_audio(path: str) -> np.ndarray | None:
    """读取音频文件为 int16 numpy 数组（16kHz 单声道）。"""
    import scipy.io.wavfile as wavfile

    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}")
        return None
    try:
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        if sr != 16000:
            import scipy.signal
            data = scipy.signal.resample_poly(data, 16000, sr).astype(np.int16)
        return data
    except Exception as e:
        print(f"[ERROR] 读取失败: {e}")
        return None


def calc_wer(hypothesis: str, reference: str) -> float:
    """计算词错误率 (Word Error Rate)。

    对中文按字分割，对英文按空格分割。
    使用最小编辑距离算法（Levenshtein distance）。
    """
    # 中文按字分割
    if any('一' <= c <= '鿿' for c in reference):
        hyp_chars = list(hypothesis)
        ref_chars = list(reference)
    else:
        hyp_chars = hypothesis.split()
        ref_chars = reference.split()

    n = len(ref_chars)
    m = len(hyp_chars)

    if n == 0:
        return 0.0 if m == 0 else 1.0

    # DP 编辑距离
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / n


def collect_audio_files(path: str) -> list[str]:
    """收集目录下所有音频文件，或返回单个文件。"""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        print(f"[ERROR] 路径不存在: {path}")
        return []
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not files:
        print(f"[WARN] 目录下无音频文件: {path}")
    return files


def run_benchmark(
    audio_files: list[str],
    mode: str,
    rounds: int,
    gt_map: dict[str, str] | None,
):
    """执行 ASR 性能基准测试，分阶段计时。

    阶段:
      load        读取音频文件 + 重采样到 16kHz
      preprocess  音频标准化 + 编码为 WAV bytes（仅 api 模式）
      infer       API 调用 / 模型推理
      total       端到端总耗时
    """
    streaming = mode == "api"
    print(f"[初始化] ASR 模式: {'vLLM API' if streaming else 'Transformers'}")

    asr = ASRModule(streaming=streaming)
    if streaming and asr.client is None:
        print("[ERROR] vLLM API 客户端初始化失败，请检查 ASR 服务是否已启动")
        return
    if not streaming and asr.model is None:
        print("[ERROR] 模型加载失败，请检查 config.py 路径")
        return

    print(f"[测试] {len(audio_files)} 个文件 × {rounds} 轮\n")

    all_totals = []
    results = []

    for filepath in audio_files:
        fname = os.path.basename(filepath)
        print(f"--- {fname} ---")

        texts = []
        timings = []  # 每轮: {load, preprocess, infer, total}

        for i in range(rounds):
            # ---- Stage 1: 加载音频 ----
            t0 = time.perf_counter()
            audio_data = load_audio(filepath)
            t_load = time.perf_counter() - t0

            if audio_data is None:
                break

            # ---- Stage 2: 预处理（仅 api 模式） ----
            t_preprocess = 0.0
            b64_audio = None
            if streaming:
                t0 = time.perf_counter()
                normalized = asr._normalize_audio(audio_data, 16000)
                wav_bytes = asr._encode_wav(normalized, config.STREAM_SAMPLE_RATE)
                b64_audio = base64.b64encode(wav_bytes).decode("utf-8")
                t_preprocess = time.perf_counter() - t0

            # ---- Stage 3: 推理 ----
            t0 = time.perf_counter()
            if streaming:
                text = None

                # 方法 1: Transcriptions API（官方格式: raw bytes）
                try:
                    with open(filepath, "rb") as raw_f:
                        transcription = asr.client.audio.transcriptions.create(
                            model=asr.model_name,
                            file=raw_f,
                        )
                    text = transcription.text.strip()
                except Exception as e1:
                    if i == 0:
                        print(f"  [Transcriptions API 失败] {type(e1).__name__}: {e1}")
                        if hasattr(e1, "response") and hasattr(e1.response, "text"):
                            print(f"    响应体: {e1.response.text[:500]}")

                # 方法 2: Chat Completions（base64 audio_url）
                if text is None:
                    try:
                        response = asr.client.chat.completions.create(
                            model=asr.model_name,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "audio_url", "audio_url": {
                                        "url": f"data:audio/wav;base64,{b64_audio}"
                                    }},
                                ]
                            }],
                        )
                        text = response.choices[0].message.content.strip()
                    except Exception as e2:
                        if i == 0:
                            print(f"  [Chat Completions 失败] {type(e2).__name__}: {e2}")
                            if hasattr(e2, "response") and hasattr(e2.response, "text"):
                                print(f"    响应体: {e2.response.text[:500]}")

                if text is None:
                    text = ""
                    if i == 0:
                        print("  [WARN] 两种 API 均失败，请检查:")
                        print("    1. vLLM 是否安装了音频依赖: pip install 'vllm[audio]'")
                        print("    2. 服务端日志是否有报错")
                        print(f"    3. curl 测试: curl http://localhost:{config.ASR_VLLM_PORT}/v1/models")
            else:
                text = asr.transcribe(filepath)
            t_infer = time.perf_counter() - t0

            t_total = t_load + t_preprocess + t_infer
            timing = {
                "load": t_load,
                "preprocess": t_preprocess,
                "infer": t_infer,
                "total": t_total,
            }
            timings.append(timing)
            texts.append(text)

            if i == 0:
                print(f"  识别结果: {text}")
            parts = [f"load={t_load:.3f}s"]
            if streaming:
                parts.append(f"preprocess={t_preprocess:.3f}s")
            parts.append(f"infer={t_infer:.3f}s")
            parts.append(f"total={t_total:.3f}s")
            print(f"  第 {i + 1}/{rounds} 轮: {' | '.join(parts)}")

        if not timings:
            continue

        # 汇总统计
        avg = lambda key: np.mean([t[key] for t in timings])
        duration_s = len(audio_data) / 16000
        rtf = avg("infer") / duration_s if duration_s > 0 else float('inf')

        entry = {
            "file": fname,
            "text": texts[0],
            "duration_s": duration_s,
            "rtf": rtf,
        }
        for key in ("load", "preprocess", "infer", "total"):
            entry[f"avg_{key}"] = avg(key)

        # WER
        gt = (gt_map or {}).get(filepath) or (gt_map or {}).get(fname)
        if gt:
            wer = calc_wer(texts[0], gt)
            entry["wer"] = wer
            entry["gt"] = gt
            print(f"  WER: {wer:.2%}  (参考: {gt})")

        # 多轮一致性
        if rounds > 1:
            same = sum(1 for t in texts if t == texts[0])
            consistency = same / rounds
            entry["consistency"] = consistency
            print(f"  多轮一致性: {consistency:.0%} ({same}/{rounds})")

        print(f"  平均: load={avg('load'):.3f}s | preprocess={avg('preprocess'):.3f}s "
              f"| infer={avg('infer'):.3f}s | total={avg('total'):.3f}s | RTF={rtf:.3f}\n")

        all_totals.extend([t["total"] for t in timings])
        results.append(entry)

    # 汇总
    if not results:
        return

    print("=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"  文件数: {len(results)}")
    print(f"  总测试轮数: {len(all_totals)}")
    print(f"  总平均端到端: {np.mean(all_totals):.3f}s")
    print(f"  总中位端到端: {np.median(all_totals):.3f}s")

    # 表格
    hdr = f"{'文件':<25} {'音频(s)':<8} {'加载':<8} {'预处理':<8} {'推理':<8} {'总计':<8} {'RTF':<7} {'WER':<6}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in results:
        wer_str = f"{r['wer']:.2%}" if "wer" in r else "-"
        print(f"{r['file']:<25} {r['duration_s']:<8.1f} "
              f"{r['avg_load']:<8.3f} {r['avg_preprocess']:<8.3f} "
              f"{r['avg_infer']:<8.3f} {r['avg_total']:<8.3f} "
              f"{r['rtf']:<7.3f} {wer_str:<6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASR 性能单测")
    parser.add_argument("--mode", choices=["api", "local"], default="api",
                        help="测试模式: api (vLLM API) / local (transformers)")
    parser.add_argument("--audio", required=True,
                        help="音频文件路径或包含音频文件的目录")
    parser.add_argument("--gt", default=None,
                        help="ground truth 文本（用于计算 WER，单个文件时使用）")
    parser.add_argument("--rounds", type=int, default=1,
                        help="每个文件测试轮数（默认 1）")
    args = parser.parse_args()

    files = collect_audio_files(args.audio)
    if not files:
        sys.exit(1)

    gt_map = {}
    if args.gt and len(files) == 1:
        gt_map[files[0]] = args.gt

    run_benchmark(files, mode=args.mode, rounds=args.rounds, gt_map=gt_map or None)
