"""Step 4: 语音助手主入口 — 录音 → ASR → Agent → 输出

支持三种交互模式:
  1. 语音模式: 录音 → ASR 转文字 → Agent 回复
  2. 文本模式: 直接键入文字 → Agent 回复
  3. 流式模式: 录音/上传文件 → vLLM 流式 ASR → Agent 回复

用法:
    python3 step4_main.py                 # 交互式选择模式
    python3 step4_main.py --mode text     # 直接进入文本模式
    python3 step4_main.py --mode voice    # 直接进入语音模式
    python3 step4_main.py --mode stream   # 直接进入流式 ASR 模式

语音模式依赖:
    - vLLM 服务 (step1_setup_vllm.py)
    - ASR 模型 (step2_asr_module.py)
    - sounddevice (麦克风录音)
"""

import argparse
import os
import time
import wave

import numpy as np

import config
from step2_asr_module import ASRModule
from step3_agent_core import VoiceAssistant


# ============================================================
# 录音函数
# ============================================================


def record_audio(duration: int = 0, sample_rate: int = config.RECORD_SAMPLE_RATE,
                 channels: int = config.RECORD_CHANNELS) -> str:
    """录制麦克风音频并保存为 WAV 文件

    参数:
        duration: 录音时长（秒），0 表示手动停止（按回车）
        sample_rate: 采样率
        channels: 声道数

    返回:
        保存的 WAV 文件路径
    """
    import sounddevice as sd

    if duration > 0:
        # 固定时长录音
        print(f"[录音] 录制 {duration} 秒...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype="int16")
        sd.wait()
    else:
        # 手动停止：按回车开始，再按回车停止
        print("[录音] 按回车开始录音，再按回车停止录音...")
        input("按回车开始...")
        print("[录音] 录音中... 按回车停止")
        recording_chunks = []

        def callback(indata, frames, time_info, status):
            recording_chunks.append(indata.copy())

        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="int16", callback=callback):
            input()  # 等待用户按回车停止

        if not recording_chunks:
            print("[录音] 未录制到任何音频")
            return ""
        audio_data = np.concatenate(recording_chunks, axis=0)

    # 保存为 WAV
    output_dir = "assets/input"
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, f"recording_{int(time.time())}.wav")

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"[录音] 已保存: {wav_path}")
    return wav_path


def record_audio_array(duration: int = 0, sample_rate: int = config.RECORD_SAMPLE_RATE,
                       channels: int = config.RECORD_CHANNELS) -> np.ndarray | None:
    """录制麦克风音频，返回 numpy 数组（不保存文件）

    参数:
        duration: 录音时长（秒），0 表示手动停止（按回车）
        sample_rate: 采样率
        channels: 声道数

    返回:
        int16 numpy 数组（单声道），失败返回 None
    """
    import sounddevice as sd

    if duration > 0:
        print(f"[录音] 录制 {duration} 秒...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype="int16")
        sd.wait()
    else:
        print("[录音] 按回车开始录音，再按回车停止录音...")
        input("按回车开始...")
        print("[录音] 录音中... 按回车停止")
        recording_chunks = []

        def callback(indata, frames, time_info, status):
            recording_chunks.append(indata.copy())

        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="int16", callback=callback):
            input()

        if not recording_chunks:
            print("[录音] 未录制到任何音频")
            return None
        audio_data = np.concatenate(recording_chunks, axis=0)

    # 转单声道
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    print(f"[录音] 录制完成，长度: {len(audio_data) / sample_rate:.1f}s")
    return audio_data


# ============================================================
# 音频文件读取
# ============================================================

def load_audio_file(path: str) -> np.ndarray | None:
    """读取音频文件为 numpy 数组（16kHz, 单声道, int16）

    参数:
        path: 音频文件路径（支持 wav 格式）

    返回:
        int16 numpy 数组，失败返回 None
    """
    import scipy.io.wavfile as wavfile

    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}")
        return None
    try:
        sr, data = wavfile.read(path)
        # 转单声道
        if data.ndim > 1:
            data = data[:, 0]
        # 重采样到 16kHz
        if sr != config.STREAM_SAMPLE_RATE:
            import scipy.signal
            data = scipy.signal.resample_poly(data, config.STREAM_SAMPLE_RATE, sr).astype(np.int16)
        print(f"[音频] 已加载: {path}, 采样率: {sr}Hz, 长度: {len(data) / config.STREAM_SAMPLE_RATE:.1f}s")
        return data
    except Exception as e:
        print(f"[ERROR] 读取音频失败: {e}")
        return None


def run_voice_mode(assistant: VoiceAssistant, asr: ASRModule):
    """语音交互模式"""
    print("\n" + "="*50)
    print("🎤 语音助手 — 语音模式")
    print("操作: 输入 'r' 录音 | 'q' 退出 | 其他文字直接对话")
    print("="*50 + "\n")

    while True:
        user_input = input("[You] ").strip()
        if not user_input:
            continue
        if user_input.lower() == "q":
            print("[助手] 再见！")
            break

        if user_input.lower() == "r":
            # 录音模式
            wav_path = record_audio(duration=config.RECORD_DURATION if config.RECORD_DURATION > 0 else 0)
            if not wav_path:
                continue

            # ASR 识别
            print("[ASR] 识别中...")
            text = asr.transcribe(wav_path)
            print(f"[ASR] 识别结果: {text}")

            if not text or text.startswith("[ERROR]"):
                print(f"[助手] 语音识别失败: {text}")
                continue

            user_input = text

        # Agent 回复
        try:
            reply = assistant.chat(user_input)
            print(f"[助手] {reply}\n")
        except Exception as e:
            print(f"[ERROR] Agent 调用失败: {e}\n")


def run_text_mode(assistant: VoiceAssistant):
    """文本交互模式"""
    print("\n" + "="*50)
    print("💬 语音助手 — 文本模式")
    print("直接输入文字对话，输入 'q' 退出")
    print("="*50 + "\n")

    while True:
        user_input = input("[You] ").strip()
        if not user_input:
            continue
        if user_input.lower() == "q":
            print("[助手] 再见！")
            break

        try:
            reply = assistant.chat(user_input)
            print(f"[助手] {reply}\n")
        except Exception as e:
            print(f"[ERROR] Agent 调用失败: {e}\n")


def run_streaming_mode(assistant: VoiceAssistant, asr: ASRModule):
    """流式 ASR 模式：手动录音 / 上传音频文件 → 流式 ASR → Agent

    交互命令:
        回车       → 录音（再按回车停止）
        f <路径>   → 上传音频文件（如 f assets/input/test.wav）
        其他文字   → 直接对话
        q          → 退出
    """
    print("\n" + "=" * 50)
    print(" streaming ASR 模式")
    print("  回车       → 录音（再按回车停止）")
    print("  f <路径>   → 上传音频文件（如 f assets/input/test.wav）")
    print("  其他文字   → 直接对话")
    print("  q          → 退出")
    print("=" * 50 + "\n")

    while True:
        cmd = input("[You] ").strip()
        if not cmd:
            # 空回车 → 录音
            audio_data = record_audio_array(
                duration=config.RECORD_DURATION if config.RECORD_DURATION > 0 else 0
            )
            if audio_data is None:
                continue
        elif cmd.lower() == "q":
            print("[助手] 再见！")
            break
        elif cmd.startswith("f "):
            # 上传文件
            file_path = cmd[2:].strip()
            audio_data = load_audio_file(file_path)
            if audio_data is None:
                continue
        else:
            # 直接文字对话
            try:
                reply = assistant.chat(cmd)
                print(f"[助手] {reply}\n")
            except Exception as e:
                print(f"[ERROR] Agent 调用失败: {e}\n")
            continue

        # 流式 ASR 识别
        print("[ASR] 流式识别中...")
        text = asr.transcribe_audio_array(audio_data, sample_rate=config.STREAM_SAMPLE_RATE)
        print(f"[ASR] 识别结果: {text}")
        if not text or text.startswith("[ERROR]"):
            print(f"[助手] 识别失败: {text}")
            continue

        # Agent 回复
        try:
            reply = assistant.chat(text)
            print(f"[助手] {reply}\n")
        except Exception as e:
            print(f"[ERROR] Agent 调用失败: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="语音助手主入口")
    parser.add_argument("--mode", choices=["voice", "text", "stream"], default=None,
                        help="交互模式: voice(语音) / text(文本) / stream(流式ASR)")
    args = parser.parse_args()

    # 选择模式
    if args.mode:
        mode = args.mode
    else:
        print("请选择交互模式:")
        print("  1) 语音模式 (录音 → ASR → Agent)")
        print("  2) 文本模式 (直接键入文字 → Agent)")
        print("  3) 流式模式 (录音/上传文件 → vLLM流式ASR → Agent)")
        choice = input("请输入 1/2/3: ").strip()
        mode_map = {"1": "voice", "2": "text", "3": "stream"}
        mode = mode_map.get(choice, "text")

    # 初始化 Agent
    print("[初始化] 正在启动 LangChain Agent...")
    assistant = VoiceAssistant()
    print("[初始化] Agent 就绪")

    if mode == "stream":
        # 流式模式：vLLM 后端 ASR
        print("[初始化] 正在加载 ASR 模型 (vLLM streaming)...")
        asr = ASRModule(streaming=True)
        if asr.model is None:
            print("[ERROR] ASR 模型加载失败，请检查 config.ASR_MODEL_PATH")
            print("[提示] 切换到文本模式...")
            run_text_mode(assistant)
        else:
            print("[初始化] ASR 流式模型就绪")
            run_streaming_mode(assistant, asr)
    elif mode == "voice":
        # 语音模式：transformers 后端 ASR
        print("[初始化] 正在加载 ASR 模型...")
        asr = ASRModule()
        if asr.model is None:
            print("[ERROR] ASR 模型加载失败，请检查 config.ASR_MODEL_PATH")
            print("[提示] 切换到文本模式...")
            run_text_mode(assistant)
        else:
            print("[初始化] ASR 模型就绪")
            run_voice_mode(assistant, asr)
    else:
        run_text_mode(assistant)


if __name__ == "__main__":
    main()
