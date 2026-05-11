"""Step 1: 启动 vLLM 服务加载 Ministral-3-8B-Instruct-2512

用法:
    python3 step1_setup_vllm.py                  # 启动服务（阻塞）
    python3 step1_setup_vllm.py --check-only     # 仅检查服务是否在线

服务启动后，可通过 OpenAI 兼容接口访问:
    curl http://localhost:8000/v1/chat/completions
"""

import subprocess
import sys
import time
import os
import urllib.request
import urllib.error
import json
import argparse

import config


def _connect_host(host: str) -> str:
    """服务绑定 0.0.0.0 时，客户端健康检查应连接 localhost。"""
    return "localhost" if host == "0.0.0.0" else host


def check_vllm_health(host: str = config.VLLM_HOST, port: int = config.VLLM_PORT, timeout: int = 5) -> bool:
    """检查 vLLM 服务是否就绪"""
    url = f"http://{_connect_host(host)}:{port}/v1/models"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"[vLLM] 服务在线，可用模型: {models}")
            return True
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
        return False


def wait_for_vllm(host: str = config.VLLM_HOST, port: int = config.VLLM_PORT,
                  max_wait: int = 300, proc: subprocess.Popen | None = None,
                  log_path: str | None = None) -> bool:
    """等待 vLLM 服务启动就绪，最多等 max_wait 秒"""
    print(f"[vLLM] 等待服务启动 (最多 {max_wait}s) ...")
    start = time.time()
    while time.time() - start < max_wait:
        if proc is not None and proc.poll() is not None:
            print(f"[vLLM] 服务进程已退出，exit code: {proc.returncode}")
            if log_path and os.path.exists(log_path):
                print(f"[vLLM] 日志尾部 ({log_path}):")
                with open(log_path, "r", errors="replace") as f:
                    lines = f.readlines()[-80:]
                print("".join(lines).rstrip())
            return False
        if check_vllm_health(host, port):
            print(f"[vLLM] 服务就绪！耗时 {time.time() - start:.1f}s")
            return True
        time.sleep(5)
    print(f"[vLLM] 等待超时 ({max_wait}s)，服务未就绪")
    return False


def start_vllm_server():
    """启动 vLLM 服务进程（阻塞）"""
    model_path = config.LLM_MODEL_PATH
    served_model_name = os.path.basename(model_path.rstrip(os.sep))
    if "/path/to/" in model_path:
        print(f"[ERROR] 请先在 config.py 中设置 LLM_MODEL_PATH 为实际的模型路径")
        print(f"  当前值: {model_path}")
        sys.exit(1)

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", served_model_name,
        "--tokenizer_mode", "mistral",
        "--config_format", "mistral",
        "--load_format", "mistral",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "mistral",
        "--host", config.VLLM_HOST,
        "--port", str(config.VLLM_PORT),
        "--gpu-memory-utilization", str(config.VLLM_GPU_MEMORY_UTILIZATION),
        "--max-model-len", str(config.VLLM_MAX_MODEL_LEN),
    ]

    print(f"[vLLM] 启动命令: {' '.join(cmd)}")
    print(f"[vLLM] 模型路径: {model_path}")
    print(f"[vLLM] API 模型名: {served_model_name}")
    print(f"[vLLM] 服务地址: http://{config.VLLM_HOST}:{config.VLLM_PORT}")

    # 启动 vLLM 作为子进程，日志输出到文件
    log_path = "assets/logs/vllm_server.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    print(f"[vLLM] 进程 PID: {proc.pid}，日志: {log_path}")
    print(f"[vLLM] 按 Ctrl+C 停止服务")

    try:
        # 等待服务就绪
        if wait_for_vllm(proc=proc, log_path=log_path):
            print("[vLLM] 服务已启动，可以开始使用")
            print_asr_start_command()
        proc.wait()
    except KeyboardInterrupt:
        print("\n[vLLM] 收到中断信号，停止服务...")
        proc.terminate()
        proc.wait(timeout=10)
        log_file.close()
        print("[vLLM] 服务已停止")


def print_asr_start_command():
    """打印 ASR vLLM 服务的启动命令（需在另一终端运行）"""
    model_path = config.ASR_MODEL_PATH
    if "/path/to/" in model_path:
        print("[ASR] 请先在 config.py 中设置 ASR_MODEL_PATH")
        return
    cmd = (
        f"vllm serve {model_path}"
        f" --host {config.ASR_VLLM_HOST}"
        f" --port {config.ASR_VLLM_PORT}"
        f" --gpu-memory-utilization {config.ASR_VLLM_GPU_MEMORY_UTILIZATION}"
        f" --dtype auto"
    )
    print(f"\n[ASR] 如需语音功能，请在另一终端运行:")
    print(f"  {cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动/检查 vLLM 服务")
    parser.add_argument("--check-only", action="store_true", help="仅检查服务是否在线")
    args = parser.parse_args()

    if args.check_only:
        ok = check_vllm_health()
        sys.exit(0 if ok else 1)
    else:
        start_vllm_server()
