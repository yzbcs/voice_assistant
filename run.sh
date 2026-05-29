#!/usr/bin/env bash
# 语音助手运行脚本
# 用法:
#   bash run.sh setup     # 首次：安装依赖 + 合并 LoRA
#   bash run.sh llm       # 终端 1: 启动 LLM vLLM 服务
#   bash run.sh asr       # 终端 2: 启动 ASR vLLM 服务
#   bash run.sh text      # 终端 3: 文本模式
#   bash run.sh voice     # 终端 3: 语音模式（Mega-ASR transformers + LoRA 路由）
#   bash run.sh stream    # 终端 3: 流式 ASR 模式（推荐）
#   bash run.sh check     # 检查服务是否在线
#   bash run.sh bench     # ASR 性能单测

set -euo pipefail

# 从 config.py 读取关键配置
read_config() {
    python3 -c "
import config
print(f'LLM_MODEL_PATH={config.LLM_MODEL_PATH}')
print(f'LLM_PORT={config.VLLM_PORT}')
print(f'MEGA_ASR_CKPT_DIR={config.MEGA_ASR_CKPT_DIR}')
print(f'ASR_VLLM_PORT={config.ASR_VLLM_PORT}')
print(f'ASR_VLLM_GPU_MEMORY_UTILIZATION={config.ASR_VLLM_GPU_MEMORY_UTILIZATION}')
print(f'VLLM_GPU_MEMORY_UTILIZATION={config.VLLM_GPU_MEMORY_UTILIZATION}')
print(f'VLLM_HOST={config.VLLM_HOST}')
print(f'ASR_VLLM_HOST={config.ASR_VLLM_HOST}')
"
}

eval "$(read_config)"

ASR_MATERIALIZED="${MEGA_ASR_CKPT_DIR}/mega-asr-vllm-materialized"
LLM_SERVED_NAME="$(basename "${LLM_MODEL_PATH}" | sed 's:/*$::')"

case "${1:-help}" in

    setup)
        echo "=== 安装依赖 ==="
        pip install torch --index-url https://download.pytorch.org/whl/cu126
        pip install -r requirements.txt

        echo ""
        echo "=== 合并 Mega-ASR LoRA 权重（一次性） ==="
        python3 scripts/merge_lora.py \
            --base "${MEGA_ASR_CKPT_DIR}/Qwen3-ASR-1.7B" \
            --lora "${MEGA_ASR_CKPT_DIR}/mega-asr-merged" \
            --output "${MEGA_ASR_CKPT_DIR}/mega-asr-vllm-materialized"

        echo ""
        echo "=== 完成！接下来运行: ==="
        echo "  终端 1: bash run.sh llm"
        echo "  终端 2: bash run.sh asr"
        echo "  终端 3: bash run.sh stream"
        ;;

    llm)
        echo "=== 启动 LLM vLLM 服务 ==="
        echo "  模型: ${LLM_MODEL_PATH}"
        echo "  端口: ${LLM_PORT}"
        echo "  日志: assets/logs/vllm_server.log"
        echo ""

        mkdir -p assets/logs

        python3 -m vllm.entrypoints.openai.api_server \
            --model "${LLM_MODEL_PATH}" \
            --served-model-name "${LLM_SERVED_NAME}" \
            --tokenizer_mode mistral \
            --config_format mistral \
            --load_format mistral \
            --enable-auto-tool-choice \
            --tool-call-parser mistral \
            --host "${VLLM_HOST}" \
            --port "${LLM_PORT}" \
            --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
            --max-model-len 8192 \
            2>&1 | tee assets/logs/vllm_server.log
        ;;

    asr)
        # 检查是否已合并
        if [ ! -d "${ASR_MATERIALIZED}" ]; then
            echo "[ERROR] 未找到合并后的 checkpoint: ${ASR_MATERIALIZED}"
            echo "  请先运行: bash run.sh setup"
            exit 1
        fi

        echo "=== 启动 ASR vLLM 服务 ==="
        echo "  模型: ${ASR_MATERIALIZED}"
        echo "  端口: ${ASR_VLLM_PORT}"
        echo ""

        vllm serve "${ASR_MATERIALIZED}" \
            --host "${ASR_VLLM_HOST}" \
            --port "${ASR_VLLM_PORT}" \
            --gpu-memory-utilization "${ASR_VLLM_GPU_MEMORY_UTILIZATION}" \
            --dtype auto
        ;;

    text)
        echo "=== 文本模式 ==="
        python3 step4_main.py --mode text
        ;;

    voice)
        echo "=== 语音模式 (Mega-ASR transformers + LoRA 路由) ==="
        python3 step4_main.py --mode voice
        ;;

    stream)
        echo "=== 流式 ASR 模式 (推荐) ==="
        python3 step4_main.py --mode stream
        ;;

    check)
        echo "=== 检查服务状态 ==="

        LLM_HOST="localhost"
        [ "${VLLM_HOST}" = "0.0.0.0" ] || LLM_HOST="${VLLM_HOST}"

        ASR_HOST="localhost"
        [ "${ASR_VLLM_HOST}" = "0.0.0.0" ] || ASR_HOST="${ASR_VLLM_HOST}"

        echo -n "LLM 服务 (port ${LLM_PORT}): "
        if curl -s "http://${LLM_HOST}:${LLM_PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'在线，模型: {[m[\"id\"] for m in d[\"data\"]]}')" 2>/dev/null; then
            :
        else
            echo "离线"
        fi

        echo -n "ASR 服务 (port ${ASR_VLLM_PORT}): "
        if curl -s "http://${ASR_HOST}:${ASR_VLLM_PORT}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'在线，模型: {[m[\"id\"] for m in d[\"data\"]]}')" 2>/dev/null; then
            :
        else
            echo "离线"
        fi
        ;;

    bench)
        shift
        echo "=== ASR 性能单测 ==="
        python3 scripts/bench_asr.py "$@"
        ;;

    help|*)
        echo "用法: bash run.sh <命令>"
        echo ""
        echo "命令:"
        echo "  setup    首次设置（安装依赖 + 合并 LoRA）"
        echo "  llm      启动 LLM vLLM 服务（终端 1）"
        echo "  asr      启动 ASR vLLM 服务（终端 2）"
        echo "  text     文本模式（终端 3）"
        echo "  voice    语音模式 - Mega-ASR transformers + LoRA 路由（终端 3）"
        echo "  stream   流式 ASR 模式 - 推荐（终端 3）"
        echo "  check    检查 LLM / ASR 服务是否在线"
        echo "  bench    ASR 性能单测 (参数: --mode api/local --audio <path> [--gt <text>] [--rounds N])"
        ;;
esac
