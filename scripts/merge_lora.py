"""独立 LoRA 合并脚本 — 不依赖 MegaASR 类，仅用 peft + transformers。

用法:
    python3 scripts/merge_lora.py \
        --base /home/yzb/voice_assistant/model/Mega-ASR/Qwen3-ASR-1.7B \
        --lora /home/yzb/voice_assistant/model/Mega-ASR/mega-asr-merged \
        --output /home/yzb/voice_assistant/model/Mega-ASR/mega-asr-vllm-materialized
"""

import argparse
import os
import shutil


def merge_lora(base_path: str, lora_path: str, output_path: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1/4] 加载基础模型: {base_path}")
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype="auto", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    print(f"[2/4] 加载 LoRA 权重: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print(f"[3/4] 合并 LoRA 到基础权重...")
    model = model.merge_and_unload()

    print(f"[4/4] 保存到: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # 确保 config.json 保留原始 model_type，vLLM 需要识别
    print("[完成] 合并成功！")
    print(f"  输出: {output_path}")
    print(f"  启动 vLLM: vllm serve {output_path} --port 8001 --gpu-memory-utilization 0.12 --dtype auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 Mega-ASR LoRA 权重")
    parser.add_argument("--base", required=True, help="基础模型路径 (Qwen3-ASR-1.7B)")
    parser.add_argument("--lora", required=True, help="LoRA 权重路径 (mega-asr-merged)")
    parser.add_argument("--output", required=True, help="合并后输出路径")
    args = parser.parse_args()
    merge_lora(args.base, args.lora, args.output)
