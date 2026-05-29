"""独立 LoRA 合并脚本 — 从 Mega-ASR 官方 LoRADeltaSwitch 提取合并逻辑。

不依赖 qwen-asr、MegaASR 类或 vLLM，仅需 torch + safetensors。

用法:
    python3 scripts/merge_lora.py \
        --base /path/to/Mega-ASR/Qwen3-ASR-1.7B \
        --lora /path/to/Mega-ASR/mega-asr-merged \
        --output /path/to/Mega-ASR/mega-asr-vllm-materialized
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _load_state(path: str) -> dict[str, torch.Tensor]:
    """加载 safetensors 或 bin 格式的权重。"""
    st_path = os.path.join(path, "adapter_model.safetensors")
    if os.path.exists(st_path):
        print(f"  加载 safetensors: {st_path}")
        return load_file(st_path)
    bin_path = os.path.join(path, "adapter_model.bin")
    if os.path.exists(bin_path):
        print(f"  加载 bin: {bin_path}")
        return torch.load(bin_path, map_location="cpu")
    # 尝试直接作为目录下的多个 safetensors
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if files:
        state = {}
        for f in files:
            print(f"  加载: {f}")
            state.update(load_file(os.path.join(path, f)))
        return state
    raise FileNotFoundError(f"未找到权重文件: {path}")


def _load_base_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """加载基础模型的所有权重文件。"""
    weights = {}
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not files:
        raise FileNotFoundError(f"未找到 safetensors 文件: {model_dir}")
    for f in files:
        print(f"  加载: {f}")
        weights.update(load_file(os.path.join(model_dir, f)))
    return weights


def _normalize_key(key: str) -> str:
    """标准化 LoRA key 名称。"""
    for prefix in ("base_model.model.",):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key.startswith("thinker.layers."):
        key = key.replace("thinker.layers.", "thinker.model.layers.", 1)
    return key


def _key_candidates(name: str) -> list[str]:
    """生成可能的 base weight key 名称候选列表。"""
    candidates = [name]
    if name.startswith("model."):
        candidates.append(name[len("model."):])
    if name.startswith("thinker.layers."):
        candidates.append(name.replace("thinker.layers.", "thinker.model.layers.", 1))
    if name.startswith("thinker.model."):
        candidates.append(name.replace("thinker.model.", "thinker.", 1))
    return list(dict.fromkeys(candidates))


def _parse_lora_key(key: str) -> tuple[str | None, str | None]:
    """解析 LoRA key，返回 (module_name, kind_A_or_B)。"""
    key = _normalize_key(key)
    for marker, kind in [(".lora_A.", "A"), (".lora_B.", "B")]:
        if marker in key:
            module_name = key.split(marker)[0]
            return module_name, kind
    return None, None


def _find_base_key(module_name: str, base_keys: set[str]) -> str | None:
    """在 base 权重中查找匹配的 key。"""
    for candidate in _key_candidates(module_name):
        target = candidate + ".weight"
        if target in base_keys:
            return target
    return None


def merge_lora(base_path: str, lora_path: str, output_path: str):
    """合并 LoRA 权重到基础模型权重。

    核心公式: merged_weight = base_weight + B @ A * (alpha / rank)
    支持 block-wise LoRA (mega_lora_blocks.json)。
    """
    # 1. 加载 LoRA 配置
    print("[1/5] 加载 LoRA 配置")
    config_path = os.path.join(lora_path, "adapter_config.json")
    with open(config_path, "r") as f:
        adapter_config = json.load(f)

    lora_alpha = adapter_config.get("lora_alpha", 1)
    rank = adapter_config.get("r")
    alpha_pattern = adapter_config.get("alpha_pattern") or {}
    rank_pattern = adapter_config.get("rank_pattern") or {}
    fan_in_fan_out = bool(adapter_config.get("fan_in_fan_out", False))

    # block-wise LoRA
    blocks_path = os.path.join(lora_path, "mega_lora_blocks.json")
    blocks = {}
    if os.path.exists(blocks_path):
        with open(blocks_path, "r") as f:
            blocks = json.load(f)
        print(f"  发现 block-wise LoRA: {len(blocks)} 个模块")

    # 2. 加载权重
    print("[2/5] 加载权重")
    base_weights = _load_base_weights(base_path)
    adapter_state = _load_state(lora_path)
    base_keys = set(base_weights.keys())

    # 3. 分组 LoRA A/B 对
    print("[3/5] 解析 LoRA 权重对")
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in adapter_state.items():
        module_name, kind = _parse_lora_key(key)
        if module_name is None:
            continue
        item = grouped.setdefault(module_name, {})
        item[kind] = tensor.cpu()

    print(f"  找到 {len(grouped)} 个 LoRA 权重对")

    # 4. 合并
    print("[4/5] 合并 LoRA 到基础权重")
    merged_count = 0
    missing = []

    for module_name, pair in grouped.items():
        if "A" not in pair or "B" not in pair:
            continue

        base_key = _find_base_key(module_name, base_keys)
        if base_key is None:
            missing.append(module_name)
            continue

        a_matrix = pair["A"].to(torch.float32)
        b_matrix = pair["B"].to(torch.float32)

        # block-wise or standard
        module_blocks = blocks.get(module_name) or blocks.get(base_key.replace(".weight", ""))
        if module_blocks:
            deltas = []
            for block in module_blocks:
                start, end = int(block["start"]), int(block["end"])
                block_rank = int(block.get("rank", end - start))
                block_alpha = int(block.get("alpha", block_rank))
                delta = torch.matmul(b_matrix[:, start:end], a_matrix[start:end])
                delta = delta * (float(block_alpha) / float(block_rank))
                if fan_in_fan_out:
                    delta = delta.T
                deltas.append(delta)
        else:
            adapter_rank = rank_pattern.get(module_name, rank)
            if adapter_rank is None:
                adapter_rank = a_matrix.shape[0]
            adapter_alpha = alpha_pattern.get(module_name, lora_alpha)
            scaling = float(adapter_alpha) / float(adapter_rank)
            delta = torch.matmul(b_matrix, a_matrix) * scaling
            if fan_in_fan_out:
                delta = delta.T
            deltas = [delta]

        for delta in deltas:
            base_weight = base_weights[base_key]
            if delta.shape != base_weight.shape:
                delta = delta.reshape(base_weight.shape)
            base_weights[base_key] = base_weight + delta.to(dtype=base_weight.dtype)
            merged_count += 1

    print(f"  合并了 {merged_count} 个权重")
    if missing:
        print(f"  WARNING: {len(missing)} 个模块未匹配: {missing[:5]}")

    # 5. 保存
    print(f"[5/5] 保存到: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # 分片保存 (每片 <5GB)
    max_shard = "5GB"
    save_file(base_weights, os.path.join(output_path, "model.safetensors"))

    # 复制基础模型的配置文件
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json",
                  "preprocessor_config.json", "processor_config.json"]:
        src = os.path.join(base_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, fname))
            print(f"  复制: {fname}")

    print(f"\n[完成] 合并成功！")
    print(f"  输出: {output_path}")
    print(f"  启动: vllm serve {output_path} --port 8001 --gpu-memory-utilization 0.12 --dtype auto")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 Mega-ASR LoRA 权重 (独立脚本)")
    parser.add_argument("--base", required=True, help="基础模型路径 (Qwen3-ASR-1.7B)")
    parser.add_argument("--lora", required=True, help="LoRA 权重路径 (mega-asr-merged)")
    parser.add_argument("--output", required=True, help="合并后输出路径")
    args = parser.parse_args()
    merge_lora(args.base, args.lora, args.output)
