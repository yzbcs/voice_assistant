# Repository Guidelines

## Project Structure & Module Organization

This repository is a local voice assistant prototype built around two vLLM services and a Python CLI pipeline.

- `config.py` centralizes model paths, ports, ASR settings, recording settings, and prompts.
- `step1_setup_vllm.py` starts the LLM vLLM server.
- `step2_asr_module.py` contains ASR logic, including transformers file mode and vLLM Transcriptions API streaming mode.
- `step3_agent_core.py` defines the LangChain/LangGraph agent and tool registrations.
- `step4_main.py` is the CLI entry point for text, voice, and stream modes.
- `assets/` stores sample audio or local test inputs; `docs/` stores architecture notes.

## Build, Test, and Development Commands

Create an environment, install CUDA-compatible PyTorch first, then install project dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Configure local model paths in `config.py` before running.

```bash
python3 step1_setup_vllm.py
vllm serve <ASR_MODEL_PATH> --port 8001 --gpu-memory-utilization 0.15 --dtype auto
python3 step4_main.py --mode text
python3 step4_main.py --mode stream
python3 step4_main.py --mode voice
```

Use `--mode stream` for the recommended ASR path. Avoid `qwen-asr[vllm]`; it conflicts with the pinned vLLM version.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, type annotations for public functions, and clear docstrings where the LLM depends on descriptions, especially tools in `step3_agent_core.py`. Keep configuration values in `config.py` instead of hard-coding ports, model names, or prompts in modules. Prefer descriptive snake_case for variables/functions and PascalCase for classes.

## Testing Guidelines

No formal test suite is currently present. For changes, run the narrowest relevant mode manually:

```bash
python3 step4_main.py --mode text
python3 step4_main.py --mode stream
```

When adding tests, place them under `tests/`, name files `test_*.py`, and prefer `pytest`. Mock external vLLM and network calls where practical.

## Commit & Pull Request Guidelines

Recent history uses concise messages such as `feat: ...`, `fix: ...`, and `Update ...`. Prefer Conventional Commit prefixes for behavior changes, for example `feat: add streaming ASR retry` or `fix: handle empty transcription`.

Pull requests should include a short summary, affected modes (`text`, `voice`, `stream`), required config changes, and manual validation results. Include logs or screenshots only when they clarify CLI behavior or service startup issues.

## Security & Configuration Tips

Do not commit local model paths, credentials, API keys, generated audio, or large model artifacts. Keep environment-specific values in `config.py` during local experiments and document any required changes in the PR.
