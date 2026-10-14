# llm-fit

[![CI](https://github.com/OWNER/llm-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/llm-fit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Find the best open-source AI model your computer can actually run.**

`llm-fit` scans your CPU, RAM, GPU and disk, then ranks open-weight LLMs (Qwen, gpt-oss, Gemma, DeepSeek-R1, Llama, Devstral…) by what fits in memory, how fast it will run, and how capable it is — and gives you the exact `ollama run` command for each.

- **Zero dependencies.** One Python file, standard library only.
- **Cross-platform.** Windows, macOS (Apple Silicon + Intel), Linux.
- **GPU-aware.** NVIDIA, AMD, Intel Arc, and Apple unified memory, including GPU+RAM split placement.
- **MoE-aware.** Knows that Mixture-of-Experts models (Qwen3 30B-A3B, gpt-oss) only read their active weights per token, so they run fast even when they don't fit fully on the GPU.
- **What-if mode.** Check what a GPU upgrade would get you before buying it.

## Quick start

```bash
git clone https://github.com/OWNER/llm-fit.git
cd llm-fit
python llm_fit.py
```

Or install it as a command:

```bash
pip install .
llm-fit --use coding
```

Optional: `pip install psutil` for slightly more accurate memory and core counts.

## Example output

```
Your system
  OS        Windows 11 (AMD64)
  CPU       AMD Ryzen 7 7700X 8-Core Processor  (8 cores / 16 threads)
  RAM       32.0 GB  ·  ~60 GB/s (assumed)
  GPU       RTX 4070  ·  12.0 GB  ·  ~504 GB/s
  Budget    10.7 GB on GPU, 23.0 GB in system RAM

Top picks
  Best overall: Qwen3.6 35B-A3B Q4_K_M  —  Split 40% GPU, ~31 tok/s (Fast), needs ~23.3 GB
      ollama run qwen3.6:35b
  Best with vision (images): Gemma 3 12B Q4_K_M  —  GPU, ~44 tok/s (Fast), needs ~9.5 GB
      ollama run gemma3:12b

Ranked list (top 6 of 18 that run ≥ 8.0 tok/s)
   #  Model                  Quant   Where          ~tok/s  Mem GB  Uses
   1  Qwen3.6 35B-A3B        Q4_K_M  Split 40% GPU      31    23.3  general,reasoning,coding
   2  Qwen3-Coder 30B-A3B    Q4_K_M  Split 46% GPU      31    20.7  coding
   3  gpt-oss 20B            MXFP4   Split 61% GPU      34    16.0  general,reasoning,coding
   4  Qwen3 30B-A3B          Q4_K_M  Split 46% GPU      31    20.7  general,reasoning
   5  Qwen3 14B              Q4_K_M  Split 94% GPU      25    11.2  general,reasoning
   6  Gemma 3 12B            Q4_K_M  GPU                44     9.5  general,vision
```

## Options

| Flag | What it does |
|---|---|
| `--use coding\|reasoning\|vision\|general` | Favour models for a job |
| `--ctx 32768` | Plan for a longer context window (more memory) |
| `--min-speed 15` | Hide models slower than 15 tokens/sec (default 8) |
| `--top 20` | Rows in the ranked list |
| `--all` | Also show models that are too slow or don't fit |
| `--json` | Machine-readable output |
| `--vram 16 --gpu-name "RTX 4080"` | What-if: pretend you have this GPU |
| `--ram 64` | What-if: pretend you have this much RAM |
| `--gpu-bw`, `--ram-bw` | Override memory bandwidth (GB/s) if you know it |
| `--no-gpu` | CPU-only |
| `--catalog file.json` | Use your own model list (start from `--dump-catalog`) |

## What it detects

| | How |
|---|---|
| CPU name, threads, AVX flags | Windows registry / `sysctl` / `/proc/cpuinfo` |
| RAM total + free | psutil, or OS-native fallbacks |
| RAM bandwidth | Windows: real stick speed and channel count. Elsewhere: assumed 60 GB/s |
| NVIDIA GPUs | `nvidia-smi`, bandwidth from a lookup table |
| AMD GPUs | Linux sysfs, Windows registry |
| Intel Arc | Windows registry |
| Apple Silicon | Chip name → unified-memory bandwidth; ~70–75% of RAM usable by the GPU |
| Disk, Ollama | Free space where models are stored; `ollama list` marks installed models |

## How it works

1. For every model × quantization (Q8 → Q3), estimate memory = weights + KV cache for your context + runtime overhead.
2. Place it: fully on GPU, split across GPU and RAM, CPU-only, or "won't fit".
3. Estimate decode speed from memory bandwidth (token generation is bandwidth-bound): `tok/s ≈ bandwidth × efficiency ÷ bytes read per token`.
4. Score = model quality − quantization penalty + speed bonus (+ bonus for your `--use`, + a small bonus for fitting fully on the GPU). Anything under `--min-speed` is dropped.

## Caveats

- Speeds are estimates (roughly ±30%). Use them to compare options, not as benchmarks.
- Quality scores in the catalog are rough relative rankings and will age as new models ship. PRs updating the catalog are welcome.
- Ollama's default tag is Q4_K_M. If llm-fit recommends Q8/Q6/Q5, look for a matching tag on the model's Ollama page (e.g. `qwen3:8b-q8_0`).
- Entries marked **verify** (Qwen3.6, Gemma 4) are newer; confirm the exact tag on [ollama.com/library](https://ollama.com/library).

## Development

```bash
python -m unittest discover -s tests -v
```

CI runs the tests plus a real hardware scan on Linux, Windows and macOS for every push.

## License

[MIT](LICENSE)
