#!/usr/bin/env python3
"""
llm-fit — scan this machine and recommend open-weight LLMs that will run well on it.

Works on Windows, macOS (Intel + Apple Silicon) and Linux using only the Python
standard library. Optional: `pip install psutil` for more accurate memory info.

Examples
    python llm_fit.py                      # scan + recommend
    python llm_fit.py --use coding         # favour coding models
    python llm_fit.py --ctx 32768          # plan for a 32K context window
    python llm_fit.py --all                # also list models that don't fit
    python llm_fit.py --json > report.json
    python llm_fit.py --vram 12 --gpu-name "RTX 4070" --ram 32   # "what if" mode

All speed and memory numbers are estimates (memory-bandwidth model). Treat them as
a ranking guide, not a benchmark.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, asdict

CATALOG_VERSION = "2026-10"

# --------------------------------------------------------------------------------------
# Model catalog
#   params_b : total parameters (billions)
#   active_b : parameters active per token (Mixture-of-Experts models only)
#   quality  : rough relative capability score, 0-100 (a ranking aid, not a benchmark)
#   uses     : general | coding | reasoning | vision | tiny
#   fixed    : [quant_label, size_GB] for models that ship in a single format
#   verify   : newer tag — confirm it exists on ollama.com/library before relying on it
# Edit freely, or pass your own file with --catalog (see --dump-catalog for the format).
# --------------------------------------------------------------------------------------
DEFAULT_CATALOG = [
    {"name": "Llama 3.2 1B", "tag": "llama3.2:1b", "params_b": 1.24, "quality": 18, "uses": ["general", "tiny"], "license": "Llama"},
    {"name": "Llama 3.2 3B", "tag": "llama3.2:3b", "params_b": 3.21, "quality": 28, "uses": ["general", "tiny"], "license": "Llama"},
    {"name": "Phi-4 mini", "tag": "phi4-mini", "params_b": 3.84, "quality": 34, "uses": ["general", "reasoning", "tiny"], "license": "MIT"},
    {"name": "Gemma 3 4B", "tag": "gemma3:4b", "params_b": 4.3, "quality": 36, "uses": ["general", "vision", "tiny"], "license": "Gemma"},
    {"name": "Qwen3 4B", "tag": "qwen3:4b", "params_b": 4.0, "quality": 40, "uses": ["general", "reasoning", "tiny"], "license": "Apache-2.0"},
    {"name": "Gemma 4 E4B", "tag": "gemma4:e4b", "params_b": 8.0, "active_b": 4.0, "quality": 47, "uses": ["general", "vision"], "license": "Gemma", "verify": True},
    {"name": "Qwen2.5-Coder 7B", "tag": "qwen2.5-coder:7b", "params_b": 7.6, "quality": 44, "uses": ["coding"], "license": "Apache-2.0"},
    {"name": "DeepSeek-R1 8B", "tag": "deepseek-r1:8b", "params_b": 8.2, "quality": 50, "uses": ["reasoning"], "license": "MIT"},
    {"name": "Qwen3 8B", "tag": "qwen3:8b", "params_b": 8.2, "quality": 52, "uses": ["general", "reasoning"], "license": "Apache-2.0"},
    {"name": "Gemma 3 12B", "tag": "gemma3:12b", "params_b": 12.2, "quality": 55, "uses": ["general", "vision"], "license": "Gemma"},
    {"name": "Qwen2.5-Coder 14B", "tag": "qwen2.5-coder:14b", "params_b": 14.7, "quality": 54, "uses": ["coding"], "license": "Apache-2.0"},
    {"name": "DeepSeek-R1 14B", "tag": "deepseek-r1:14b", "params_b": 14.8, "quality": 57, "uses": ["reasoning"], "license": "MIT"},
    {"name": "Qwen3 14B", "tag": "qwen3:14b", "params_b": 14.8, "quality": 60, "uses": ["general", "reasoning"], "license": "Apache-2.0"},
    {"name": "gpt-oss 20B", "tag": "gpt-oss:20b", "params_b": 20.9, "active_b": 3.6, "quality": 64, "uses": ["general", "reasoning", "coding"], "license": "Apache-2.0", "fixed": ["MXFP4", 13.8]},
    {"name": "Mistral Small 3.1 24B", "tag": "mistral-small3.1:24b", "params_b": 24.0, "quality": 60, "uses": ["general", "vision"], "license": "Apache-2.0"},
    {"name": "Devstral 24B", "tag": "devstral:24b", "params_b": 23.6, "quality": 62, "uses": ["coding"], "license": "Apache-2.0"},
    {"name": "Gemma 3 27B", "tag": "gemma3:27b", "params_b": 27.4, "quality": 63, "uses": ["general", "vision"], "license": "Gemma"},
    {"name": "Qwen3.6 27B", "tag": "qwen3.6:27b", "params_b": 27.0, "quality": 74, "uses": ["general", "reasoning", "coding"], "license": "Apache-2.0", "verify": True},
    {"name": "Qwen3 30B-A3B", "tag": "qwen3:30b", "params_b": 30.5, "active_b": 3.3, "quality": 65, "uses": ["general", "reasoning"], "license": "Apache-2.0"},
    {"name": "Qwen3-Coder 30B-A3B", "tag": "qwen3-coder:30b", "params_b": 30.5, "active_b": 3.3, "quality": 68, "uses": ["coding"], "license": "Apache-2.0"},
    {"name": "Qwen2.5-Coder 32B", "tag": "qwen2.5-coder:32b", "params_b": 32.8, "quality": 64, "uses": ["coding"], "license": "Apache-2.0"},
    {"name": "DeepSeek-R1 32B", "tag": "deepseek-r1:32b", "params_b": 32.8, "quality": 66, "uses": ["reasoning"], "license": "MIT"},
    {"name": "Qwen3 32B", "tag": "qwen3:32b", "params_b": 32.8, "quality": 67, "uses": ["general", "reasoning"], "license": "Apache-2.0"},
    {"name": "Qwen3.6 35B-A3B", "tag": "qwen3.6:35b", "params_b": 35.0, "active_b": 3.0, "quality": 72, "uses": ["general", "reasoning", "coding"], "license": "Apache-2.0", "verify": True},
    {"name": "Llama 3.3 70B", "tag": "llama3.3:70b", "params_b": 70.6, "quality": 66, "uses": ["general"], "license": "Llama"},
    {"name": "DeepSeek-R1 70B", "tag": "deepseek-r1:70b", "params_b": 70.6, "quality": 68, "uses": ["reasoning"], "license": "Llama"},
    {"name": "gpt-oss 120B", "tag": "gpt-oss:120b", "params_b": 116.8, "active_b": 5.1, "quality": 78, "uses": ["general", "reasoning", "coding"], "license": "Apache-2.0", "fixed": ["MXFP4", 65.0]},
]

# (label, effective bits per weight incl. scales, quality penalty)
QUANTS = [
    ("Q8_0", 8.5, 0.0),
    ("Q6_K", 6.6, -0.5),
    ("Q5_K_M", 5.7, -1.0),
    ("Q4_K_M", 4.85, -2.0),
    ("Q3_K_M", 3.9, -6.0),
]

# Memory bandwidth tables (GB/s). Order matters: more specific patterns first.
NVIDIA_BW = [
    (r"5090", 1792), (r"5080", 960), (r"5070 ?Ti", 896), (r"5070", 672), (r"5060 ?Ti", 448), (r"5060", 448),
    (r"4090", 1008), (r"4080", 717), (r"4070 ?Ti", 504), (r"4070", 504), (r"4060 ?Ti", 288), (r"4060", 272),
    (r"3090", 936), (r"3080 ?Ti", 912), (r"3080", 760), (r"3070", 448), (r"3060 ?Ti", 448), (r"3060", 360), (r"3050", 224),
    (r"2080 ?Ti", 616), (r"2080", 448), (r"2070", 448), (r"2060", 336), (r"1660", 192), (r"1650", 128),
    (r"1080 ?Ti", 484), (r"1080", 320), (r"1070", 256), (r"1060", 192),
    (r"H200", 4800), (r"H100", 3350), (r"A100", 1935), (r"L40", 864), (r"A6000", 768), (r"A5000", 768), (r"A4000", 448),
    (r"T4", 320), (r"V100", 900),
]
AMD_BW = [
    (r"9070 ?XT", 640), (r"9070", 640), (r"7900 ?XTX", 960), (r"7900 ?XT", 800), (r"7900 ?GRE", 576), (r"7800 ?XT", 624),
    (r"7700 ?XT", 432), (r"7600", 288), (r"6950", 576), (r"6900", 512), (r"6800", 512), (r"6750", 432), (r"6700", 384),
    (r"6600", 224), (r"MI300", 5300), (r"MI250", 3200), (r"MI210", 1600),
]
INTEL_BW = [(r"B580", 456), (r"B570", 380), (r"A770", 560), (r"A750", 512), (r"A580", 512), (r"A380", 186)]
APPLE_BW = {  # chip -> GB/s
    "M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": 400, "M3 Ultra": 819,
    "M4": 120, "M4 Pro": 273, "M4 Max": 546,
    "M5": 153,
}

GPU_EFF = 0.65   # fraction of peak bandwidth realistically achieved during decoding
CPU_EFF = 0.60
TPS_CAP = 250    # beyond this, compute/overhead dominates; don't over-promise


# --------------------------------------------------------------------------------------
# Hardware detection
# --------------------------------------------------------------------------------------
@dataclass
class GPU:
    name: str
    vendor: str
    vram_gb: float
    bw_gbs: float
    bw_guessed: bool = False


@dataclass
class Hardware:
    os: str = ""
    cpu: str = ""
    cores_logical: int = 0
    cores_physical: int | None = None
    cpu_flags: list = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_avail_gb: float | None = None
    ram_bw_gbs: float = 60.0
    ram_bw_guessed: bool = True
    gpus: list = field(default_factory=list)
    unified: bool = False          # Apple Silicon: GPU shares system RAM
    disk_free_gb: float = 0.0
    ollama: str | None = None
    ollama_models: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    # derived budgets
    @property
    def vram_total(self):
        return sum(g.vram_gb for g in self.gpus)

    @property
    def gpu_bw(self):
        return min((g.bw_gbs for g in self.gpus), default=0.0)

    @property
    def gpu_budget(self):
        if self.unified:
            frac = 0.70 if self.ram_total_gb <= 36 else 0.75   # macOS caps GPU-wired memory
            return self.ram_total_gb * frac
        if not self.gpus:
            return 0.0
        return max(0.0, self.vram_total * 0.92 - 0.3)

    @property
    def ram_budget(self):
        if self.unified:
            return 0.0  # same pool as the GPU budget
        return max(0.0, self.ram_total_gb * 0.75 - 1.0)


def run(cmd, timeout=8):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return ""


def read(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""


def lookup_bw(name, table, default):
    for pat, bw in table:
        if re.search(pat, name, re.I):
            return bw, False
    return default, True


def detect_cpu(hw):
    sysname = platform.system()
    if sysname == "Linux":
        info = read("/proc/cpuinfo")
        m = re.search(r"model name\s*:\s*(.+)", info)
        hw.cpu = m.group(1).strip() if m else platform.processor()
        fm = re.search(r"^flags\s*:\s*(.+)$", info, re.M)
        if fm:
            flags = set(fm.group(1).split())
            hw.cpu_flags = [f for f in ("avx2", "avx512f", "fma", "f16c") if f in flags]
        if not hw.cpu:  # ARM boards
            m = re.search(r"Model\s*:\s*(.+)", info)
            hw.cpu = m.group(1).strip() if m else platform.machine()
    elif sysname == "Darwin":
        hw.cpu = run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    elif sysname == "Windows":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            hw.cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            hw.cpu = platform.processor()
    hw.cores_logical = os.cpu_count() or 0
    try:
        import psutil
        hw.cores_physical = psutil.cpu_count(logical=False)
    except Exception:
        pass


def detect_ram(hw):
    try:
        import psutil
        vm = psutil.virtual_memory()
        hw.ram_total_gb, hw.ram_avail_gb = vm.total / 1e9, vm.available / 1e9
        return
    except Exception:
        pass
    sysname = platform.system()
    if sysname == "Linux":
        info = read("/proc/meminfo")
        t = re.search(r"MemTotal:\s+(\d+)", info)
        a = re.search(r"MemAvailable:\s+(\d+)", info)
        if t:
            hw.ram_total_gb = int(t.group(1)) * 1024 / 1e9
        if a:
            hw.ram_avail_gb = int(a.group(1)) * 1024 / 1e9
    elif sysname == "Darwin":
        v = run(["sysctl", "-n", "hw.memsize"])
        if v.isdigit():
            hw.ram_total_gb = int(v) / 1e9
    elif sysname == "Windows":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(st)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            hw.ram_total_gb, hw.ram_avail_gb = st.ullTotalPhys / 1e9, st.ullAvailPhys / 1e9


def detect_ram_bandwidth(hw):
    """Best-effort system RAM bandwidth. Only Windows exposes speed without admin rights."""
    if platform.system() == "Windows":
        out = run(["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_PhysicalMemory | Select-Object Speed,ConfiguredClockSpeed | ConvertTo-Json"])
        try:
            data = json.loads(out)
            sticks = data if isinstance(data, list) else [data]
            speed = max((s.get("ConfiguredClockSpeed") or s.get("Speed") or 0) for s in sticks)
            if speed:
                channels = 2 if len(sticks) >= 2 else 1
                hw.ram_bw_gbs = speed * 8 * channels / 1000
                hw.ram_bw_guessed = False
                if channels == 1:
                    hw.notes.append("Only one RAM stick detected (single-channel): CPU inference will be slower. "
                                    "Adding a matching second stick roughly doubles CPU token speed.")
                return
        except Exception:
            pass
    hw.ram_bw_gbs, hw.ram_bw_guessed = 60.0, True


def detect_nvidia():
    out = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].replace(".", "").isdigit():
            name, mib = parts[0], float(parts[1])
            bw, guessed = lookup_bw(name, NVIDIA_BW, 300)
            if "laptop" in name.lower() or "mobile" in name.lower():
                bw *= 0.6
            gpus.append(GPU(name, "NVIDIA", mib * 1.048576 / 1000, bw, guessed))
    return gpus


def detect_amd_linux():
    gpus = []
    lspci = run(["lspci"])
    names = [m.group(1) for m in re.finditer(r"(?:VGA|Display|3D)[^:]*:.*(?:AMD|ATI).*?\[([^\]]*Radeon[^\]]*|[^\]]*Instinct[^\]]*)\]", lspci)]
    for i, dev in enumerate(sorted(glob.glob("/sys/class/drm/card[0-9]*/device"))):
        if read(f"{dev}/vendor") != "0x1002":
            continue
        total = read(f"{dev}/mem_info_vram_total")
        if not total.isdigit():
            continue
        vram = int(total) / 1e9
        if vram < 2:          # APU carve-out, not a real GPU pool
            continue
        name = read(f"{dev}/product_name") or (names[len(gpus)] if len(gpus) < len(names) else "AMD Radeon GPU")
        bw, guessed = lookup_bw(name, AMD_BW, 400)
        gpus.append(GPU(name, "AMD", vram, bw, guessed))
    return gpus


def detect_windows_registry_gpus(skip_nvidia):
    gpus = []
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if not re.fullmatch(r"\d{4}", sub):
                continue
            try:
                k = winreg.OpenKey(root, sub)
                desc = winreg.QueryValueEx(k, "DriverDesc")[0]
                mem = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0]
                if isinstance(mem, bytes):
                    mem = int.from_bytes(mem, "little")
            except OSError:
                continue
            low = desc.lower()
            if skip_nvidia and "nvidia" in low:
                continue
            if "intel" in low and "arc" not in low:
                continue                     # integrated graphics: no dedicated pool
            vram = mem / 1e9
            if vram < 2:
                continue
            if "nvidia" in low:
                vendor, (bw, g) = "NVIDIA", lookup_bw(desc, NVIDIA_BW, 300)
            elif "amd" in low or "radeon" in low:
                vendor, (bw, g) = "AMD", lookup_bw(desc, AMD_BW, 400)
            elif "arc" in low:
                vendor, (bw, g) = "Intel", lookup_bw(desc, INTEL_BW, 400)
            else:
                continue
            gpus.append(GPU(desc, vendor, vram, bw, g))
    except Exception:
        pass
    return gpus


def detect_gpus(hw):
    sysname = platform.system()
    if sysname == "Darwin" and platform.machine() == "arm64":
        m = re.search(r"Apple (M\d)(?: (Pro|Max|Ultra))?", hw.cpu)
        chip = f"{m.group(1)} {m.group(2)}".strip() if m and m.group(2) else (m.group(1) if m else "")
        bw = APPLE_BW.get(chip)
        hw.unified = True
        hw.gpus = [GPU(f"Apple {chip or 'Silicon'} GPU (unified memory)", "Apple",
                       hw.ram_total_gb, bw or 150, bw is None)]
        return
    gpus = detect_nvidia()
    if sysname == "Linux":
        gpus += detect_amd_linux()
    elif sysname == "Windows":
        gpus += detect_windows_registry_gpus(skip_nvidia=bool(gpus))
    hw.gpus = gpus
    for g in gpus:
        if g.vendor == "Intel":
            hw.notes.append(f"{g.name}: Ollama's Intel GPU support is limited; LM Studio or llama.cpp (Vulkan/SYCL) "
                            "will use it more reliably.")
        if g.vendor == "AMD" and sysname == "Windows":
            hw.notes.append(f"{g.name}: check it's on Ollama's supported AMD list; otherwise use LM Studio (Vulkan).")


def detect_disk_and_ollama(hw):
    models_dir = os.environ.get("OLLAMA_MODELS") or os.path.join(os.path.expanduser("~"), ".ollama", "models")
    probe = models_dir if os.path.exists(models_dir) else os.path.expanduser("~")
    try:
        hw.disk_free_gb = shutil.disk_usage(probe).free / 1e9
    except Exception:
        pass
    hw.ollama = shutil.which("ollama")
    if hw.ollama:
        out = run(["ollama", "list"], timeout=10)
        for line in out.splitlines()[1:]:
            if line.strip():
                hw.ollama_models.append(line.split()[0])


def scan(args):
    hw = Hardware(os=f"{platform.system()} {platform.release()} ({platform.machine()})")
    detect_cpu(hw)
    detect_ram(hw)
    detect_ram_bandwidth(hw)
    if not args.no_gpu:
        detect_gpus(hw)
    detect_disk_and_ollama(hw)

    # "what if" overrides
    if args.ram:
        hw.ram_total_gb, hw.ram_avail_gb = args.ram, None
        if hw.unified:
            hw.gpus[0].vram_gb = args.ram
    if args.ram_bw:
        hw.ram_bw_gbs, hw.ram_bw_guessed = args.ram_bw, False
    if args.no_gpu:
        hw.gpus, hw.unified = [], False
    if args.vram is not None:
        name = args.gpu_name or "Simulated GPU"
        bw, guessed = lookup_bw(name, NVIDIA_BW + AMD_BW + INTEL_BW, 300)
        hw.gpus, hw.unified = [GPU(name, "Simulated", args.vram, args.gpu_bw or bw, guessed and not args.gpu_bw)], False
    elif args.gpu_bw and hw.gpus:
        for g in hw.gpus:
            g.bw_gbs, g.bw_guessed = args.gpu_bw, False
    return hw


# --------------------------------------------------------------------------------------
# Fit + ranking
# --------------------------------------------------------------------------------------
def memory_need(m, bits, ctx):
    weights = m["fixed"][1] if m.get("fixed") else m["params_b"] * bits / 8
    active = m.get("active_b")
    kv_basis = min(m["params_b"], active * 4) if active else m["params_b"]
    kv = 0.38 * math.sqrt(kv_basis) * (ctx / 8192)
    return weights, kv, weights + kv + 0.8


def place(m, weights, need, hw):
    """Return (where, tokens_per_sec) or None if it won't load."""
    active = m.get("active_b") or m["params_b"]
    bytes_per_tok = weights * active / m["params_b"]          # GB read per generated token
    gb, rb = hw.gpu_budget, hw.ram_budget
    if gb > 0 and need <= gb:
        return "GPU", hw.gpu_bw * GPU_EFF / bytes_per_tok
    if hw.unified:
        return None
    if gb > 0 and need <= gb + rb:
        gpu_weights = max(0.0, gb - (need - weights))
        frac = min(1.0, gpu_weights / weights)
        t = bytes_per_tok * frac / (hw.gpu_bw * GPU_EFF) + bytes_per_tok * (1 - frac) / (hw.ram_bw_gbs * CPU_EFF)
        return f"Split {int(frac * 100)}% GPU", 1 / t
    if need <= rb:
        return "CPU", hw.ram_bw_gbs * CPU_EFF / bytes_per_tok
    return None


def evaluate(catalog, hw, args):
    fits, too_slow, no_fit = [], [], []
    for m in catalog:
        options = [(m["fixed"][0], None, 0.0)] if m.get("fixed") else QUANTS
        best = None
        best_slow = None
        for label, bits, pen in options:
            weights, kv, need = memory_need(m, bits or 0, args.ctx)
            p = place(m, weights, need, hw)
            if not p:
                continue
            where, tps = p
            tps = min(tps, TPS_CAP)
            s = m["quality"] + pen
            if args.use != "any":
                s += 12 if args.use in m["uses"] else (-15 if args.use != "general" else 0)
            s += max(-20.0, min(8.0, 4 * math.log2(tps / args.min_speed)))
            if where == "GPU":
                s += 2.0   # fully on GPU: prompt processing is far faster than split/CPU
            rec = {"name": m["name"], "tag": m["tag"], "quant": label, "where": where,
                   "tok_s": round(tps, 1), "mem_gb": round(need, 1), "file_gb": round(weights, 1),
                   "quality": m["quality"], "score": round(s, 1), "uses": m["uses"],
                   "license": m.get("license", ""), "verify": bool(m.get("verify")),
                   "installed": installed(m["tag"], hw.ollama_models),
                   "low_disk": weights > hw.disk_free_gb * 0.9 if hw.disk_free_gb else False}
            if tps >= args.min_speed:
                if not best or rec["score"] > best["score"]:
                    best = rec
            elif not best_slow or rec["tok_s"] > best_slow["tok_s"]:
                best_slow = rec
        if best:
            fits.append(best)
        elif best_slow:
            too_slow.append(best_slow)
        else:
            _, _, need = memory_need(m, 3.9, args.ctx)   # smallest quant we consider
            no_fit.append({"name": m["name"], "tag": m["tag"], "min_mem_gb": round(need, 1)})
    fits.sort(key=lambda r: -r["score"])
    return fits, too_slow, no_fit


def installed(tag, models):
    t = tag if ":" in tag else tag + ":latest"
    return t in models


def picks(fits):
    def top(pred):
        c = [r for r in fits if pred(r)]
        return c[0] if c else None
    by_quality = sorted(fits, key=lambda r: (-r["quality"], -r["tok_s"]))
    snappy = [r for r in by_quality if r["tok_s"] >= 40]
    return [
        ("Best overall", top(lambda r: True)),
        ("Best for coding", top(lambda r: "coding" in r["uses"])),
        ("Best for reasoning", top(lambda r: "reasoning" in r["uses"])),
        ("Best with vision (images)", top(lambda r: "vision" in r["uses"])),
        ("Snappiest good model (40+ tok/s)", snappy[0] if snappy else None),
    ]


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
if os.name == "nt" and USE_COLOR:
    os.system("")  # enable ANSI escapes in Windows terminals


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def speed_label(tps):
    if tps >= 30:
        return c("Fast", "32")
    if tps >= 12:
        return c("Comfortable", "36")
    return c("Usable", "33")


def pull_cmd(r):
    if r["quant"] in ("Q4_K_M", "MXFP4"):
        return f"ollama run {r['tag']}"
    return f"ollama run {r['tag']}   ({r['quant']} if a tag exists; default is Q4_K_M)"


def print_report(hw, fits, too_slow, no_fit, args):
    print(c(f"\nllm-fit  ·  catalog {CATALOG_VERSION}  ·  context {args.ctx:,} tokens  ·  use: {args.use}", "1"))
    print(c("\nYour system", "1;4"))
    print(f"  OS        {hw.os}")
    cores = f"{hw.cores_physical} cores / {hw.cores_logical} threads" if hw.cores_physical else f"{hw.cores_logical} threads"
    flags = f"  [{', '.join(hw.cpu_flags)}]" if hw.cpu_flags else ""
    print(f"  CPU       {hw.cpu}  ({cores}){flags}")
    avail = f", {hw.ram_avail_gb:.1f} GB free now" if hw.ram_avail_gb else ""
    bwnote = " (assumed)" if hw.ram_bw_guessed else ""
    print(f"  RAM       {hw.ram_total_gb:.1f} GB{avail}  ·  ~{hw.ram_bw_gbs:.0f} GB/s{bwnote}")
    if hw.gpus:
        for g in hw.gpus:
            gn = " (estimated)" if g.bw_guessed else ""
            print(f"  GPU       {g.name}  ·  {g.vram_gb:.1f} GB  ·  ~{g.bw_gbs:.0f} GB/s{gn}")
    else:
        print("  GPU       none usable detected — CPU-only inference")
    print(f"  Budget    {hw.gpu_budget:.1f} GB on GPU, {hw.ram_budget:.1f} GB in system RAM")
    print(f"  Disk      {hw.disk_free_gb:.0f} GB free")
    print(f"  Ollama    {'installed (' + str(len(hw.ollama_models)) + ' models pulled)' if hw.ollama else 'not found — get it at ollama.com/download'}")

    if not fits:
        print(c("\nNothing in the catalog runs at your minimum speed.", "31"))
        if too_slow:
            r = max(too_slow, key=lambda r: r["tok_s"])
            print(f"  Closest: {r['name']} at ~{r['tok_s']} tok/s ({r['where']}). Try --min-speed 2.")
        return

    print(c("\nTop picks", "1;4"))
    for label, r in picks(fits):
        if not r:
            continue
        tag = " ✓ installed" if r["installed"] else ""
        ver = c("  (newer tag — verify on ollama.com/library)", "2") if r["verify"] else ""
        print(f"  {c(label, '1')}: {r['name']} {r['quant']}  —  {r['where']}, ~{r['tok_s']:.0f} tok/s "
              f"({speed_label(r['tok_s'])}), needs ~{r['mem_gb']} GB{tag}{ver}")
        print(c(f"      {pull_cmd(r)}", "2"))

    print(c(f"\nRanked list (top {min(args.top, len(fits))} of {len(fits)} that run ≥ {args.min_speed} tok/s)", "1;4"))
    hdr = f"  {'#':>2}  {'Model':<22} {'Quant':<7} {'Where':<14} {'~tok/s':>6}  {'Mem GB':>6}  {'Uses':<26} Notes"
    print(c(hdr, "2"))
    for i, r in enumerate(fits[: args.top], 1):
        notes = []
        if r["installed"]:
            notes.append("installed")
        if r["verify"]:
            notes.append("verify tag")
        if r["low_disk"]:
            notes.append("LOW DISK")
        print(f"  {i:>2}  {r['name']:<22} {r['quant']:<7} {r['where']:<14} {r['tok_s']:>6.0f}  {r['mem_gb']:>6}  "
              f"{','.join(r['uses']):<26} {', '.join(notes)}")

    if args.all:
        if too_slow:
            print(c("\nLoads, but slower than your minimum", "1;4"))
            for r in too_slow:
                print(f"      {r['name']:<22} {r['quant']:<7} {r['where']:<14} ~{r['tok_s']} tok/s")
        if no_fit:
            print(c("\nWon't fit in memory", "1;4"))
            for r in no_fit:
                print(f"      {r['name']:<22} needs ≥ {r['min_mem_gb']} GB")

    print(c("\nNotes", "1;4"))
    for n in hw.notes:
        print(f"  • {n}")
    print("  • Speeds are decode estimates from memory bandwidth; real numbers vary ±30%. "
          "'Split' and CPU modes also process long prompts slowly.")
    print("  • Quality scores are rough relative rankings in the catalog — edit them, or use --catalog with your own.")
    print("  • Larger --ctx needs more memory; if a model is borderline, lower the context or pick the next one down.\n")


def main():
    ap = argparse.ArgumentParser(description="Scan this machine and recommend open-weight LLMs that will run well on it.")
    ap.add_argument("--use", choices=["any", "general", "coding", "reasoning", "vision"], default="any",
                    help="favour models for this job (default: any)")
    ap.add_argument("--ctx", type=int, default=8192, help="context window to plan for (default 8192)")
    ap.add_argument("--min-speed", type=float, default=8.0, help="minimum acceptable tokens/sec (default 8)")
    ap.add_argument("--top", type=int, default=12, help="rows in the ranked list (default 12)")
    ap.add_argument("--all", action="store_true", help="also list models that are too slow or don't fit")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a report")
    ap.add_argument("--catalog", help="path to a custom model catalog JSON")
    ap.add_argument("--dump-catalog", action="store_true", help="print the built-in catalog as JSON and exit")
    sim = ap.add_argument_group("what-if overrides")
    sim.add_argument("--ram", type=float, help="pretend system RAM is this many GB")
    sim.add_argument("--vram", type=float, help="pretend there is one GPU with this many GB of VRAM")
    sim.add_argument("--gpu-name", help="GPU name for --vram (used to look up bandwidth), e.g. 'RTX 4070'")
    sim.add_argument("--gpu-bw", type=float, help="override GPU memory bandwidth in GB/s")
    sim.add_argument("--ram-bw", type=float, help="override system RAM bandwidth in GB/s")
    sim.add_argument("--no-gpu", action="store_true", help="ignore GPUs (CPU-only)")
    args = ap.parse_args()

    if args.dump_catalog:
        print(json.dumps(DEFAULT_CATALOG, indent=2))
        return
    catalog = DEFAULT_CATALOG
    if args.catalog:
        with open(args.catalog) as f:
            catalog = json.load(f)

    hw = scan(args)
    fits, too_slow, no_fit = evaluate(catalog, hw, args)

    if args.json:
        out = {"catalog_version": CATALOG_VERSION, "settings": vars(args),
               "hardware": {**asdict(hw), "gpu_budget_gb": round(hw.gpu_budget, 1), "ram_budget_gb": round(hw.ram_budget, 1)},
               "picks": {k: v for k, v in picks(fits)}, "ranked": fits, "too_slow": too_slow, "no_fit": no_fit}
        print(json.dumps(out, indent=2))
    else:
        print_report(hw, fits, too_slow, no_fit, args)


if __name__ == "__main__":
    main()
