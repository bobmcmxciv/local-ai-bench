#!/usr/bin/env python3
"""Cross-platform hardware, ISA, memory/VRAM, and NPU capability probe for local-ai-bench.

Outputs a standardized JSON hardware profile used to determine which ASR and LLM
optimization tiers (CPU baseline, CPU vector/INT8, CPU matrix AMX/SME2/AVX-512,
GPU CUDA/Metal/Vulkan/OpenVINO, NPU CoreML/QNN/Neuron/OpenVINO) should be executed,
and computes safe memory/VRAM ceilings so LLM benchmarks do not trigger swap thrashing.
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def run_cmd(cmd, timeout=10):
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=isinstance(cmd, str),
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def detect_cpu_and_isa():
    sys_name = platform.system()
    mach = platform.machine().lower()
    cpu_name = platform.processor() or "Unknown CPU"
    isa_flags = []
    physical_cores = os.cpu_count() or 1
    logical_cores = os.cpu_count() or 1

    if sys_name == "Windows":
        out = run_cmd(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors) | ConvertTo-Json -Compress"])
        if out:
            try:
                info = json.loads(out)
                cpu_name = info.get("Name", cpu_name).strip()
                physical_cores = info.get("NumberOfCores") or physical_cores
                logical_cores = info.get("NumberOfLogicalProcessors") or logical_cores
            except Exception:
                pass
        # Check WSL /proc/cpuinfo if available for full x86 flags
        cpuinfo = run_cmd(["wsl", "--", "cat", "/proc/cpuinfo"])
        if cpuinfo:
            for line in cpuinfo.splitlines():
                if line.startswith("flags"):
                    raw_flags = set(line.split(":", 1)[1].strip().split())
                    for f in ["sse4_2", "avx", "avx2", "fma", "f16c", "avx_vnni",
                              "avx512f", "avx512bw", "avx512dq", "avx512vl",
                              "avx512_vnni", "avx512_bf16", "avx512_vbmi", "amx_tile", "amx_int8"]:
                        if f in raw_flags:
                            isa_flags.append(f.upper().replace("_", "-"))
                    break
        if not isa_flags and ("x86" in mach or "amd64" in mach):
            isa_flags = ["SSE4.2", "AVX2"]
            if any(k in cpu_name for k in ["9950X", "9955HX", "7950X", "9900X", "9700X", "9600X"]):
                isa_flags.extend(["AVX-512F", "AVX512-VNNI", "AVX512-BF16", "AVX512-VBMI"])
            elif "Ultra" in cpu_name:
                isa_flags.append("AVX-VNNI")

    elif sys_name == "Darwin":
        brand = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            cpu_name = brand
        p_cores = run_cmd(["sysctl", "-n", "hw.perflevel0.physicalcpu"])
        e_cores = run_cmd(["sysctl", "-n", "hw.perflevel1.physicalcpu"])
        if p_cores and e_cores:
            physical_cores = int(p_cores) + int(e_cores)
            logical_cores = physical_cores
        if "arm" in mach or "aarch64" in mach:
            isa_flags = ["NEON", "FP16", "DOTPROD", "I8MM", "AMX"]
            sme = run_cmd(["sysctl", "-n", "hw.optional.arm.FEAT_SME"])
            sme2 = run_cmd(["sysctl", "-n", "hw.optional.arm.FEAT_SME2"])
            if sme == "1" or any(m in cpu_name for m in ["M4", "M5", "A18", "A19"]):
                isa_flags.append("SME")
            if sme2 == "1" or any(m in cpu_name for m in ["M4", "M5", "A18", "A19"]):
                isa_flags.append("SME2")

    elif sys_name == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            for line in text.splitlines():
                if "model name" in line or "Hardware" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
            for line in text.splitlines():
                if line.startswith("flags") or line.startswith("Features"):
                    raw_flags = set(line.split(":", 1)[1].strip().split())
                    for f in ["sse4_2", "avx", "avx2", "fma", "avx_vnni",
                              "avx512f", "avx512_vnni", "avx512_bf16", "avx512_vbmi",
                              "asimd", "fp16", "asimddp", "i8mm", "sve", "sve2", "sme", "sme2"]:
                        if f in raw_flags:
                            isa_flags.append(f.upper())
                    break
        except Exception:
            pass

    return cpu_name, f"{physical_cores}C/{logical_cores}T", sorted(list(set(isa_flags)))


def detect_memory():
    sys_name = platform.system()
    total_gb, avail_gb = 0.0, 0.0
    if sys_name == "Windows":
        out = run_cmd(["powershell", "-NoProfile", "-Command",
                       "$os = Get-CimInstance Win32_OperatingSystem; "
                       "@{total=[math]::Round($os.TotalVisibleMemorySize/1MB,2); free=[math]::Round($os.FreePhysicalMemory/1MB,2)} | ConvertTo-Json -Compress"])
        if out:
            try:
                d = json.loads(out)
                total_gb, avail_gb = float(d["total"]), float(d["free"])
            except Exception:
                pass
    elif sys_name == "Darwin":
        mem_bytes = run_cmd(["sysctl", "-n", "hw.memsize"])
        if mem_bytes.isdigit():
            total_gb = round(int(mem_bytes) / (1024 ** 3), 2)
            avail_gb = round(total_gb * 0.75, 2)
    elif sys_name == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                info = f.read()
            mt = re.search(r"MemTotal:\s+(\d+)\s+kB", info)
            ma = re.search(r"MemAvailable:\s+(\d+)\s+kB", info)
            if mt:
                total_gb = round(int(mt.group(1)) / (1024 ** 2), 2)
            if ma:
                avail_gb = round(int(ma.group(1)) / (1024 ** 2), 2)
        except Exception:
            pass
    return total_gb, avail_gb


def detect_gpu_and_npu():
    gpus = []
    npus = []
    sys_name = platform.system()

    if shutil.which("nvidia-smi"):
        out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,power.draw",
                       "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "vendor": "NVIDIA",
                    "name": parts[0],
                    "vram_total_gb": round(float(parts[1]) / 1024.0, 2),
                    "vram_free_gb": round(float(parts[2]) / 1024.0, 2),
                    "driver": parts[3],
                    "idle_power_w": float(parts[4]) if len(parts) >= 5 and parts[4].replace(".", "", 1).isdigit() else None,
                    "backends": ["CUDA", "Vulkan"],
                })

    if sys_name == "Windows":
        out = run_cmd(["powershell", "-NoProfile", "-Command",
                       "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"])
        for line in out.splitlines():
            name = line.strip()
            if name and not any(g["name"] in name or name in g["name"] for g in gpus):
                if any(k in name for k in ["Arc", "Intel", "Radeon", "AMD", "Adreno"]):
                    gpus.append({
                        "vendor": "Intel" if "Intel" in name or "Arc" in name else "AMD",
                        "name": name,
                        "vram_total_gb": None,
                        "backends": ["OpenVINO", "Vulkan"],
                    })
        pnp = run_cmd(["powershell", "-NoProfile", "-Command",
                       "Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match 'AI Boost|NPU|Hexagon' } | Select-Object -ExpandProperty FriendlyName"])
        for line in pnp.splitlines():
            if line.strip():
                npus.append({"name": line.strip(), "sdk": "OpenVINO / LevelZero"})

    elif sys_name == "Darwin":
        cpu_brand = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if "Apple" in cpu_brand:
            gpus.append({"vendor": "Apple", "name": f"{cpu_brand} Integrated GPU", "vram_total_gb": None, "backends": ["Metal", "CoreML"]})
            npus.append({"name": f"{cpu_brand} 16-core Neural Engine (ANE)", "sdk": "CoreML"})

    return gpus, npus


def recommend_plan(total_ram_gb, avail_ram_gb, gpus):
    max_vram = max((g.get("vram_total_gb") or 0.0 for g in gpus), default=0.0)
    models = []
    for name, size_gb, min_ram, min_vram in [
        ("Qwen3-0.6B-Q4_0", 0.42, 2.0, 1.0),
        ("Qwen3-4B-Instruct-2507-Q4_0", 2.38, 4.5, 3.5),
        ("Qwen3.5-4B-Q4_0", 2.58, 4.8, 3.8),
        ("Qwen3-8B-Q4_0", 4.70, 7.5, 6.2),
        ("Qwen3.5-9B-Q4_0", 5.38, 8.5, 7.2),
    ]:
        cpu_safe = avail_ram_gb >= min_ram or total_ram_gb >= (min_ram + 3.0)
        gpu_safe = max_vram >= min_vram if max_vram > 0 else cpu_safe
        gpu_overflow_risk = (max_vram > 0) and (size_gb < max_vram < min_vram + 1.2)
        models.append({
            "model": name,
            "weights_gb": size_gb,
            "cpu_runnable": cpu_safe,
            "gpu_full_offload_safe": gpu_safe,
            "vram_overflow_cliff_watch": gpu_overflow_risk,
        })
    return models


def main():
    cpu_name, cores_threads, isa_flags = detect_cpu_and_isa()
    total_ram_gb, avail_ram_gb = detect_memory()
    gpus, npus = detect_gpu_and_npu()
    profile = {
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_name": cpu_name,
        "cpu_cores_threads": cores_threads,
        "cpu_isa_flags": isa_flags,
        "ram_total_gb": total_ram_gb,
        "ram_available_gb": avail_ram_gb,
        "gpus": gpus,
        "npus": npus,
        "recommended_llm_matrix": recommend_plan(total_ram_gb, avail_ram_gb, gpus),
    }
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
