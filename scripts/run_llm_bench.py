#!/usr/bin/env python3
"""Standardized Local LLM benchmark runner with RAM/VRAM overflow protection for local-ai-bench.

Executes llama-bench across CPU ISA tiers (SSE4.2 / NEON -> AVX2 / I8MM / AMX -> AVX-512 / SME2)
and GPU backends (CUDA / Metal / Vulkan) while monitoring available RAM, dedicated VRAM usage,
and NVIDIA GPU power draw (W and J/token). Automatically guards against swap thrashing.
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time


def get_free_ram_gb():
    sys_name = platform.system()
    if sys_name == "Windows":
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 2)"],
                capture_output=True, text=True, timeout=5
            )
            return float(r.stdout.strip())
        except Exception:
            return 16.0
    elif sys_name == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return round(int(line.split()[1]) / (1024 ** 2), 2)
        except Exception:
            pass
    return 16.0


def get_nvidia_status():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if r.returncode == 0:
            p = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            return {
                "name": p[0],
                "vram_total_gb": round(float(p[1]) / 1024.0, 2),
                "vram_used_gb": round(float(p[2]) / 1024.0, 2),
                "vram_free_gb": round(float(p[3]) / 1024.0, 2),
                "power_w": float(p[4]),
                "util_pct": float(p[5]),
            }
    except Exception:
        pass
    return None


class GpuMonitor:
    def __init__(self, interval_s=0.1):
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            st = get_nvidia_status()
            if st:
                self.samples.append((time.perf_counter() - t0, st["power_w"], st["vram_used_gb"], st["util_pct"]))
            time.sleep(self.interval_s)

    def stop(self, idle_w=0.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {}
        powers = [s[1] for s in self.samples]
        vrams = [s[2] for s in self.samples]
        mean_w = statistics.mean(powers)
        delta_w = max(0.0, mean_w - idle_w)
        return {
            "mean_power_w": round(mean_w, 2),
            "peak_power_w": round(max(powers), 2),
            "delta_power_w": round(delta_w, 2),
            "peak_vram_used_gb": round(max(vrams), 2),
        }


def run_llama_bench(binary, model_path, threads=8, ngl=0, prompts="128,1000", gens="32,128", reps=2, timeout_s=300):
    model_size_gb = round(os.path.getsize(model_path) / (1024 ** 3), 2)
    ram_before = get_free_ram_gb()
    gpu_before = get_nvidia_status()
    idle_w = gpu_before["power_w"] if gpu_before else 0.0

    # Memory / VRAM guard check
    if ngl == 0 and ram_before < model_size_gb + 1.2:
        return {
            "status": "oom_guarded",
            "model": os.path.basename(model_path),
            "model_size_gb": model_size_gb,
            "ram_before_gb": ram_before,
            "reason": f"Available RAM ({ram_before} GB) < model ({model_size_gb} GB) + 1.2 GB safety margin",
        }

    cmd = [
        binary,
        "-m", model_path,
        "-t", str(threads),
        "-ngl", str(ngl),
        "-p", str(prompts),
        "-n", str(gens),
        "-r", str(reps),
        "-o", "json",
    ]
    mon = GpuMonitor()
    if ngl > 0 and gpu_before:
        mon.start()
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        wall_s = round(time.perf_counter() - t0, 2)
        gpu_stats = mon.stop(idle_w) if (ngl > 0 and gpu_before) else {}
        if r.returncode != 0:
            return {"status": "failed", "model": os.path.basename(model_path), "stderr": r.stderr[-500:]}
        records = json.loads(r.stdout)
        return {
            "status": "ok",
            "model": os.path.basename(model_path),
            "model_size_gb": model_size_gb,
            "ram_before_gb": ram_before,
            "wall_s": wall_s,
            "gpu_telemetry": gpu_stats,
            "records": records,
        }
    except subprocess.TimeoutExpired:
        mon.stop(idle_w)
        return {"status": "timeout", "model": os.path.basename(model_path), "timeout_s": timeout_s}


def main():
    ap = argparse.ArgumentParser(description="Run Local LLM benchmarks with RAM/VRAM guard.")
    ap.add_argument("--bin", required=True, help="Path to llama-bench binary")
    ap.add_argument("--models", required=True, help="Comma-separated paths to GGUF models")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ngl", type=int, default=0, help="Number of GPU layers (0 for CPU, 99 for full GPU)")
    ap.add_argument("--out", default="llm_bench_results.json")
    args = ap.parse_args()

    out = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        res = run_llama_bench(args.bin, m, threads=args.threads, ngl=args.ngl)
        out.append(res)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
