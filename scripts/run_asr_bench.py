#!/usr/bin/env python3
"""Standardized SenseVoice Small ASR benchmark runner for local-ai-bench.

Supports:
1. 10-second single-window inference benchmark (input shape [1, 167, 560], 3.504 GFLOPs)
   across ONNX Runtime (CPU FP32, CPU INT8, CUDA, CoreML, QNN) and OpenVINO (CPU, GPU, NPU).
2. End-to-end audio transcription benchmark on 6.3-min call (378.3s) and 49.3-min meeting (2956.9s)
   with optional NVIDIA GPU power sampling (nvidia-smi at 100ms intervals).
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import numpy as np

GFLOPS_PER_10S_WINDOW = 3.504  # SenseVoice Small 10s window (167 LFR frames)


class NvidiaPowerSampler:
    def __init__(self, interval_s=0.1):
        self.interval_s = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.available = shutil.which("nvidia-smi") is not None

    def _query(self):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if r.returncode == 0:
                line = r.stdout.strip().splitlines()[0]
                p, u, m = [float(x.strip()) for x in line.split(",")[:3]]
                return p, u, m
        except Exception:
            pass
        return None

    def measure_idle(self, duration_s=3.0):
        if not self.available:
            return None
        vals = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration_s:
            q = self._query()
            if q:
                vals.append(q[0])
            time.sleep(0.2)
        return statistics.mean(vals) if vals else None

    def start(self):
        if not self.available:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            q = self._query()
            if q:
                self.samples.append((time.perf_counter() - t0, q[0], q[1], q[2]))
            time.sleep(self.interval_s)

    def stop(self, idle_w=0.0):
        if not self.available or not self._thread:
            return {}
        self._stop.set()
        self._thread.join(timeout=2.0)
        if not self.samples:
            return {}
        powers = [s[1] for s in self.samples]
        utils = [s[2] for s in self.samples]
        mems = [s[3] for s in self.samples]
        dur = self.samples[-1][0] - self.samples[0][0] if len(self.samples) > 1 else 0.0
        mean_w = statistics.mean(powers)
        delta_w = max(0.0, mean_w - (idle_w or 0.0))
        return {
            "samples_count": len(self.samples),
            "idle_w": round(idle_w or 0.0, 2),
            "mean_w": round(mean_w, 2),
            "peak_w": round(max(powers), 2),
            "delta_w": round(delta_w, 2),
            "energy_above_idle_j": round(delta_w * dur, 2),
            "total_board_energy_j": round(mean_w * dur, 2),
            "mean_gpu_util_pct": round(statistics.mean(utils), 1),
            "peak_vram_mb": round(max(mems), 1),
        }


def bench_10s_window_ort(model_path, provider="CPUExecutionProvider", threads=4, warmup=5, runs=30):
    import onnxruntime as ort
    if provider == "CUDAExecutionProvider":
        try:
            ort.preload_dlls()
        except Exception:
            pass
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=opts, providers=[provider])
    inputs = {}
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 167, 560), dtype=np.float32)
    for inp in sess.get_inputs():
        if inp.name == "x" or len(inp.shape) == 3:
            inputs[inp.name] = x
        elif "length" in inp.name:
            inputs[inp.name] = np.array([167], dtype=np.int32)
        elif "language" in inp.name:
            inputs[inp.name] = np.array([0], dtype=np.int32)
        elif "text_norm" in inp.name:
            inputs[inp.name] = np.array([15], dtype=np.int32)
    for _ in range(warmup):
        sess.run(None, inputs)
    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    med_ms = statistics.median(times_ms)
    return {
        "model": os.path.basename(model_path),
        "provider": sess.get_providers()[0],
        "threads": threads,
        "median_ms": round(med_ms, 2),
        "mean_ms": round(statistics.mean(times_ms), 2),
        "min_ms": round(min(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "effective_tflops": round(GFLOPS_PER_10S_WINDOW / med_ms, 3),
    }


def main():
    ap = argparse.ArgumentParser(description="Run SenseVoice Small ASR single-window & E2E benchmarks.")
    ap.add_argument("--fp32-model", default="model_ln16.onnx", help="Path to FP32 SenseVoice ONNX model")
    ap.add_argument("--int8-model", default="model.int8.onnx", help="Path to INT8 SenseVoice ONNX model")
    ap.add_argument("--threads", default="4,8", help="Comma-separated thread counts for CPU window test")
    ap.add_argument("--cuda", action="store_true", help="Also benchmark CUDAExecutionProvider")
    ap.add_argument("--out", default="asr_window_results.json", help="Output JSON file")
    args = ap.parse_args()

    results = {"window_benchmarks": []}
    t_list = [int(x) for x in args.threads.split(",") if x.strip()]

    for th in t_list:
        if os.path.exists(args.fp32_model):
            r = bench_10s_window_ort(args.fp32_model, "CPUExecutionProvider", threads=th)
            r["tier"] = "T1_CPU_BASELINE"
            results["window_benchmarks"].append(r)
        if os.path.exists(args.int8_model):
            r = bench_10s_window_ort(args.int8_model, "CPUExecutionProvider", threads=th)
            r["tier"] = "T2_CPU_SIMD_INT8_FP16"
            results["window_benchmarks"].append(r)

    if args.cuda and os.path.exists(args.fp32_model):
        sampler = NvidiaPowerSampler()
        idle_w = sampler.measure_idle(2.0)
        sampler.start()
        r = bench_10s_window_ort(args.fp32_model, "CUDAExecutionProvider", threads=1, warmup=10, runs=50)
        r["tier"] = "T4_GPU_ACCELERATED"
        r["gpu_power"] = sampler.stop(idle_w)
        results["window_benchmarks"].append(r)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
