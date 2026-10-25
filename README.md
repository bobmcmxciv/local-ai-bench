# `local-ai-bench` — Cross-Platform Local AI Hardware & Optimization Tier Benchmark

[![Schema v1.0.0](https://img.shields.io/badge/schema-v1.0.0-087f75)](schema/benchmark-result.schema.json)
[![Live Report](https://img.shields.io/badge/live_report-bobmcmxciv.xin%2Fnpu-406cce)](https://www.bobmcmxciv.xin/npu/)

**`local-ai-bench`** is an open, reproducible benchmark suite and agent protocol for measuring real-world **end-to-end Speech-to-Text (`SenseVoice Small`)** and **Local LLM (`0.6B – 9B`)** performance, power consumption (`W`), energy per task (`J` / `J/token`), effective compute (`TFLOPS`), and hardware utilization (`%`) across **CPU instruction sets (`SSE4.2` / `NEON` → `AVX2` / `I8MM` / `AMX` → `AVX-512` / `SME2`)**, **GPUs (`CUDA` / `Metal` / `Vulkan` / `OpenVINO`)**, and **dedicated NPUs (`Apple ANE` / `Qualcomm Hexagon` / `MediaTek MDLA` / `Intel AI Boost`)**.

- **Live Interactive Report & Master Tables**: [https://www.bobmcmxciv.xin/npu/](https://www.bobmcmxciv.xin/npu/)
- **Power & Energy Deep-Dive**: [https://www.bobmcmxciv.xin/npu/power.html](https://www.bobmcmxciv.xin/npu/power.html)

---

## One-Shot Prompt for Any AI Agent

Copy and paste the following prompt into **Claude Code**, **OpenAI Codex CLI**, **Google Antigravity**, **Cursor**, **Windsurf**, or **OpenCode** on any Mac, Windows PC, Linux workstation, or Android device to automatically run the benchmark suite and export a standardized JSON report:

```text
Please benchmark this machine's local AI performance across CPU, GPU, and NPU by following the protocol in https://github.com/bobmcmxciv/local-ai-bench :

1. Clone https://github.com/bobmcmxciv/local-ai-bench (or read AGENT_PROTOCOL.md and schema/benchmark-result.schema.json).
2. Run `python scripts/probe_hardware.py` to detect this device's CPU model, core/thread count, supported SIMD/Matrix ISA flags (SSE4.2 / AVX2 / AVX-VNNI / AVX-512 / NEON / FP16 / I8MM / AMX / SME2), total & available RAM, GPU & VRAM capacity, and NPU availability.
3. Memory & VRAM Safety Rule: Before running any workload, check available RAM and VRAM. Never run an LLM configuration that exceeds available physical RAM (keep >= 1.2 GB free headroom to avoid swap thrashing). On GPUs with limited VRAM (e.g. 8 GB), test 4B Q4_0 in full GPU offload, and if testing 8B/9B Q4_0, explicitly record whether VRAM overflow into shared system memory occurs (`overflow_detected: true`).
4. Run Workload 1 (Speech-to-Text — SenseVoice Small):
   - Measure 10-second single-window inference latency (`window_10s_ms`, input shape [1, 167, 560], 3.504 GFLOPs) and end-to-end transcription speed (`realtime_factor_x`, 6.3-min call & 49.3-min meeting equivalent) across all available hardware tiers on this machine:
     * Tier 1: CPU Baseline (FP32 / basic SIMD)
     * Tier 2: CPU Vector / INT8 / FP16 (ORT INT8, XNNPACK FP16, AVX2/AVX-VNNI)
     * Tier 3: CPU Matrix ISA (AVX-512 VNNI, Apple AMX / ARM SME2)
     * Tier 4: GPU Acceleration (CUDA, Metal/CoreML GPU, OpenVINO/Vulkan iGPU)
     * Tier 5: Dedicated NPU (CoreML ANE, Qualcomm QNN, MediaTek Neuron, Intel AI Boost)
   - Record power draw (W) and task energy (J) where hardware telemetry (nvidia-smi, powermetrics, RAPL, BatteryManager) is available.
5. Run Workload 2 (Local LLM — Qwen3-4B-Instruct-2507 Q4_0, Qwen3.5-4B Q4_0, Qwen3.5-9B Q4_0):
   - Test across available CPU ISA tiers (e.g. SSE4.2 vs AVX2 vs AVX-512, or NEON vs I8MM/AMX vs SME2) and GPU/NPU backends.
   - Measure model load time (`load_s`), prefill speed (`pp128_tps`, `pp1000_tps`), decode speed (`tg32_tps`, `tg128_tps`), RAM/VRAM footprint (`vram_used_gb`, `mem_avail_before_gb`, `mem_avail_after_gb`), and energy per token (`decode_energy_j_per_tok`).
6. Export the complete results to `results/result-<device_id>.json` matching `schema/benchmark-result.schema.json`, and run `python scripts/validate_result.py results/result-<device_id>.json` to verify schema compliance and print the Markdown summary tables.
```

---

## Repository Structure

| Path | Description |
|---|---|
| [`AGENT_PROTOCOL.md`](AGENT_PROTOCOL.md) | Step-by-step execution guide for AI agents and human benchmarkers |
| [`schema/benchmark-result.schema.json`](schema/benchmark-result.schema.json) | Standardized JSON Schema (`v1.0.0`) for cross-device ASR & LLM benchmark exports |
| [`scripts/probe_hardware.py`](scripts/probe_hardware.py) | Cross-platform hardware, CPU ISA flag, RAM/VRAM, GPU, and NPU capability detector |
| [`scripts/run_asr_bench.py`](scripts/run_asr_bench.py) | Standardized SenseVoice Small 10s window & end-to-end ASR benchmark runner with GPU power sampling |
| [`scripts/run_llm_bench.py`](scripts/run_llm_bench.py) | Standardized `llama-bench` runner with RAM/VRAM pre-flight guard and NVIDIA GPU power telemetry |
| [`scripts/validate_result.py`](scripts/validate_result.py) | Zero-dependency schema validator and Markdown comparison table generator |
| [`examples/sample-9950x-rtx3080.json`](examples/sample-9950x-rtx3080.json) | Complete validated reference dataset from our `AMD Ryzen 9 9950X + NVIDIA RTX 3080 20GB` workstation |

---

## Reference Hardware Matrix Covered in Our Live Report

| Device | SoC / Processor | CPU ISA Tiers Tested | GPU Tested | NPU Tested | RAM / VRAM |
|---|---|---|---|---|---|
| **Desktop Workstation (`ht3`)** | AMD Ryzen 9 9950X (16C/32T) | `SSE4.2` → `AVX2` → `AVX-512 VNNI/BF16` | NVIDIA RTX 3080 (20 GB, CUDA) | — | 64 GB DDR5 + 20 GB VRAM |
| **Performance Laptop (`tx`)** | AMD Ryzen 9 9955HX (16C/32T) | `SSE4.2` → `AVX2` → `AVX-512 VNNI/BF16` | NVIDIA RTX 5070 Laptop (8 GB, CUDA) | — | 64 GB DDR5 + 8 GB VRAM |
| **MacBook Pro 14"** | Apple M5 (10C CPU) | `No-AMX/SME2` → `Accelerate AMX` → `KleidiAI SME2` | Apple M5 10-core GPU (Metal / CoreML) | 16-core Neural Engine (ANE) | 16 GB Unified |
| **iPhone 17 Pro Max** | Apple A19 Pro (6C CPU) | `ORT INT8 (No CoreML)` → `CoreML CPU-only (AMX/SME2)` | A19 Pro 6-core GPU (CoreML FP16) | 16-core Neural Engine (ANE) | 12 GB Unified |
| **Xiaomi Pad 8 Pro** | Snapdragon 8 Elite (`SM8750P`) | `Pure NEON` → `XNNPACK FP32` → `XNNPACK FP16 / KleidiAI I8MM` | Adreno 830 | Hexagon V79 NPU (QNN / QAIRT) | 24 GB LPDDR5X |
| **OPPO Find X9 Pro** | Dimensity 9500 (`MT6993`) | `Pure NEON` → `XNNPACK FP16 / KleidiAI I8MM` → `ARMv9.3 SME2` | Immortalis-G925 | MediaTek MDLA 6.1 ×4 (Neuron) | 16 GB LPDDR5X |
| **Redmi Turbo 5 Max** | Dimensity 9400+ (`MT6991`) | `Pure NEON` → `XNNPACK FP32` → `XNNPACK FP16 / KleidiAI I8MM` | Immortalis-G925 | MediaTek MDLA 5.5 ×2 (Neuron / ExecuTorch) | 16 GB LPDDR5X |
| **RedmiBook Pro 14** | Intel Core Ultra X7 358H | `OpenVINO FP32` → `ORT INT8 / AVX2 + AVX-VNNI` | Intel Arc B390 iGPU (OpenVINO / Vulkan) | Intel AI Boost NPU (OpenVINO) | 32 GB LPDDR5X |
| **Ryzen Mini/Laptop (`4sq`)** | AMD Ryzen 7 5800H (8C/16T) | `SSE4.2` → `AVX2 (ORT FP32 / INT8)` | — | — | 16 GB DDR4 |
