# Agent Execution Protocol (`local-ai-bench`)

This document defines the exact, reproducible benchmarking protocol for AI coding agents (Claude Code, OpenAI Codex CLI, Google Antigravity, Cursor, Windsurf, OpenCode, etc.) to evaluate a local device across **Speech-to-Text (SenseVoice Small)** and **Local LLMs (0.6B–9B)** and export a validated `results/result-<device_id>.json` file.

---

## 0. Core Principles & Safety Guardrails

1. **Test the Full Hardware Optimization Ladder**:
   Do not stop at a single default run. On every device, test as many of the 5 hardware optimization tiers as exist on the machine:
   - `T1_CPU_BASELINE`: Scalar / unaccelerated or basic SIMD (`SSE4.2` on x86, `Pure NEON` or `TFLite without XNNPACK` on ARM, `ORT FP32` without matrix coprocessors).
   - `T2_CPU_SIMD_INT8_FP16`: Vectorized CPU execution (`AVX2` / `AVX-VNNI` on x86, `ARMv8.2 FP16` / `I8MM` via XNNPACK or KleidiAI on ARM, `ORT INT8`).
   - `T3_CPU_MATRIX_ISA`: Advanced CPU matrix / wide-vector extensions (`AVX-512 VNNI + BF16` on AMD Zen 4/5, `AMX` / `ARMv9.3 SME2` via Accelerate / KleidiAI on Apple M4/M5, A19 Pro, Dimensity 9500).
   - `T4_GPU_ACCELERATED`: Dedicated or integrated GPU (`NVIDIA CUDA`, `Apple Metal / CoreML GPU`, `Intel Arc OpenVINO / Vulkan`, `Adreno / Mali Vulkan`).
   - `T5_NPU_DEDICATED`: Dedicated Neural Processing Unit (`Apple ANE via CoreML`, `Qualcomm Hexagon V79 via QNN/QAIRT`, `MediaTek MDLA via Neuron/NeuroPilot`, `Intel AI Boost via OpenVINO`).

2. **Strict Memory & VRAM Overflow Protection (No Swap Thrashing)**:
   - Before running any LLM, inspect available system RAM (`ram_available_gb`) and dedicated GPU VRAM (`vram_total_gb`).
   - **Rule**: Never force a model onto CPU if `model_size_gb + 1.2 GB > ram_available_gb`. Record `"status": "oom_guarded"` instead.
   - **VRAM Cliff Tracking**: On GPUs with 8 GB VRAM (e.g., RTX 5070 Laptop 8GB), `4B Q4_0` (`~2.5 GB`) fits 100% in VRAM (`-ngl 99`), whereas `9B Q4_0` (`~5.4 GB` weights + KV cache + compute buffers) approaches or crosses the 8 GB VRAM boundary at long contexts or high batch sizes. Always record `vram_used_gb`, `vram_total_gb`, and `overflow_detected` (`true` if layers spill to shared PCIe system RAM and cause a 3×–10× speed drop).

---

## Step 1: Clone & Probe Hardware

```bash
git clone https://github.com/bobmcmxciv/local-ai-bench.git
cd local-ai-bench
python scripts/probe_hardware.py > results/hardware-profile.json
```

Inspect `results/hardware-profile.json` to identify:
- `cpu_name`, `cpu_cores_threads`, `cpu_isa_flags`
- `ram_total_gb`, `ram_available_gb`
- `gpus` (and `vram_total_gb`)
- `npus`
- `recommended_llm_matrix`

---

## Step 2: Run Workload 1 — Speech-to-Text (`SenseVoice Small`)

SenseVoice Small (`~230M` parameters) is evaluated on two complementary metrics:
1. **Single 10-Second Window Inference (`window_10s_ms`)**:
   - Input tensor shape `[1, 167, 560]` (`167` LFR frames = `10.0s` audio, `3.504 GFLOPs` per forward pass).
   - Effective compute throughput:
     $$\text{Effective TFLOPS} = \frac{3.504}{\text{window\_10s\_ms}}$$
   - Hardware utilization:
     $$\text{Utilization (\%)} = \frac{\text{Effective TFLOPS}}{\text{Peak Hardware TFLOPS at Execution Precision}} \times 100\%$$
2. **End-to-End Audio Transcription (`call_378s_s` and `meeting_2957s_s`)**:
   - Standard test durations: **6.3-minute call (`378.3 s`)** and **49.3-minute meeting (`2956.9 s`)** at `16 kHz` mono WAV (or normalized via `realtime_factor_x = audio_duration_s / process_s_excl_load` if using user-supplied recordings of similar length).
   - Includes VAD segmentation + acoustic feature extraction (Kaldi 80-dim fbank + LFR `m=7, n=6`) + model inference + CTC greedy decoding, **excluding one-time model load/compilation time** (`model_load_s` is recorded separately).

Run the single-window & E2E benchmark:
```bash
python scripts/run_asr_bench.py \
  --fp32-model /path/to/model_ln16.onnx \
  --int8-model /path/to/model.int8.onnx \
  --threads 4,8 \
  --cuda \
  --out results/asr_raw.json
```

---

## Step 3: Run Workload 2 — Local LLMs (`0.6B` / `4B` / `8B` / `9B`)

Standard GGUF reference models (`Q4_0` quantization for cross-platform parity across CPU, GPU, KleidiAI SME2/I8MM, and AVX-512):
- `Qwen3-4B-Instruct-2507-Q4_0.gguf` (`2.38 GB`, `4.02B` params)
- `Qwen3.5-4B-Q4_0.gguf` (`2.58 GB`, `4.21B` params — hybrid Gated DeltaNet + Attention)
- `Qwen3.5-9B-Q4_0.gguf` (`5.38 GB`, `9.0B` params)

For each applicable tier on the host machine:
1. **CPU ISA Ablation** (on x86: `SSE4.2` vs `AVX2` vs `AVX-512`; on ARM: `Pure NEON` vs `KleidiAI I8MM / AMX` vs `KleidiAI SME2`):
   - Measure both short synthetic microbenchmarks (`pp128`, `tg32`) and long-context workloads (`pp1000` / ~1000-token prompt prefill, `tg128` decode).
2. **GPU Acceleration** (`CUDA`, `Metal`, or `Vulkan` via `llama-bench -ngl 99`):
   - Measure `pp128`, `pp1000`, `tg32`, `tg128` + `vram_used_gb` + GPU active power (`decode_delta_power_w` and `decode_energy_j_per_tok`).
   - If testing `9B` on an 8 GB GPU, test both `-ngl 99` (to observe whether shared memory spillover occurs) and partial offload if needed, setting `"overflow_detected": true` if PCIe shared-memory fallback degrades throughput.
3. **NPU Acceleration** (if supported via QAIRT / LiteRT-LM / ExecuTorch / Apple FoundationModels).

```bash
python scripts/run_llm_bench.py \
  --bin /path/to/llama-bench \
  --models /path/to/Qwen3-4B-Instruct-2507-Q4_0.gguf,/path/to/Qwen3.5-4B-Q4_0.gguf,/path/to/Qwen3.5-9B-Q4_0.gguf \
  --threads 8 \
  --ngl 0 \
  --out results/llm_cpu_raw.json
```

---

## Step 4: Format & Validate the Standardized JSON Output

Assemble the final JSON file at `results/result-<device_id>.json` conforming to `schema/benchmark-result.schema.json` (see `examples/sample-9950x-rtx3080.json` for a complete reference), then run:

```bash
python scripts/validate_result.py results/result-<device_id>.json
```

The validator checks all required fields against the schema and prints two ready-to-share Markdown tables (ASR Matrix & Local LLM Matrix).
