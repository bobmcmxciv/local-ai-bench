#!/usr/bin/env python3
"""Zero-dependency validator and Markdown summary generator for local-ai-bench results."""
import json
import pathlib
import sys


REQUIRED_TOP = {"schema_version", "generated_at", "device", "asr_benchmarks", "llm_benchmarks"}
REQUIRED_DEV = {"id", "name", "soc_or_cpu", "os", "ram_gb", "compute_units"}


def validate(data):
    errors = []
    missing = REQUIRED_TOP - set(data.keys())
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
    if data.get("schema_version") != "1.0.0":
        errors.append(f"Expected schema_version '1.0.0', got {data.get('schema_version')!r}")
    dev = data.get("device", {})
    missing_dev = REQUIRED_DEV - set(dev.keys())
    if missing_dev:
        errors.append(f"Missing device keys: {sorted(missing_dev)}")
    for i, row in enumerate(data.get("asr_benchmarks", [])):
        for k in ("unit", "tier", "route_label", "runtime", "precision", "status"):
            if k not in row:
                errors.append(f"asr_benchmarks[{i}] missing required key '{k}'")
    for i, row in enumerate(data.get("llm_benchmarks", [])):
        for k in ("model", "quantization", "unit", "tier", "route_label", "status"):
            if k not in row:
                errors.append(f"llm_benchmarks[{i}] missing required key '{k}'")
    return errors


def print_summary(data):
    dev = data["device"]
    print(f"# Benchmark Report: {dev['name']} ({dev['soc_or_cpu']})")
    print(f"- **OS**: {dev['os']} | **RAM**: {dev['ram_gb']} GB | **Compute Units**: {', '.join(dev['compute_units'])}")
    if dev.get("gpu_model"):
        print(f"- **GPU**: {dev['gpu_model']} ({dev.get('gpu_vram_gb') or '?'} GB VRAM)")
    if dev.get("npu_model"):
        print(f"- **NPU**: {dev['npu_model']}")
    print("\n## 1. Speech-to-Text (SenseVoice Small) Matrix\n")
    print("| Unit | Tier | Route | 10s Win (ms) | 6.3m Call (s) | 49.3m Meeting (s) | Realtime (×) | Eff. TFLOPS | Util (%) | Energy (J) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in data["asr_benchmarks"]:
        fmt = lambda v, d=1: f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
        print(f"| {r['unit']} | `{r['tier']}` | {r['route_label']} | {fmt(r.get('window_10s_ms'), 1)} | "
              f"{fmt(r.get('call_378s_s'), 2)} | {fmt(r.get('meeting_2957s_s'), 2)} | {fmt(r.get('realtime_factor_x'), 1)}× | "
              f"{fmt(r.get('effective_tflops'), 3)} | {fmt(r.get('hardware_utilization_pct'), 1)}% | {fmt(r.get('total_energy_j'), 1)} |")

    print("\n## 2. Local LLM (0.6B – 9B) Matrix\n")
    print("| Model | Unit | Tier | Route / ISA | pp128 (tok/s) | pp1000 (tok/s) | tg32 / Decode (tok/s) | VRAM / RAM | Energy (J/tok) | Status |")
    print("|---|---|---|---|---:|---:|---:|---|---:|---|")
    for r in data["llm_benchmarks"]:
        fmt = lambda v, d=1: f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"
        mem_info = f"VRAM {r['vram_used_gb']:.1f}/{r['vram_total_gb']:.1f} GB" if r.get("vram_used_gb") else (
            f"RAM free {r['mem_avail_before_gb']:.1f}->{r['mem_avail_after_gb']:.1f} GB" if r.get("mem_avail_after_gb") else "—"
        )
        dec = r.get("decode_tps") or r.get("tg32_tps")
        print(f"| {r['model']} ({r['quantization']}) | {r['unit']} | `{r['tier']}` | {r['route_label']} | "
              f"{fmt(r.get('pp128_tps'), 1)} | {fmt(r.get('pp1000_tps') or r.get('prefill_tps'), 1)} | "
              f"{fmt(dec, 1)} | {mem_info} | {fmt(r.get('decode_energy_j_per_tok'), 3)} | {r['status']} |")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_result.py <result-file.json>", file=sys.stderr)
        sys.exit(1)
    path = pathlib.Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate(data)
    if errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        sys.exit(1)
    print(f"VALIDATED OK: {path.name}\n")
    print_summary(data)


if __name__ == "__main__":
    main()
