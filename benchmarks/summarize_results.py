"""Print summary tables from stored T4, L4, A100, and CPU benchmark JSONs."""

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def print_gpu_summary(filename: str) -> None:
    data = json.loads((RESULTS_DIR / filename).read_text())
    sysinfo = data["system"]
    print(
        f"\n=== {sysinfo['gpu_name']} ({sysinfo['capability']}, {sysinfo['vram_gb']}GB) "
        f"[torch={sysinfo['torch_version']}, triton={sysinfo['triton_version']}] ==="
    )
    gains, trats, vrats = [], [], []
    for ds in data["datasets"]:
        ref_k = ds.get("ref_method", "emeans_auto")
        ref = ds["methods"][ref_k]
        s_tri = ds["methods"].get("evolved_triton", {})
        s_nat = ds["methods"].get("evolved_native", {})
        best = s_tri if "te_mse" in s_tri else s_nat
        gain = 100.0 * (1.0 - best["te_mse"] / ref["te_mse"])
        trat = best["time_ms"] / ref["time_ms"]
        vrat = best["peak_delta_mb"] / ref["peak_delta_mb"]
        gains.append(gain)
        trats.append(trat)
        vrats.append(vrat)
        print(
            f"  {ds['name']:24s} | MSE {best['te_mse']:8.2f} vs {ref['te_mse']:8.2f} ({gain:+6.2f}%) | "
            f"Time {best['time_ms']:6.1f}ms ({trat:.2f}x) | VRAM {best['peak_delta_mb']:5.1f}MB ({vrat:.2f}x)"
        )
    print(
        f"  -> Aggregate: mean MSE gain={statistics.mean(gains):+.2f}%, "
        f"median time={statistics.median(trats):.3f}x, median VRAM={statistics.median(vrats):.3f}x"
    )


def print_cpu_summary() -> None:
    data = json.loads((RESULTS_DIR / "cpu_unseen_results.json").read_text())
    print("\n=== CPU 12 Unseen Datasets (5-Seed Mean) ===")
    for row in data["datasets"]:
        print(
            f"  {row['name']:22s} | Test {row['d_te_pct']:+6.2f}% | Train {row['d_tr_pct']:+6.2f}% | "
            f"Time {row['t_rat']:.3f}x e-means ({row['sk_t_rat']:.1f}x faster sklearn) | Mem {row['m_rat']:.3f}x"
        )


if __name__ == "__main__":
    for fname in ("t4_results.json", "l4_results.json", "a100_results.json"):
        print_gpu_summary(fname)
    print_cpu_summary()
