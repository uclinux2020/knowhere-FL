# Copyright (C) 2026 Knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
#
# Phase 1.2 micro-benchmark: full Lloyd kmeans composed purely from FlagOS
# operator-library primitives (design principle D1). No new Triton kernel:
#
#   assignment  = argmin_c ||x - c||^2  = argmin( ||x||^2 - 2 x@C.T + ||c||^2 )
#                 -> mm + sum + argmin          (no atomics, PPU-safe)
#   update      = mean of assigned vectors per cluster
#                 -> index_add (ATOMICS) + div  <-- the op this bench audits
#
# Purpose: decide whether the device-side centroid UPDATE is viable on PPU.
# The C++ FL path (FlIvfFlatIndexNode) already does device assignment +
# CPU update; this script measures both hybrid and full-device Lloyd and
# reports whether index_add atomics are fast enough to move the update
# on-device too (Phase 1.2 exit criteria, plan section 5.2).
#
# Usage (remote PPU dev box):
#   python kmeans_composition.py --nb 1000000 --dim 128 --nlist 1024
#
# Outputs: per-stage latency (assign / update), hybrid vs full-device
# comparison, cluster-quality check vs faiss.Kmeans, JSON report for the
# Phase 1.2 算子支持度清单.

import argparse
import json
import time

import torch


def enable_flaggems():
    """Route ATen ops to FlagGems (FlagOS operator library)."""
    try:
        import flaggems  # noqa: F401
        from flaggems import enable  # type: ignore

        enable()
        print("[fl] FlagGems enabled: ATen ops -> FlagGems Triton kernels")
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"[fl] FlagGems not enabled ({e}); running on stock backend")


def assign(x: torch.Tensor, cent: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid routing via mm+sum+argmin (no atomics)."""
    x_sq = (x * x).sum(dim=1, keepdim=True)            # (nb, 1)
    c_sq = (cent * cent).sum(dim=1)                    # (nlist,)
    cross = torch.mm(x, cent.t())                      # (nb, nlist)
    return (x_sq + c_sq.unsqueeze(0) - 2.0 * cross).argmin(dim=1)


def update_on_device(x: torch.Tensor, owner: torch.Tensor, nlist: int) -> torch.Tensor:
    """Centroid recompute via index_add (ATOMICS under audit)."""
    dim = x.shape[1]
    sums = torch.zeros(nlist, dim, device=x.device, dtype=torch.float32)
    counts = torch.zeros(nlist, device=x.device, dtype=torch.float32)
    ones = torch.ones_like(owner, dtype=torch.float32)
    sums.index_add_(0, owner, x)                       # <-- atomic scatter-add
    counts.index_add_(0, owner, ones)
    counts.clamp_(min=1.0)                             # empty clusters keep old value
    return sums / counts.unsqueeze(1)


def update_on_cpu(x: torch.Tensor, owner: torch.Tensor, nlist: int) -> torch.Tensor:
    """CPU scatter-add update (the C++ hybrid mode's update step).

    Vectorized index_add on CPU tensors (same semantics as the C++ serial
    loop; the previous per-row Python loop was O(nb) interpreter time and
    dominated the benchmark, skewing the hybrid-vs-device verdict).
    """
    dim = x.shape[1]
    x_cpu = x.cpu().to(torch.float64)
    owner_cpu = owner.cpu()
    sums = torch.zeros(nlist, dim, dtype=torch.float64)
    counts = torch.zeros(nlist, dtype=torch.float64)
    ones = torch.ones_like(owner_cpu, dtype=torch.float64)
    sums.index_add_(0, owner_cpu, x_cpu)
    counts.index_add_(0, owner_cpu, ones)
    counts.clamp_(min=1.0)
    return (sums / counts.unsqueeze(1)).to(torch.float32).to(x.device)


def lloyd(x: torch.Tensor, nlist: int, iters: int, update: str, seed: int = 42):
    """Returns (centroids, assign_ms_p50, update_ms_p50)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)[:nlist]
    cent = x[perm.to(x.device)].clone()

    assign_times, update_times = [], []
    for _ in range(iters):
        t0 = time.perf_counter()
        owner = assign(x, cent)
        _sync(x)
        t1 = time.perf_counter()
        if update == "device":
            cent = update_on_device(x, owner, nlist)
        else:
            cent = update_on_cpu(x, owner, nlist)
        _sync(x)
        t2 = time.perf_counter()
        assign_times.append(t1 - t0)
        update_times.append(t2 - t1)

    assign_times.sort()
    update_times.sort()
    return cent, assign_times[len(assign_times) // 2], update_times[len(update_times) // 2]


def _sync(x: torch.Tensor):
    """Synchronize the active accelerator (PPU/Ascend/DCU are NOT cuda)."""
    if x.device.type != "cpu":
        torch.get_device_module(x.device.type).synchronize()


def faiss_reference(x: torch.Tensor, nlist: int, iters: int):
    import faiss

    km = faiss.Kmeans(x.shape[1], nlist, niter=iters, seed=42, verbose=False)
    km.train(x.cpu().numpy())
    return km.centroids


def mean_inertia(x: torch.Tensor, cent: torch.Tensor) -> float:
    owner = assign(x, cent)
    d = ((x - cent[owner]) ** 2).sum(dim=1)
    return float(d.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-faiss", action="store_true")
    ap.add_argument("--out", default="fl_phase1_kmeans_microbench.json")
    args = ap.parse_args()

    enable_flaggems()
    dev = torch.device(args.device)
    torch.manual_seed(42)
    x = torch.randn(args.nb, args.dim, device=dev, dtype=torch.float32)

    # Mode A: hybrid = device assign + CPU update (current C++ v1 behavior)
    cent_h, a_h, u_h = lloyd(x, args.nlist, args.iters, update="cpu")
    # Mode B: full device = device assign + device update (index_add atomics)
    cent_d, a_d, u_d = lloyd(x, args.nlist, args.iters, update="device")

    report = {
        "shape": {"nb": args.nb, "dim": args.dim, "nlist": args.nlist, "iters": args.iters},
        "device": args.device,
        "hybrid_assign_plus_cpu_update": {"assign_p50_s": a_h, "update_p50_s": u_h, "total_p50_s": a_h + u_h},
        "full_device": {"assign_p50_s": a_d, "update_p50_s": u_d, "total_p50_s": a_d + u_d},
        "atomics_verdict": {
            "index_add_update_p50_s": u_d,
            "cpu_update_p50_s": u_h,
            "device_update_faster": u_d < u_h,
            "note": "if device update is not faster, keep C++ hybrid mode "
                    "(device assign + CPU update) as the shipping default",
        },
        "quality": {"inertia_hybrid": mean_inertia(x, cent_h), "inertia_full_device": mean_inertia(x, cent_d)},
    }

    if not args.skip_faiss:
        try:
            ref = faiss_reference(x, args.nlist, args.iters)
            ref_t = torch.from_numpy(ref).to(dev)
            inertia_ref = mean_inertia(x, ref_t)
            report["quality"]["inertia_faiss_ref"] = inertia_ref
            report["quality"]["within_5pct_of_faiss"] = (
                report["quality"]["inertia_full_device"] <= inertia_ref * 1.05
            )
        except Exception as e:  # pragma: no cover
            report["quality"]["faiss_error"] = str(e)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[fl] report written to {args.out}")


if __name__ == "__main__":
    main()
