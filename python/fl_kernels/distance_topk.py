# Copyright (C) 2026 Knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
#
# Phase 0 micro-benchmark: FL brute-force distance + top-k composed purely
# from FlagOS operator-library primitives (design principle D1 - "FlagOS
# operator libraries first"). NO new Triton kernel is written here:
#
#   L2 distance matrix  = ||a||^2 + ||b||^2 - 2 * a @ b.T   (mm / norm / add)
#   IP distance matrix  = a @ b.T                           (mm)
#   top-k               = torch.topk                        (FlagGems topk)
#
# On a FlagOS-supported chip (PPU/Ascend/DCU/...), enable_flaggems() routes
# these ATen ops to FlagGems Triton implementations compiled by FlagTree.
#
# Usage (on the remote PPU dev box):
#   python distance_topk.py --nb 1000000 --nq 1024 --dim 128 --k 10 --metric L2
#
# Outputs: recall vs faiss brute force (ground truth), throughput (QPS),
# per-stage latency breakdown. Results feed the Phase 0 算子支持度清单.

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


def sync_device(dev: torch.device):
    """Synchronize the active accelerator (PPU/Ascend/DCU are NOT cuda).

    torch.get_device_module routes to the correct backend module; hardcoded
    torch.cuda.synchronize() would break or silently mistime on non-NVIDIA
    FlagOS targets.
    """
    if dev.type != "cpu":
        torch.get_device_module(dev.type).synchronize()


def l2_distance_matrix(q: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
    """L2 via GEMM decomposition: ||q||^2 + ||xb||^2 - 2 q@xb.T.

    All primitives (norm/mul/mm/add) come from FlagOS operator libraries.
    """
    q_sq = (q * q).sum(dim=1, keepdim=True)          # (nq, 1)   norm family
    b_sq = (xb * xb).sum(dim=1)                       # (nb,)     norm family
    cross = torch.mm(q, xb.t())                       # (nq, nb)  mm (FlagGems GEMM)
    return (q_sq + b_sq.unsqueeze(0) - 2.0 * cross).clamp_(min=0.0)


def ip_distance_matrix(q: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
    return torch.mm(q, xb.t())


def brute_force_topk(q, xb, k, metric):
    if metric == "L2":
        dist = l2_distance_matrix(q, xb)
        return torch.topk(dist, k, dim=1, largest=False)
    else:  # IP
        dist = ip_distance_matrix(q, xb)
        return torch.topk(dist, k, dim=1, largest=True)


def faiss_ground_truth(q, xb, k, metric):
    import faiss  # CPU faiss is the recall gold standard (design 4.6)

    dim = xb.shape[1]
    if metric == "L2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(xb.cpu().numpy())
    return index.search(q.cpu().numpy(), k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", choices=["L2", "IP"], default="L2")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="fl_phase0_microbench.json")
    args = ap.parse_args()

    enable_flaggems()
    dev = torch.device(args.device)
    torch.manual_seed(42)

    xb = torch.randn(args.nb, args.dim, device=dev, dtype=torch.float32)
    q = torch.randn(args.nq, args.dim, device=dev, dtype=torch.float32)

    # warmup + correctness
    t0 = time.perf_counter()
    vals, ids = brute_force_topk(q, xb, args.k, args.metric)
    sync_device(dev)
    t1 = time.perf_counter()

    gt_dist, gt_ids = faiss_ground_truth(q, xb, args.k, args.metric)
    ids_cpu = ids.cpu().numpy()
    recall = (ids_cpu == gt_ids).sum() / float(gt_ids.size)

    # throughput
    times = []
    for _ in range(args.iters):
        ta = time.perf_counter()
        brute_force_topk(q, xb, args.k, args.metric)
        sync_device(dev)
        times.append(time.perf_counter() - ta)
    times.sort()
    p50 = times[len(times) // 2]

    report = {
        "shape": {"nb": args.nb, "nq": args.nq, "dim": args.dim, "k": args.k, "metric": args.metric},
        "device": args.device,
        "recall@k_vs_faiss": recall,
        "first_call_s": t1 - t0,
        "p50_s": p50,
        "qps": args.nq / p50,
        "acceptance": {
            "recall_must_be_1.0": recall == 1.0,
            "note": "GPU_BRUTE_FORCE(FL) recall@k must equal faiss CPU ground truth",
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[fl] report written to {args.out}")


if __name__ == "__main__":
    main()
