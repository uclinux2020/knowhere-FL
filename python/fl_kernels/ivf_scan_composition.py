# Copyright (C) 2026 Knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
#
# Phase 1.3/P2 micro-benchmark: IVF candidate scan composed purely from
# FlagOS operator-library primitives (design D1). No new Triton kernel:
#
#   IVF_FLAT scan  = gather candidates (index_select) + distance (mm combo)
#                    + topk per query
#   IVF_PQ  scan   = per-query LUT (mm vs reshaped codebooks) + embedding
#                    lookup accumulation (gather + sum) + topk
#
# Purpose: measure whether the pure-composition scan is fast enough on PPU,
# or whether a fused `ivf_scan` kernel (KernelGen baseline -> FlagGems
# contribution) is warranted. Decision rule (plan section 5.2):
#   if composition reaches >= 60% of the roofline estimate OR meets the
#   QPS target, ship composition; else develop the fused op.
#
# Usage (remote PPU dev box):
#   python ivf_scan_composition.py --nb 1000000 --dim 128 --nlist 1024 --nprobe 8
#
# Outputs: per-mode latency/QPS across candidate ratios, JSON report for the
# Phase 1.3 算子支持度清单.

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


def sync(dev):
    """Synchronize the active accelerator (PPU/Ascend/DCU are NOT cuda)."""
    if dev.type != "cpu":
        torch.get_device_module(dev.type).synchronize()


def ivf_flat_scan_compose(xb, q, cand_ids, cand_offsets, k):
    """IVF_FLAT candidate scan via gather + mm + topk (all FlagGems ops).

    cand_ids: LongTensor (total_cand,), cand_offsets: LongTensor (nq+1,)
    Composed per query over its candidate slice.
    """
    nq, dim = q.shape
    outs_ids, outs_vals = [], []
    for i in range(nq):
        ids = cand_ids[cand_offsets[i]:cand_offsets[i + 1]]
        cand = xb.index_select(0, ids)                      # gather (FlagGems)
        qv = q[i:i + 1]                                     # (1, dim)
        # L2 via GEMM decomposition
        q_sq = (qv * qv).sum(dim=1, keepdim=True)
        c_sq = (cand * cand).sum(dim=1)
        cross = torch.mm(qv, cand.t())                      # mm (FlagGems)
        dist = (q_sq + c_sq.unsqueeze(0) - 2.0 * cross).clamp_(min=0.0)
        vals, top = torch.topk(dist.squeeze(0), min(k, ids.numel()), largest=False)
        outs_vals.append(vals)
        outs_ids.append(ids[top])
    return outs_vals, outs_ids


def ivf_pq_scan_compose(codebooks, codes, q, cand_ids, cand_offsets, m, ksub, dsub, k):
    """IVF_PQ ADC scan: LUT via mm, then gather+sum accumulation + topk."""
    nq = q.shape[0]
    outs_ids, outs_vals = [], []
    for i in range(nq):
        # per-query LUT: for each subspace s, distance(q_sub, codebook_s)
        # via bmm decomposition (bmm is a FlagGems batched mm)
        qv = q[i].view(m, 1, dsub)                          # (m, 1, dsub)
        cb = codebooks.view(m, ksub, dsub)
        q_sq = (qv * qv).sum(dim=2)                         # (m, 1)
        c_sq = (cb * cb).sum(dim=2)                         # (m, ksub)
        cross = torch.bmm(qv, cb.transpose(1, 2)).squeeze(1)  # (m, ksub)
        lut = (q_sq + c_sq - 2.0 * cross).clamp_(min=0.0)
        ids = cand_ids[cand_offsets[i]:cand_offsets[i + 1]]
        cand_codes = codes.index_select(0, ids)             # (nc, m) gather
        # ADC: sum over m of lut[s, code]
        dist = lut.gather(1, cand_codes.long()).sum(dim=1)   # gather (FlagGems)
        vals, top = torch.topk(dist, min(k, ids.numel()), largest=False)
        outs_vals.append(vals)
        outs_ids.append(ids[top])
    return outs_vals, outs_ids


def bench(fn, iters, dev):
    fn()  # warmup: Triton JIT compile must not pollute the first timed run
    sync(dev)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=64)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16, help="subquantizers for PQ mode")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="fl_phase1_ivf_scan_microbench.json")
    args = ap.parse_args()

    enable_flaggems()
    dev = torch.device(args.device)
    torch.manual_seed(42)

    xb = torch.randn(args.nb, args.dim, device=dev, dtype=torch.float32)
    q = torch.randn(args.nq, args.dim, device=dev, dtype=torch.float32)

    # synthetic balanced inverted lists: each list gets nb/nlist members
    list_len = args.nb // args.nlist
    cand_per_q = list_len * args.nprobe
    all_ids = torch.arange(args.nb, device=dev)
    cand_ids = all_ids.repeat(args.nq)[: args.nq * cand_per_q]
    cand_offsets = torch.arange(0, args.nq + 1, device=dev) * cand_per_q

    report = {"shape": vars(args), "device": args.device}

    # --- IVF_FLAT composition ---
    t_flat = bench(lambda: ivf_flat_scan_compose(xb, q, cand_ids, cand_offsets, args.k), args.iters, dev)
    report["ivf_flat_composition"] = {
        "p50_s": t_flat,
        "qps": args.nq / t_flat,
        "cand_per_query": cand_per_q,
        "scanned_vectors_per_s": args.nq * cand_per_q / t_flat,
    }

    # --- IVF_PQ composition ---
    if args.dim % args.m == 0:
        dsub = args.dim // args.m
        ksub = 256
        codebooks = torch.randn(args.m * ksub * dsub, device=dev, dtype=torch.float32)
        codes = torch.randint(0, ksub, (args.nb, args.m), device=dev, dtype=torch.uint8)
        t_pq = bench(
            lambda: ivf_pq_scan_compose(codebooks, codes, q, cand_ids, cand_offsets, args.m, ksub, dsub, args.k),
            args.iters, dev)
        report["ivf_pq_composition"] = {
            "p50_s": t_pq,
            "qps": args.nq / t_pq,
            "cand_per_query": cand_per_q,
        }

    report["fused_op_decision"] = {
        "rule": "if composition QPS meets the Phase 1 target (>= 3x CPU HNSW), "
                "ship composition; otherwise develop the fused ivf_scan op via KernelGen",
        "note": "compare against CPU baseline measured on the same box (faiss IndexIVFFlat)",
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[fl] report written to {args.out}")


if __name__ == "__main__":
    main()
