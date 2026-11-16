# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Fused distance and local top-k in the same Triton/TLE execution chain."""
import no_torch
no_torch.install()
import triton
import triton.language as tl
from retrieval_triton import vector_load


@triton.jit
def fused_distance_topk(XB, XQ, CAND, OFFSETS, OUT_SCORES, OUT_IDS,
                       NB, WIDTH, TILES, DIM: tl.constexpr, K: tl.constexpr,
                       METRIC: tl.constexpr, IVF: tl.constexpr,
                       BLOCK: tl.constexpr, BD: tl.constexpr, USE_TLE: tl.constexpr):
    query = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, BD)
    if IVF:
        begin = tl.load(OFFSETS + query)
        end = tl.load(OFFSETS + query + 1)
        valid = (col < WIDTH) & (begin + col < end)
        ids = tl.load(CAND + begin + col, valid, other=-1)
        valid = valid & (ids >= 0) & (ids < NB)
    else:
        ids = col.to(tl.int64)
        valid = (col < WIDTH) & (col < NB)
    q = vector_load(XQ + query * DIM + d, d < DIM, 0, USE_TLE)
    b = vector_load(XB + ids[:, None] * DIM + d[None, :],
                    valid[:, None] & (d[None, :] < DIM), 0, False)
    if METRIC == 0:
        diff = b - q[None, :]
        scores = tl.sum(diff * diff, 1)
    else:
        scores = tl.sum(b * q[None, :], 1)
    worst: tl.constexpr = float('inf') if METRIC == 0 else -float('inf')
    ids = tl.where(valid, ids, -1)
    for rank in range(K):
        candidate = tl.where(ids >= 0, scores, worst)
        if METRIC == 0:
            best = tl.min(candidate, 0)
        else:
            best = tl.max(candidate, 0)
        winner = tl.min(tl.where((ids >= 0) & (scores == best), ids, 0x7fffffffffffffff), 0)
        found = winner != 0x7fffffffffffffff
        out = (query * TILES + tile) * K + rank
        tl.store(OUT_SCORES + out, tl.where(found, best, worst))
        tl.store(OUT_IDS + out, tl.where(found, winner, -1))
        ids = tl.where(ids == winner, -1, ids)
