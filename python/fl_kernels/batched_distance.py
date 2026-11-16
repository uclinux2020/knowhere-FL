# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Reuse a resident base tile for several queries in one Triton/TLE program."""
import no_torch
no_torch.install()
import triton
import triton.language as tl
from retrieval_triton import vector_load


@triton.jit
def batched_distance_scores(XB, XQ, SCORES, IDS, NB, WIDTH, NQ,
                            DIM: tl.constexpr, METRIC: tl.constexpr,
                            BLOCK: tl.constexpr, BD: tl.constexpr,
                            QPB: tl.constexpr, USE_TLE: tl.constexpr):
    group = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, BD)
    valid = (col < WIDTH) & (col < NB)
    # Large base tiles use normal loads: forced async promotion inflated SMEM
    # to 66 KiB on ZW810E. TLE remains useful for the small query vector.
    b = tl.load(XB + col[:, None] * DIM + d[None, :],
                valid[:, None] & (d[None, :] < DIM), other=0)
    worst: tl.constexpr = float('inf') if METRIC == 0 else -float('inf')
    for p in tl.static_range(QPB):
        query = group * QPB + p
        q = vector_load(XQ + query * DIM + d, (query < NQ) & (d < DIM), 0, USE_TLE)
        if METRIC == 0:
            diff = b - q[None, :]
            score = tl.sum(diff * diff, 1)
        else:
            score = tl.sum(b * q[None, :], 1)
        tl.store(SCORES + query * WIDTH + col, tl.where(valid, score, worst),
                 (query < NQ) & (col < WIDTH))
        tl.store(IDS + query * WIDTH + col, tl.where(valid, col.to(tl.int64), -1),
                 (query < NQ) & (col < WIDTH))
