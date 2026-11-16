# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""IEEE matrix distances with exact local top-1 before global storage."""
import no_torch
no_torch.install()
import triton
import triton.language as tl
from retrieval_triton import vector_load


@triton.jit
def matrix_compact(XB, XQ, OUT_SCORES, OUT_IDS, NB, WIDTH, TILES, NQ,
                   DIM: tl.constexpr, K: tl.constexpr, METRIC: tl.constexpr,
                   BLOCK: tl.constexpr, BD: tl.constexpr, QPB: tl.constexpr,
                   USE_TLE: tl.constexpr):
    rows = tl.program_id(0) * QPB + tl.arange(0, QPB)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, BD)
    anchor = tl.load(XQ + d, d < DIM, other=0)
    shift = (METRIC == 0) & (tl.max(tl.abs(anchor), 0) > 1024)
    if shift:
        q = vector_load(XQ + rows[:, None] * DIM + d[None, :],
                        (rows[:, None] < NQ) & (d[None, :] < DIM), 0, USE_TLE) - anchor[None, :]
        b = tl.load(XB + d[:, None] * NB + cols[None, :],
                    (cols[None, :] < WIDTH) & (cols[None, :] < NB) & (d[:, None] < DIM), other=0) - anchor[:, None]
    else:
        q = vector_load(XQ + rows[:, None] * DIM + d[None, :],
                        (rows[:, None] < NQ) & (d[None, :] < DIM), 0, USE_TLE)
        b = tl.load(XB + d[:, None] * NB + cols[None, :],
                    (cols[None, :] < WIDTH) & (cols[None, :] < NB) & (d[:, None] < DIM), other=0)
    cross = tl.dot(q, b, input_precision='ieee')
    if METRIC == 0:
        scores = tl.maximum(tl.sum(q * q, 1)[:, None] + tl.sum(b * b, 0)[None, :] - 2 * cross, 0)
    else:
        scores = cross
    valid = (cols < WIDTH) & (cols < NB)
    ids = tl.broadcast_to(tl.where(valid, cols.to(tl.int64), -1)[None, :], (QPB, BLOCK))
    worst: tl.constexpr = float('inf') if METRIC == 0 else -float('inf')
    for rank in range(K):
        candidates = tl.where(ids >= 0, scores, worst)
        if METRIC == 0:
            best = tl.min(candidates, 1)
        else:
            best = tl.max(candidates, 1)
        winner = tl.min(tl.where((ids >= 0) & (scores == best[:, None]), ids, 0x7fffffffffffffff), 1)
        found = winner != 0x7fffffffffffffff
        out = (rows * TILES + tile) * K + rank
        tl.store(OUT_SCORES + out, tl.where(found, best, worst), rows < NQ)
        tl.store(OUT_IDS + out, tl.where(found, winner, -1), rows < NQ)
        ids = tl.where(ids == winner[:, None], -1, ids)
