# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""IEEE FP32 distance tiles with conditional common-offset stabilization."""
import no_torch
no_torch.install()
import triton
import triton.language as tl
from retrieval_triton import vector_load


@triton.jit
def matrix_distance_scores(XB, XQ, SCORES, IDS, NB, WIDTH, NQ,
                           DIM: tl.constexpr, METRIC: tl.constexpr,
                           BLOCK: tl.constexpr, BD: tl.constexpr,
                           QPB: tl.constexpr, USE_TLE: tl.constexpr, IMPLICIT_IDS: tl.constexpr = False,
                           TRANSPOSED: tl.constexpr = False):
    rows = tl.program_id(0) * QPB + tl.arange(0, QPB)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, BD)
    base_offsets = d[:, None] * NB + cols[None, :] if TRANSPOSED else cols[None, :] * DIM + d[:, None]
    anchor = tl.load(XQ + d, d < DIM, other=0)
    shift = (METRIC == 0) & (tl.max(tl.abs(anchor), 0) > 1024)
    # Load inside each branch: this avoids an async-promotion dominance bug in
    # the current PPU compiler when an async-loaded tensor crosses a branch.
    if shift:
        q = vector_load(XQ + rows[:, None] * DIM + d[None, :],
                        (rows[:, None] < NQ) & (d[None, :] < DIM), 0, USE_TLE) - anchor[None, :]
        b = tl.load(XB + base_offsets,
                    (cols[None, :] < NB) & (cols[None, :] < WIDTH) & (d[:, None] < DIM), other=0) - anchor[:, None]
    else:
        q = vector_load(XQ + rows[:, None] * DIM + d[None, :],
                        (rows[:, None] < NQ) & (d[None, :] < DIM), 0, USE_TLE)
        b = tl.load(XB + base_offsets,
                    (cols[None, :] < NB) & (cols[None, :] < WIDTH) & (d[:, None] < DIM), other=0)
    cross = tl.dot(q, b, input_precision='ieee')
    if METRIC == 0:
        score = tl.maximum(tl.sum(q * q, 1)[:, None] + tl.sum(b * b, 0)[None, :] - 2 * cross, 0)
    else:
        score = cross
    mask = (rows[:, None] < NQ) & (cols[None, :] < WIDTH)
    tl.store(SCORES + rows[:, None] * WIDTH + cols[None, :], score, mask)
    if not IMPLICIT_IDS:
        tl.store(IDS + rows[:, None] * WIDTH + cols[None, :],
                 tl.broadcast_to(cols[None, :].to(tl.int64), (QPB, BLOCK)), mask)
