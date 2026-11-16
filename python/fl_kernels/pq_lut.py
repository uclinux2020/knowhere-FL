# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Build one PQ LUT per query/subspace, then reuse it across candidate tiles."""
import no_torch
no_torch.install()
import triton
import triton.language as tl
from retrieval_triton import vector_load


@triton.jit
def pq_lookup(CB, XQ, LUT, DIM: tl.constexpr, M: tl.constexpr,
              KSUB: tl.constexpr, DSUB: tl.constexpr, METRIC: tl.constexpr,
              BD: tl.constexpr, BK: tl.constexpr, USE_TLE: tl.constexpr):
    query = tl.program_id(0)
    sub = tl.program_id(1)
    code = tl.arange(0, BK)
    d = tl.arange(0, BD)
    q = vector_load(XQ + query * DIM + sub * DSUB + d, d < DSUB, 0, USE_TLE)
    b = tl.load(CB + (sub * KSUB + code[:, None]) * DSUB + d[None, :],
                (code[:, None] < KSUB) & (d[None, :] < DSUB), other=0)
    if METRIC == 0:
        diff = b - q[None, :]
        score = tl.sum(diff * diff, 1)
    else:
        score = tl.sum(b * q[None, :], 1)
    tl.store(LUT + (query * M + sub) * KSUB + code, score, code < KSUB)


@triton.jit
def pq_adc(LUT, CODES, CAND, OFFSETS, SCORES, IDS, NB, WIDTH,
           M: tl.constexpr, KSUB: tl.constexpr, METRIC: tl.constexpr,
           BLOCK: tl.constexpr):
    query = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(OFFSETS + query)
    end = tl.load(OFFSETS + query + 1)
    valid = (col < WIDTH) & (begin + col < end)
    ids = tl.load(CAND + begin + col, valid, other=-1)
    valid = valid & (ids >= 0) & (ids < NB)
    score = tl.full((BLOCK,), 0, tl.float32)
    for sub in range(M):
        code = tl.load(CODES + ids * M + sub, valid, other=0).to(tl.int64)
        valid = valid & (code < KSUB)
        value = tl.load(LUT + (query * M + sub) * KSUB + code, valid, other=0)
        score += value
    worst: tl.constexpr = float('inf') if METRIC == 0 else -float('inf')
    tl.store(SCORES + query * WIDTH + col, tl.where(valid, score, worst), col < WIDTH)
    tl.store(IDS + query * WIDTH + col, tl.where(valid, ids, -1), col < WIDTH)
