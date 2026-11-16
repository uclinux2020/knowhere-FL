# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Torch-free retrieval kernels, compiled offline by FlagTree's PPU backend.

C++ owns allocation, transfers, streams and launches. No ATen/CUDA-C operator
is used. Each score program handles one query and a bounded candidate tile.
Selection is hierarchical, avoiding a full nq * nb distance matrix.
"""

import no_torch
no_torch.install()

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def vector_load(ptr, mask, other: tl.constexpr, USE_TLE: tl.constexpr):
    if USE_TLE:
        return tle.load(ptr, mask, other=other, is_async=True)
    return tl.load(ptr, mask, other=other)


@triton.jit
def distance_scores(XB, XQ, CAND, OFFSETS, SCORES, IDS,
                    NB, DIM: tl.constexpr, WIDTH, BASE,
                    METRIC: tl.constexpr, IVF: tl.constexpr,
                    BLOCK: tl.constexpr, BD: tl.constexpr, USE_TLE: tl.constexpr = False):
    query = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    global_col = BASE + col
    d = tl.arange(0, BD)
    if IVF:
        begin = tl.load(OFFSETS + query)
        end = tl.load(OFFSETS + query + 1)
        valid = (col < WIDTH) & (begin + global_col < end)
        ids = tl.load(CAND + begin + global_col, valid, other=-1)
        valid = valid & (ids >= 0) & (ids < NB)
    else:
        ids = global_col.to(tl.int64)
        valid = (col < WIDTH) & (global_col < NB)
    q = vector_load(XQ + query * DIM + d, d < DIM, 0, USE_TLE)
    b = vector_load(XB + ids[:, None] * DIM + d[None, :],
                    valid[:, None] & (d[None, :] < DIM), 0, USE_TLE)
    if METRIC == 0:
        diff = b - q[None, :]
        scores = tl.sum(diff * diff, 1)
    else:
        scores = tl.sum(b * q[None, :], 1)
    worst: tl.constexpr = float('inf') if METRIC == 0 else -float('inf')
    tl.store(SCORES + query * WIDTH + col, tl.where(valid, scores, worst), col < WIDTH)
    tl.store(IDS + query * WIDTH + col, tl.where(valid, ids, -1), col < WIDTH)


@triton.jit
def pq_scores(CB, CODES, XQ, CAND, OFFSETS, SCORES, IDS,
              NB, DIM: tl.constexpr, WIDTH, BASE,
              M: tl.constexpr, KSUB: tl.constexpr, DSUB: tl.constexpr,
              METRIC: tl.constexpr, BLOCK: tl.constexpr, BD: tl.constexpr, USE_TLE: tl.constexpr = False):
    query = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(OFFSETS + query)
    end = tl.load(OFFSETS + query + 1)
    valid = (col < WIDTH) & (begin + BASE + col < end)
    ids = tl.load(CAND + begin + BASE + col, valid, other=-1)
    valid = valid & (ids >= 0) & (ids < NB)
    d = tl.arange(0, BD)
    score = tl.full((BLOCK,), 0, tl.float32)
    for s in range(M):
        code = tl.load(CODES + ids * M + s, valid, other=0).to(tl.int64)
        valid = valid & (code < KSUB)
        q = vector_load(XQ + query * DIM + s * DSUB + d, d < DSUB, 0, USE_TLE)
        b = vector_load(CB + (s * KSUB + code[:, None]) * DSUB + d[None, :],
                        valid[:, None] & (d[None, :] < DSUB), 0, USE_TLE)
        if METRIC == 0:
            diff = b - q[None, :]
            score += tl.sum(diff * diff, 1)
        else:
            score += tl.sum(b * q[None, :], 1)
    worst: tl.constexpr = float('inf') if METRIC == 0 else -float('inf')
    tl.store(SCORES + query * WIDTH + col, tl.where(valid, score, worst), col < WIDTH)
    tl.store(IDS + query * WIDTH + col, tl.where(valid, ids, -1), col < WIDTH)


@triton.jit
def select_topk(SCORES, IDS, OUT_SCORES, OUT_IDS,
                WIDTH, K: tl.constexpr, TILES,
                METRIC: tl.constexpr, BLOCK: tl.constexpr):
    query = tl.program_id(0)
    tile = tl.program_id(1)
    pos = tile * BLOCK + tl.arange(0, BLOCK)
    worst: tl.constexpr = float('inf') if METRIC == 0 else -float('inf')
    scores = tl.load(SCORES + query * WIDTH + pos, pos < WIDTH, other=worst)
    ids = tl.load(IDS + query * WIDTH + pos, pos < WIDTH, other=-1)
    for rank in range(K):
        eligible = ids >= 0
        candidate = tl.where(eligible, scores, worst)
        if METRIC == 0:
            best = tl.min(candidate, 0)
        else:
            best = tl.max(candidate, 0)
        # Break score ties by global ID, never by the reduction tree's order.
        winner = tl.min(tl.where(eligible & (scores == best), ids, 0x7fffffffffffffff), 0)
        found = winner != 0x7fffffffffffffff
        out = (query * TILES + tile) * K + rank
        tl.store(OUT_SCORES + out, tl.where(found, best, worst))
        tl.store(OUT_IDS + out, tl.where(found, winner, -1))
        ids = tl.where(ids == winner, -1, ids)
