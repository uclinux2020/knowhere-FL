# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Request-owned base packing for coalesced FP32 matrix distance loads."""
import no_torch
no_torch.install()
import triton
import triton.language as tl


@triton.jit
def transpose_base(XB, PACKED, NB, DIM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    dims = tl.program_id(1) * BD + tl.arange(0, BD)
    values = tl.load(XB + rows[:, None] * DIM + dims[None, :],
                     (rows[:, None] < NB) & (dims[None, :] < DIM), other=0)
    tl.store(PACKED + dims[None, :] * NB + rows[:, None], values,
             (rows[:, None] < NB) & (dims[None, :] < DIM))
