# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Compile a shape-specialized PPU retrieval bundle without importing PyTorch.

The PPU FlagTree plugin currently uses a CUDA-named GPUTarget internally; that
is a compiler target identifier, not a CUDA-C implementation or CUDA runtime.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--metric', choices=['L2', 'IP'], default='L2')
    ap.add_argument('--m', type=int, default=0)
    ap.add_argument('--ksub', type=int, default=256)
    ap.add_argument('--block', type=int, default=128)
    ap.add_argument('--select-block', type=int, default=1024)
    ap.add_argument('--num-warps', type=int, choices=[2, 4, 8], default=4)
    ap.add_argument('--implicit-ids', action='store_true', help='Eliminate brute-force score-table ID storage')
    ap.add_argument('--pq-lut', action='store_true', help='Reuse per-query PQ lookup tables')
    ap.add_argument('--matrix', action='store_true', help='Opt-in IEEE FP32 matrix distance configuration')
    ap.add_argument('--queries-per-block', type=int, choices=[1, 2, 4, 16, 32], default=4)
    ap.add_argument('--fused', action='store_true', help='Experimental distance/local-topk fusion')
    ap.add_argument('--fused-block', type=int, default=256)
    ap.add_argument('--tle', action='store_true', help='Use Triton-TLE asynchronous vector loads')
    args = ap.parse_args()
    if args.fused_block <= 0 or args.fused_block & (args.fused_block - 1):
        ap.error('fused-block must be a positive power of two')
    if args.dim <= 0 or not 0 < args.k <= args.select_block // 2:
        ap.error('dim > 0 and 0 < k < select-block required')
    if args.block <= 0 or args.block & (args.block - 1) or args.select_block & (args.select_block - 1):
        ap.error('block sizes must be powers of two')
    if args.m and (args.m < 0 or args.dim % args.m or not 1 <= args.ksub <= 256):
        ap.error('PQ requires m dividing dim and ksub <= 256')
    os.environ['FLAGTREE_BACKEND'] = 'ppu'
    import no_torch
    no_torch.install()
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from retrieval_triton import distance_scores, pq_scores, select_topk
    from fused_retrieval import fused_distance_topk
    from batched_distance import batched_distance_scores
    if args.matrix and (args.queries_per_block < 16 or args.dim < 16 or args.block < 16 or args.m != 0 or args.fused):
        ap.error('matrix requires queries-per-block >= 16, dim/block >= 16, m=0 and no fused flag')
    if not args.matrix and args.queries_per_block > 4:
        ap.error('direct distance configuration supports queries-per-block <= 4')
    from matrix_distance import matrix_distance_scores
    from pq_lut import pq_lookup, pq_adc
    if args.pq_lut and args.m == 0:
        ap.error('pq-lut requires m > 0')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metric = int(args.metric == 'IP')
    entries = {}

    def emit(name, fn, signature, constants):
        source = ASTSource(fn, signature, constants)
        kernel = triton.compile(source, target=GPUTarget('cuda', 80, 32),
                                options={'num_warps': args.num_warps, 'num_stages': 3})
        binary = kernel.asm.get('hgbin')
        if not isinstance(binary, bytes):
            raise RuntimeError('FlagTree did not emit PPU hgbin; refusing a non-PPU artifact')
        path = out / (name + '.hgbin')
        path.write_bytes(binary)
        meta = kernel.metadata
        entries[name] = {'file': path.name, 'symbol': meta.name,
                         'shared': meta.shared, 'num_warps': meta.num_warps,
                         'num_ctas': meta.num_ctas,
                         'global_scratch_size': getattr(meta, 'global_scratch_size', 0),
                         'profile_scratch_size': getattr(meta, 'profile_scratch_size', 0)}

    dist_sig = {name: '*fp32' for name in ('XB', 'XQ', 'SCORES')}
    dist_sig.update({name: '*i64' for name in ('CAND', 'OFFSETS', 'IDS')})
    dist_sig.update({'NB': 'i64', 'WIDTH': 'i64', 'BASE': 'i64'})
    for ivf in (False, True):
        emit('ivf_flat' if ivf else 'distance', distance_scores, dist_sig,
             {'DIM': args.dim, 'METRIC': metric, 'IVF': ivf,
              'BLOCK': args.block, 'BD': triton.next_power_of_2(args.dim), 'USE_TLE': args.tle})
    sel_sig = {'SCORES': '*fp32', 'IDS': '*i64', 'OUT_SCORES': '*fp32',
               'OUT_IDS': '*i64', 'WIDTH': 'i64', 'TILES': 'i64'}
    emit('select', select_topk, sel_sig,
         {'K': args.k, 'METRIC': metric, 'BLOCK': args.select_block})
    if args.implicit_ids:
        if args.m != 0 or args.fused:
            ap.error('implicit-ids requires non-PQ, non-fused distance')
        emit('select_implicit', select_topk, sel_sig,
             {'K': args.k, 'METRIC': metric, 'BLOCK': args.select_block, 'IMPLICIT_IDS': True})
    if args.m:
        sig = {'CB': '*fp32', 'CODES': '*u8', 'XQ': '*fp32', 'CAND': '*i64',
               'OFFSETS': '*i64', 'SCORES': '*fp32', 'IDS': '*i64', 'NB': 'i64', 'WIDTH': 'i64', 'BASE': 'i64'}
        emit('pq', pq_scores, sig,
             {'DIM': args.dim, 'M': args.m, 'KSUB': args.ksub, 'DSUB': args.dim // args.m,
              'METRIC': metric, 'BLOCK': args.block, 'BD': triton.next_power_of_2(args.dim // args.m),
              'USE_TLE': args.tle})
    if args.pq_lut:
        emit('pq_lut', pq_lookup, {'CB': '*fp32', 'XQ': '*fp32', 'LUT': '*fp32'},
             {'DIM': args.dim, 'M': args.m, 'KSUB': args.ksub, 'DSUB': args.dim // args.m,
              'METRIC': metric, 'BD': triton.next_power_of_2(args.dim // args.m),
              'BK': triton.next_power_of_2(args.ksub), 'USE_TLE': args.tle})
        emit('pq_adc', pq_adc,
             {'LUT': '*fp32', 'CODES': '*u8', 'CAND': '*i64', 'OFFSETS': '*i64',
              'SCORES': '*fp32', 'IDS': '*i64', 'NB': 'i64', 'WIDTH': 'i64'},
             {'M': args.m, 'KSUB': args.ksub, 'METRIC': metric, 'BLOCK': args.block})
    if args.m == 0:
        emit('batched_distance', matrix_distance_scores if args.matrix else batched_distance_scores,
             {'XB': '*fp32', 'XQ': '*fp32', 'SCORES': '*fp32', 'IDS': '*i64', 'NB': 'i64', 'WIDTH': 'i64', 'NQ': 'i64'},
             {'DIM': args.dim, 'METRIC': metric, 'BLOCK': args.block, 'BD': triton.next_power_of_2(args.dim),
              'QPB': args.queries_per_block, 'USE_TLE': args.tle, 'IMPLICIT_IDS': args.implicit_ids})
    if args.fused and args.m == 0 and args.k <= args.fused_block:
        fused_sig = {'XB': '*fp32', 'XQ': '*fp32', 'CAND': '*i64', 'OFFSETS': '*i64',
                     'OUT_SCORES': '*fp32', 'OUT_IDS': '*i64', 'NB': 'i64', 'WIDTH': 'i64', 'TILES': 'i64'}
        for ivf in (False, True):
            emit('fused_ivf' if ivf else 'fused_distance', fused_distance_topk, fused_sig,
                 {'DIM': args.dim, 'K': args.k, 'METRIC': metric, 'IVF': ivf, 'BLOCK': args.fused_block,
                  'BD': triton.next_power_of_2(args.dim), 'USE_TLE': args.tle})
    manifest = {'schema': 1, 'backend': 'flagtree-ppu', 'implementation': 'triton-tle' if args.tle else 'triton',
                'dim': args.dim, 'k': args.k, 'metric': metric, 'm': args.m,
                'ksub': args.ksub, 'block': args.block, 'select_block': args.select_block, 'fused_block': args.fused_block,
                'kernels': entries, 'queries_per_block': args.queries_per_block,
                'implicit_ids': args.implicit_ids,
                'distance_implementation': 'matrix-ieee-fp32' if args.matrix else 'direct-fp32'}
    # Publish the manifest last so partially built bundles cannot be used.
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if 'torch' in __import__('sys').modules:
        raise RuntimeError('AOT compilation unexpectedly imported PyTorch')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
