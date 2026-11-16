# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Paired real-PPU comparison; report failures rather than promise a speedup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import no_torch
no_torch.install()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--executable', required=True)
    ap.add_argument('--legacy-artifact', required=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--workspace-mb', type=int, default=1024)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    kernels = root / 'kernels'
    compiler = Path(__file__).with_name('compile_ppu.py')
    with (root / 'compile.log').open('w') as log:
        subprocess.run([sys.executable, str(compiler), '--tle', '--matrix', '--queries-per-block', '16',
                        '--block', '64', '--dim', '128', '--k', '10', '--output',
                        str(kernels / 'd128_k10_l2_m0_s0')], stdout=log, stderr=subprocess.STDOUT, check=True)
    env = dict(os.environ, KNOWHERE_FL_PPU_KERNEL_DIR=str(kernels),
               KNOWHERE_FL_PPU_WORKSPACE_MB=str(args.workspace_mb), OMP_PROC_BIND='close', OMP_PLACES='cores')
    env.pop('KNOWHERE_PPU_BENCH_SEED', None)
    env.pop('KNOWHERE_PPU_LEGACY_CUBIN', None)
    results = {}
    for nb in (100000, 500000, 1000000):
        for repeat in (1, 2):
            # Reverse the pair's order on the second pass.
            order = ('legacy', 'matrix') if repeat == 1 else ('matrix', 'legacy')
            for backend in order:
                name = f'{backend}_{nb}_r{repeat}'
                active_env = dict(env)
                if backend == 'legacy':
                    active_env['KNOWHERE_PPU_LEGACY_CUBIN'] = str(Path(args.legacy_artifact).resolve())
                command = ['taskset', '-c', '0-7', str(Path(args.executable).resolve()), str(root / (name + '.json')),
                           'bf', str(nb), '200', '128', '10', 'L2', '1', 'random', '15', '8', str(args.device)]
                with (root / (name + '.log')).open('w') as log:
                    subprocess.run(command, env=active_env, stdout=log, stderr=subprocess.STDOUT, check=True)
                results[name] = json.loads((root / (name + '.json')).read_text())
                print(name, results[name]['ppu_search']['batch_p50_s'], flush=True)
    speedups = {}
    for nb in (100000, 500000, 1000000):
        speedups[str(nb)] = [results[f'legacy_{nb}_r{r}']['ppu_search']['batch_p50_s'] /
                             results[f'matrix_{nb}_r{r}']['ppu_search']['batch_p50_s'] for r in (1, 2)]
    summary = {'results': results, 'speedups': speedups, 'workspace_mb': args.workspace_mb,
               'torch_imported': 'torch' in sys.modules, 'all_pairs_faster': all(s > 1 for v in speedups.values() for s in v)}
    (root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    if not summary['all_pairs_faster']:
        raise SystemExit('Not every measured pair exceeded the historical CUDA-C artifact')


if __name__ == '__main__':
    main()
