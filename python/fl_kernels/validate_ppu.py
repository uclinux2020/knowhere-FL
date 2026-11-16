# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Reproducible validation of ONE Triton/TLE -> FlagTree -> HG -> PPU path."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import no_torch
no_torch.install()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--build', required=True)
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    kernels = output / 'kernels'
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    builds = []
    for k, metric, m, ksub in [(1, 'L2', 0, 0), (10, 'L2', 0, 0), (100, 'L2', 0, 0),
                                (10, 'IP', 0, 0), (1, 'IP', 0, 0),
                                (10, 'L2', 16, 256), (10, 'L2', 16, 8), (10, 'IP', 16, 8)]:
        name = f'd128_k{k}_{metric.lower()}_m{m}_s{ksub}'
        start = time.perf_counter()
        with (logs / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(script_dir / 'compile_ppu.py'), '--tle', '--dim', '128',
                            '--k', str(k), '--metric', metric, '--m', str(m), '--ksub', str(ksub),
                            '--output', str(kernels / name)], stdout=log, stderr=subprocess.STDOUT, check=True)
        builds.append({'bundle': name, 'compile_wall_s_existing_cache': time.perf_counter() - start})
    env = dict(os.environ, KNOWHERE_FL_PPU_KERNEL_DIR=str(kernels), OMP_PROC_BIND='close', OMP_PLACES='cores')
    build = Path(args.build).resolve()
    with (logs / 'functionality.log').open('w') as log:
        subprocess.run([str(build / 'knowhere_ppu_test'), str(args.device), '128', '10', '16'],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    # CPU timing and PPU timing are sequential, never simultaneous benchmark jobs.
    workloads = [
        ('bf_10k_k10', 'bf', 10000, 16, 10, 'L2', 1, 'random'),
        ('bf_100k_k10', 'bf', 100000, 16, 10, 'L2', 1, 'random'),
        ('bf_1m_k10', 'bf', 1000000, 16, 10, 'L2', 1, 'random'),
        ('bf_100k_k1', 'bf', 100000, 16, 1, 'L2', 1, 'random'),
        ('bf_100k_k100', 'bf', 100000, 16, 100, 'L2', 1, 'random'),
        ('bf_100k_ip', 'bf', 100000, 16, 10, 'IP', 1, 'random'),
        ('bf_10k_nq1', 'bf', 10000, 1, 10, 'L2', 1, 'random'),
        ('ivf_random_50pct', 'ivf', 100000, 16, 10, 'L2', .5, 'random'),
        ('ivf_random_1pct', 'ivf', 100000, 16, 10, 'L2', .01, 'random'),
        ('ivf_clustered_1pct', 'ivf', 100000, 16, 10, 'L2', .01, 'clustered'),
        ('ivf_fewer_than_k', 'ivf', 1000, 16, 10, 'L2', .005, 'random'),
        ('ivf_empty', 'ivf', 1000, 16, 10, 'L2', 0, 'random'),
        ('pq_10k_all', 'pq', 10000, 16, 10, 'L2', 1, 'random'),
        ('assign_256_64', 'assign', 256, 64, 1, 'L2', 1, 'random'),
    ]
    failures = []
    for name, mode, nb, nq, k, metric, fraction, distribution in workloads:
        command = ['taskset', '-c', '0-7', str(build / 'knowhere_ppu_bench'), str(output / (name + '.json')),
                   mode, str(nb), str(nq), '128', str(k), metric, str(fraction), distribution, '7', '8', str(args.device)]
        with (logs / (name + '.log')).open('w') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(name)
        print(f'{name}: {"PASS" if not result.returncode else "FAIL"}', flush=True)
    metadata = {'path': 'Triton with Triton-TLE -> FlagTree AOT -> native HG -> PPU',
                'torch_imported': 'torch' in sys.modules, 'builds': builds, 'failed_workloads': failures,
                'cpu_affinity': '0-7', 'cpu_threads': 8, 'concurrency': 1, 'iterations': 7,
                'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    (output / 'validation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if failures:
        raise SystemExit('Failed workloads: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
