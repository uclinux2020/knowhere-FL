# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Sequential, paired native-PPU tile ablation; no service/database operations."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--executable', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', type=int, required=True)
    ap.add_argument('--nb', type=int, default=100000)
    ap.add_argument('--dim', type=int, default=1024)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--metric', choices=['L2', 'IP'], default='IP')
    ap.add_argument('--iterations', type=int, default=30)
    args = ap.parse_args()
    if args.device < 0 or args.nb <= 0 or args.dim <= 0 or not 0 < args.k <= args.nb or args.iterations < 3:
        ap.error('device >= 0, nb/dim > 0, 0 < k <= nb and iterations >= 3 required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.pop('KNOWHERE_FL_PPU_PROFILE', None)
    env.pop('KNOWHERE_PPU_LEGACY_CUBIN', None)
    env.pop('KNOWHERE_PPU_FULL_INPUT_E2E', None)
    env.pop('KNOWHERE_PPU_BENCH_SEED', None)
    env.update(KNOWHERE_PPU_QUERY_E2E='1', KNOWHERE_FL_PPU_WORKSPACE_MB='64',
               KNOWHERE_FL_PPU_WORKSPACE_CACHE_MB='0', OMP_NUM_THREADS='8',
               OMP_WAIT_POLICY='PASSIVE', OMP_PROC_BIND='true', OMP_PLACES='cores',
               UMD_PLATFORM_TYPE='1', HGGC_DRIVER_CANDIDATE='UMD')
    roots = [args.baseline, args.candidate]
    reports = []
    # nq=1 matches the application; nq=17 crosses query batches. CSR represents
    # already filtered candidates, not expression evaluation or filter overhead.
    for nq, fraction, distribution in [(1, .1, 'random'), (1, 1., 'random'),
                                        (17, .5, 'random'), (17, .5, 'skewed')]:
        blocks = [[], []]
        for block, variant in enumerate([0, 1, 1, 0]):
            env['KNOWHERE_FL_PPU_KERNEL_DIR'] = str(Path(roots[variant]).resolve())
            name = f'nq{nq}-f{fraction}-{distribution}-b{block}'
            path = out / (name + '.json')
            cmd = [args.executable, str(path), 'ivf', str(args.nb), str(nq), str(args.dim),
                   str(args.k), args.metric, str(fraction), distribution,
                   str(args.iterations), '8', str(args.device)]
            with (out / (name + '.log')).open('w') as log:
                subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
            report = json.loads(path.read_text())
            assert report['correctness']['ordered_id_mismatches'] == 0
            assert report['correctness']['score_mismatches'] == 0
            assert report['seed'] == 42 and report['query_request_e2e'] is True
            assert len(report['ppu_search']['samples_s']) == args.iterations
            blocks[variant].append(report['ppu_search']['batch_p50_s'] * 1000)
        baseline, candidate = [statistics.mean(x) for x in blocks]
        reports.append(dict(nq=nq, fraction=fraction, distribution=distribution,
                            baseline_block_p50_ms=blocks[0], candidate_block_p50_ms=blocks[1],
                            baseline_ms=baseline, candidate_ms=candidate, speedup=baseline / candidate))
        (out / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
