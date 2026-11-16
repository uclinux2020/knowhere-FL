# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Summarize verified PPU benchmark JSON; never turn a slowdown into a speedup."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.results)
    metadata = json.loads((root / 'validation.json').read_text())
    reports = {f.stem: json.loads(f.read_text()) for f in sorted(root.glob('*.json')) if f.name != 'validation.json'}
    result = {'validation': metadata, 'workloads': reports}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print('| Workload | CPU p50 ms | PPU p50 ms | PPU batch max ms | QPS | CPU/PPU | Recall |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for name, d in reports.items():
        cpu, ppu, correct = d['cpu_reference'], d['ppu_search'], d['correctness']
        print('| {} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.4f} | {:.3f} |'.format(
            name, 1000 * cpu['batch_p50_s'], 1000 * ppu['batch_p50_s'], 1000 * ppu['max_s'],
            ppu['qps'], d['cpu_over_ppu_speedup'], correct['recall_at_k_same_representation']))


if __name__ == '__main__':
    main()
