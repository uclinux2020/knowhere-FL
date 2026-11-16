# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Compile and test actual PPU kernels without PyTorch or NumPy.

Run from the repository root with a FlagTree PPU Python environment. Unlike
Triton's interpreter (incompatible with this installed FlagTree build), this
runner validates the real AOT binaries through the C++ native HG test program.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import no_torch
no_torch.install()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--test-executable', required=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--pq-lut', action='store_true')
    ap.add_argument('--tle', action='store_true')
    ap.add_argument('--dim', type=int, default=5)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--m', type=int, default=1)
    ap.add_argument('--block', type=int, default=8)
    ap.add_argument('--select-block', type=int, default=32)
    args = ap.parse_args()
    root = Path(args.output).resolve()
    compiler = Path(__file__).with_name('compile_ppu.py')
    configs = [(args.k, metric, m, ksub) for metric in ('L2', 'IP')
               for m, ksub in ((0, 0), (args.m, 8))]
    configs += [(1, metric, 0, 0) for metric in ('L2', 'IP')]
    for k, metric, m, ksub in configs:
        bundle = root / f'd{args.dim}_k{k}_{metric.lower()}_m{m}_s{ksub}'
        command = [sys.executable, str(compiler), '--dim', str(args.dim), '--k', str(k), '--metric', metric,
                   '--block', str(args.block), '--select-block', str(args.select_block), '--m', str(m), '--ksub', str(ksub),
                   '--output', str(bundle)]
        if args.pq_lut and m > 0:
            command.append('--pq-lut')
        if args.tle:
            command.append('--tle')
        subprocess.run(command, check=True)
    env = dict(os.environ, KNOWHERE_FL_PPU_KERNEL_DIR=str(root))
    subprocess.run([str(Path(args.test_executable).resolve()), str(args.device), str(args.dim),
                    str(args.k), str(args.m)], env=env, check=True)
    assert 'torch' not in sys.modules
    print('PASS: no PyTorch imported by the test runner')


if __name__ == '__main__':
    main()
