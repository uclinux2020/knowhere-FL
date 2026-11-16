# Copyright (C) 2026 knowhere-FL contributors.
# Licensed under the Apache License, Version 2.0.
"""Keep optional Triton-TLE integrations from importing PyTorch."""

import importlib.abc
import sys


class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('PyTorch is not used by knowhere PPU AOT tools', name=fullname)
        return None


def install():
    if 'torch' in sys.modules:
        raise RuntimeError('Run the PPU AOT tools in a fresh process without PyTorch')
    sys.meta_path.insert(0, _NoTorch())
