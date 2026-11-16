# PPU (Zhenwu ZW810E) Runtime Shim & Fused Top-K Operator

This directory contains the PPU enablement artifacts for knowhere-FL:
a `libppu_runtime.so` shim implementing the 13 `hg*` launch-wrapper APIs
referenced by `fl_ppu_adapter.cc`, plus a self-developed fused CUDA-C
operator (L2 distance + top-k in a single kernel) and its benchmarks.

All artifacts were validated from scratch on an 8x ZW810E host with the
vendor PPU SDK 2.0.0 (`/usr/local/PPU_SDK`).

## Layout

| Path | Description |
|---|---|
| `ppu_runtime.c` | FlagFit-compatible launch-wrapper shim. dlopens the vendor `libhg_wrapper.so` with `RTLD_NOW \| RTLD_LOCAL \| RTLD_DEEPBIND` and forwards the 13 `hg*` APIs; implements `hgAotKernelLoad` (cuModuleLoadData) and `hgAotLaunchTopK` (`hgLaunchKernelEx`, `HGLaunchConfig` layout per fitel-sys `ppu_sys.rs`). |
| `kernels/brute_force_topk.cu` | Fused L2 + top-k kernel (no distance-matrix materialization). Tuned config: 256 threads/block, 2-4 queries/block, float4 loads, query in shared memory. Build: `$PPU_SDK/CUDA_SDK/bin/nvcc -cubin -O3 brute_force_topk.cu -o topk.cubin`. |
| `bench/fl_bench.cc` | Deep functional/perf benchmark vs exact OpenMP ground truth (scale sweep, nprobe sweep, top-k sweep, concurrency). |
| `bench/t7.c` | End-to-end PPU operator benchmark: H2D, module load, launch, D2H, correctness vs CPU GT, QPS comparison vs CPU FL_BRUTE_FORCE. |
| `bench/build_bench.sh`, `bench/build_smoke.sh` | Build scripts (conan generators environment). |
| `test/t2.c` | Shim unit test (devices/setdev/alloc/H2D/D2H verify/free/stream/sync). |
| `test/t5.c` | Clean-process probe used to isolate symbol-interposition recursion from vendor implementation defects. |

## Key findings (2026-11-15)

1. **ELF symbol interposition root cause**: the shim must export the same
   `hg*` symbols that vendor libraries (libhggc facade / libhggcrt1 /
   libcudart-embedded hgcrt) reference internally. Default global symbol
   resolution interposes the shim's exports into vendor-internal calls,
   causing infinite recursion and a crash in
   `hgppuDeviceInitializeEmptyCtx(device=-4)`.
   Fix: dlopen vendor libraries with `RTLD_LOCAL | RTLD_DEEPBIND`.
   Evidence: `LD_DEBUG=bindings` showed
   `binding file libhggc.so -> libppu_runtime.so: hgStreamCreate`.
2. **Fused-operator speedup (nb x dim=128, nq=200, topk=10, 0/200 errors)**:

   | nb | CPU FL_BRUTE_FORCE | PPU fused op | Speedup |
   |---|---|---|---|
   | 100k | 44.94 ms | 6.65 ms | 6.76x |
   | 500k | 216.30 ms | 27.69 ms | 7.81x |
   | 1M | 431.48 ms | 52.24 ms | 8.26x |

   PPU exact brute-force (52 ms, recall=1.0) is faster than CPU IVF_FLAT
   approximate search (106 ms, recall~0.31) at 1M scale.
3. **FL_IVF_FLAT bug**: recall is non-monotonic in nprobe (16 -> 0.3135,
   64 -> 0.0505, 128 -> 0.0800), deterministic across runs, reproduces on
   both CPU fallback and PPU paths. Needs investigation in the nprobe
   list-selection/sorting logic.

See `docs/build-verification-report-20261115.md` for the full analysis.
