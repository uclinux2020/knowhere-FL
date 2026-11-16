// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Native HG host adapter for FlagTree AOT Triton/Triton-TLE retrieval kernels.
// No CUDA-C device code, ATen, PyTorch, or speculative hgAot wrapper ABI.
#include "common/fl/fl_device.h"

#ifdef KNOWHERE_FL_WITH_PPU
#include "common/fl/adapter/ppu/fl_ppu_runtime.h"
#include "knowhere/log.h"

namespace knowhere::fl {
namespace {
int TopK(const void* xb, int64_t nb, const void* xq, int64_t nq, int64_t dim, int64_t k, int metric,
         float* distances, int64_t* ids, void* stream) {
    return ppu::Search(xb, nb, xq, nq, dim, k, metric, nullptr, nullptr, distances, ids, stream);
}
int IvfFlat(const void* xb, int64_t nb, int64_t dim, const void* xq, int64_t nq, const int64_t* candidates,
            const int64_t* offsets, int64_t k, int metric, float* distances, int64_t* ids, void* stream) {
    if (!offsets) return -1;
    return ppu::Search(xb, nb, xq, nq, dim, k, metric, candidates, offsets, distances, ids, stream);
}
int IvfPq(const void* codebooks, int64_t m, int64_t ksub, int64_t dsub, const void* codes, int64_t nb,
          const void* xq, int64_t nq, int64_t dim, const int64_t* candidates, const int64_t* offsets,
          int64_t k, int metric, float* distances, int64_t* ids, void* stream) {
    if (!codes || !offsets) return -1;
    return ppu::Search(codebooks, nb, xq, nq, dim, k, metric, candidates, offsets, distances, ids, stream,
                       codes, m, ksub, dsub);
}
const FlDeviceOps ops = {
    ppu::DeviceCount, ppu::SetDevice, ppu::DeviceSync,
    ppu::Alloc, ppu::Free, ppu::CopyH2D, ppu::CopyD2H, ppu::CopyD2D,
    ppu::StreamCreate, ppu::StreamDestroy, ppu::StreamSync,
    TopK, IvfFlat, ppu::Assign, IvfPq,
    nullptr  // Graph traversal remains an explicit CPU fallback; no fake PPU acceleration.
};
}  // namespace
bool RegisterPpuAdapter() {
    if (ppu::DeviceCount() <= 0) return false;
    RegisterDeviceOps(&ops);
    LOG_KNOWHERE_INFO_ << "FL PPU native HG adapter registered (FlagTree Triton/Triton-TLE AOT)";
    return true;
}
}  // namespace knowhere::fl
#endif
