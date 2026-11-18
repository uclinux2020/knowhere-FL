// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <cstddef>
#include <cstdint>

namespace knowhere::fl::ppu {
int
DeviceCount();
int
SetDevice(int id);
int
DeviceSync();
void*
Alloc(size_t bytes);
void
Free(void* ptr);
int
CopyH2D(void* dst, const void* src, size_t bytes, void* stream);
int
CopyD2H(void* dst, const void* src, size_t bytes, void* stream);
int
CopyD2D(void* dst, const void* src, size_t bytes, void* stream);
void*
StreamCreate();
void
StreamDestroy(void* stream);
int
StreamSync(void* stream);

// Cache policy is read once at first use. Limits account pooled allocations,
// not uncached fallback buffers or index/query memory. Trimming never touches
// active leases; they may return to the cache after this call completes.
struct WorkspaceCacheStats {
    size_t limit_bytes = 0;
    size_t retained_bytes = 0;
    size_t idle_bytes = 0;
    size_t idle_entries = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};
WorkspaceCacheStats
GetWorkspaceCacheStats();
void
TrimWorkspaceCache();

// Device inputs, host candidate lists and host outputs. A missing or
// incompatible bundle returns -1; callers may explicitly fall back to CPU.
int
Search(const void* xb, int64_t nb, const void* xq, int64_t nq, int64_t dim, int64_t k, int metric,
       const int64_t* candidates, const int64_t* offsets, float* distances, int64_t* ids, void* stream,
       const void* codes = nullptr, int64_t m = 0, int64_t ksub = 0, int64_t dsub = 0);
int
Assign(const void* xb, int64_t nb, int64_t dim, const void* centroids, int64_t nlist, int metric, int64_t* assign,
       void* stream);
}  // namespace knowhere::fl::ppu
