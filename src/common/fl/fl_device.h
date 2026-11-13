// Copyright (C) 2026 Knowhere-FL contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef KNOWHERE_FL_DEVICE_H
#define KNOWHERE_FL_DEVICE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace knowhere {
namespace fl {

// ---------------------------------------------------------------------------
// Device abstraction layer for the FL (FlagOS) backend.
//
// Knowhere-FL never calls a chip runtime directly. All device interaction goes
// through FlDeviceOps, a C-ABI function table implemented by a per-chip
// adapter (e.g. the PPU adapter that wraps the validated FlagFit launch path:
// HIDDEN_PARAM / hgLaunchKernelEx / device context binding). The adapter is
// registered at process start (or lazily via RegisterPpuAdapter()) before any
// FL index is built.
//
// Metric encoding for kernels: 0 = L2, 1 = IP. COSINE is intentionally not a
// kernel metric; per design decision D4 it must be handled by normalizing
// vectors and using IP.
// ---------------------------------------------------------------------------

struct FlDeviceOps {
    // device management
    int (*device_count)();
    int (*set_device)(int device_id);
    int (*device_sync)();

    // memory
    void* (*malloc)(size_t bytes);
    void (*free)(void* ptr);
    int (*memcpy_h2d)(void* dst, const void* src, size_t bytes, void* stream);
    int (*memcpy_d2h)(void* dst, const void* src, size_t bytes, void* stream);
    int (*memcpy_d2d)(void* dst, const void* src, size_t bytes, void* stream);

    // stream
    void* (*stream_create)();
    void (*stream_destroy)(void* stream);
    int (*stream_sync)(void* stream);

    // retrieval kernels (AOT-compiled Triton binaries produced through
    // FlagGems compositions + FlagTree, loaded and launched by the adapter).
    //
    // brute_force_topk: computes top-k of distance(xq[i], xb[j]) for all j.
    //   xb/xq are DEVICE pointers, out_dist/out_ids are HOST pointers.
    //   Returns 0 on success.
    int (*brute_force_topk)(const void* xb, int64_t nb, const void* xq, int64_t nq, int64_t dim, int64_t k, int metric,
                            float* out_dist, int64_t* out_ids, void* stream);

    // ivf_flat_scan: per-query top-k over a candidate subset of xb.
    //   xb/xq are DEVICE pointers; cand_ids (concatenated member ids) and
    //   cand_offsets (nq+1 prefix offsets into cand_ids) are HOST pointers;
    //   the adapter uploads the candidate lists itself. out_* are HOST.
    //   Returns 0 on success. May be nullptr -> caller falls back to CPU scan.
    int (*ivf_flat_scan)(const void* xb, int64_t nb, int64_t dim, const void* xq, int64_t nq, const int64_t* cand_ids,
                         const int64_t* cand_offsets, int64_t k, int metric, float* out_dist, int64_t* out_ids,
                         void* stream);
};

// Returns the currently registered ops table, or nullptr when no chip adapter
// has been registered (DeviceAvailable() == false in that case).
const FlDeviceOps*
GetDeviceOps();

// Called by chip adapters at init time. Not thread-safe w.r.t. concurrent
// registration; call once during process/library init.
void
RegisterDeviceOps(const FlDeviceOps* ops);

// True when an adapter is registered and reports at least one device.
bool
DeviceAvailable();

int
DeviceCount();

// ---------------------------------------------------------------------------
// FL device memory pool.
//
// Mirrors the semantics of the official CUDA path's gpu.initMemSize /
// gpu.maxMemSize, plus the OOM pre-check that the CUDA path lacks (design D5):
// callers must Ensure() the bytes they are about to stage; on failure they
// must fall back to CPU instead of crashing the QueryNode.
// ---------------------------------------------------------------------------
class FlMemoryPool {
 public:
    static FlMemoryPool&
    Instance() {
        static FlMemoryPool pool;
        return pool;
    }

    // (Re)configure pool limits in bytes. init_bytes is reserved eagerly on
    // first use; max_bytes is the hard ceiling.
    void
    Configure(size_t init_bytes, size_t max_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        init_bytes_ = init_bytes;
        max_bytes_ = max_bytes;
    }

    // Pre-flight admission check for an upcoming allocation of `bytes`.
    // Returns true when the allocation fits within max_bytes_. Reserving here
    // lets callers decide CPU fallback *before* touching the device.
    bool
    Ensure(size_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (max_bytes_ != 0 && used_ + bytes > max_bytes_) {
            return false;
        }
        return true;
    }

    void
    TrackAlloc(size_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        used_ += bytes;
    }

    void
    TrackFree(size_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        used_ = bytes > used_ ? 0 : used_ - bytes;
    }

    size_t
    Used() const {
        std::lock_guard<std::mutex> lk(mu_);
        return used_;
    }

 private:
    FlMemoryPool() = default;
    mutable std::mutex mu_;
    size_t init_bytes_{0};
    size_t max_bytes_{0};  // 0 = unlimited (mirrors official default semantics)
    size_t used_{0};
};

// RAII helper for device buffers allocated through the ops table. Frees on
// destruction and keeps the pool accounting in sync.
class FlDeviceBuffer {
 public:
    FlDeviceBuffer() = default;
    FlDeviceBuffer(void* ptr, size_t bytes) : ptr_(ptr), bytes_(bytes) {}
    FlDeviceBuffer(const FlDeviceBuffer&) = delete;
    FlDeviceBuffer&
    operator=(const FlDeviceBuffer&) = delete;
    FlDeviceBuffer(FlDeviceBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
        o.ptr_ = nullptr;
        o.bytes_ = 0;
    }
    ~FlDeviceBuffer() {
        reset();
    }

    void*
    get() const {
        return ptr_;
    }
    size_t
    bytes() const {
        return bytes_;
    }
    explicit
    operator bool() const {
        return ptr_ != nullptr;
    }

    void
    reset();

 private:
    void* ptr_{nullptr};
    size_t bytes_{0};
};

}  // namespace fl
}  // namespace knowhere

#endif  // KNOWHERE_FL_DEVICE_H
