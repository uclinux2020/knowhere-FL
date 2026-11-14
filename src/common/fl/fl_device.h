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
// Knowhere-FL: a fork of Knowhere that accelerates vector search on diverse
// AI chips through a backend-neutral device abstraction layer, built on top
// of FlagOS (a unified open-source AI system software stack).
//
// This file is the heart of that abstraction. Knowhere-FL never calls a chip
// runtime directly. All device interaction goes through FlDeviceOps, a C-ABI
// function table implemented by a per-chip adapter (e.g. the PPU adapter that
// wraps the validated FlagFit launch path: HIDDEN_PARAM / hgLaunchKernelEx /
// device context binding). The adapter is registered at process start (or
// lazily via RegisterPpuAdapter()) before any FL index is built.
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
    // Contract: like std::free, must tolerate nullptr (callers pass it on
    // partial-failure cleanup paths).
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

    // kmeans_assign: routes every base vector to its nearest centroid
    // (mm + argmin composition; no atomics, safe on all FlagOS targets).
    //   xb/centroids are DEVICE pointers; assign (nb centroid ids) is HOST.
    //   Returns 0 on success. May be nullptr -> caller falls back to CPU
    //   routing. Centroid *update* is intentionally NOT a device op in v1:
    //   it needs atomic index_add whose PPU performance must be measured
    //   first (see python/fl_kernels/kmeans_composition.py), so the update
    //   step stays on CPU until the benchmark clears it.
    int (*kmeans_assign)(const void* xb, int64_t nb, int64_t dim, const void* centroids, int64_t nlist, int metric,
                         int64_t* assign, void* stream);

    // ivf_pq_scan: per-query top-k over PQ-encoded candidates via ADC
    // (asymmetric distance computation).
    //   codebooks (m * ksub * dsub floats) and codes (nb * m uint8, one code
    //   per subquantizer; v1 supports nbits=8 only) are DEVICE pointers; xq
    //   is DEVICE; cand_ids/cand_offsets are HOST (the adapter uploads them).
    //   The adapter computes per-query LUTs (nq x m x ksub, via mm against
    //   reshaped codebooks - FlagGems composition) on device, then scans
    //   candidates accumulating LUT entries. out_* are HOST.
    //   Returns 0 on success. May be nullptr -> caller falls back to CPU ADC.
    int (*ivf_pq_scan)(const void* codebooks, int64_t m, int64_t ksub, int64_t dsub, const void* codes, int64_t nb,
                       const void* xq, int64_t nq, int64_t dim, const int64_t* cand_ids, const int64_t* cand_offsets,
                       int64_t k, int metric, float* out_dist, int64_t* out_ids, void* stream);

    // graph_search: greedy top-k traversal of a kNN graph (CAGRA-like).
    //   xb/xq are DEVICE pointers; graph CSR (graph_ids concatenated
    //   neighbor ids, graph_offsets nb+1 prefix offsets, max_degree padded
    //   row stride) is HOST and uploaded by the adapter. Traversal keeps a
    //   candidate frontier of `itopk` entries per query. out_* are HOST.
    //   Returns 0 on success. May be nullptr -> caller falls back to the
    //   CPU traversal (the adapt_for_cpu-style mode from the plan).
    int (*graph_search)(const void* xb, int64_t nb, int64_t dim, const int64_t* graph_ids,
                        const int64_t* graph_offsets, int64_t max_degree, const void* xq, int64_t nq, int64_t k,
                        int64_t itopk, int metric, float* out_dist, int64_t* out_ids, void* stream);
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

    // (Re)configure pool limits in bytes. max_bytes is the hard ceiling.
    // init_bytes is currently a reservation HINT only: it is stored and
    // logged but no eager allocation is implemented yet (adapters allocate
    // lazily through ops->malloc). Kept to mirror the official CUDA path's
    // gpu.initMemSize knob.
    void
    Configure(size_t init_bytes, size_t max_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        init_bytes_ = init_bytes;
        max_bytes_ = max_bytes;
    }

    // Read pool limits from the environment (called automatically by
    // RegisterDeviceOps; can be called again to re-apply):
    //   KNOWHERE_FL_POOL_INIT_MB - initial reservation hint, MiB (default 0)
    //   KNOWHERE_FL_POOL_MAX_MB  - hard ceiling, MiB (default 0 = unlimited)
    // Values < 0 or unparseable are ignored with a warning. A max < init is
    // clamped to init. Defined in fl_device.cc.
    void
    ConfigureFromEnv();

    // Pre-flight admission check AND atomic reservation for an upcoming
    // allocation of `bytes`. Returns false without reserving when the pool
    // cap would be exceeded; on success the bytes are already accounted, so
    // concurrent threads cannot race past the ceiling (the previous
    // check-then-TrackAlloc split had a TOCTOU hole). On failure callers
    // must fall back to CPU instead of touching the device (design D5).
    //
    // IMPORTANT: on success the caller owns the reservation. If the actual
    // allocation fails afterwards, it MUST call TrackFree(bytes) to release
    // the reservation. On the RAII path, construct FlDeviceBuffer with
    // tracked=true (the default) and skip the manual TrackAlloc call.
    bool
    TryReserve(size_t bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (max_bytes_ != 0 && used_ + bytes > max_bytes_) {
            return false;
        }
        used_ += bytes;
        return true;
    }

    // Manual accounting for code paths that manage raw ops->malloc/free
    // without FlDeviceBuffer (e.g. the kmeans training staging loop). New
    // code should prefer TryReserve + tracked FlDeviceBuffer.
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
//
// The free function is captured at construction (not re-read from the global
// ops table at destruction), so a buffer allocated by adapter A is never
// freed through a re-registered adapter B. `tracked` must mirror whether the
// owner accounted this allocation with FlMemoryPool::TrackAlloc: only tracked
// buffers are TrackFree'd on release. Transient buffers the caller never
// tracked must pass tracked=false, otherwise the pool counter would be
// driven negative (clamped to 0) and corrupt every later OOM pre-check.
class FlDeviceBuffer {
 public:
    FlDeviceBuffer() = default;
    explicit FlDeviceBuffer(void* ptr, size_t bytes, bool tracked = true)
        : ptr_(ptr),
          bytes_(bytes),
          free_fn_(CaptureFreeFn()),
          tracked_(tracked && ptr != nullptr) {}
    FlDeviceBuffer(const FlDeviceBuffer&) = delete;
    FlDeviceBuffer&
    operator=(const FlDeviceBuffer&) = delete;
    FlDeviceBuffer(FlDeviceBuffer&& o) noexcept
        : ptr_(o.ptr_), bytes_(o.bytes_), free_fn_(o.free_fn_), tracked_(o.tracked_) {
        o.ptr_ = nullptr;
        o.bytes_ = 0;
        o.free_fn_ = nullptr;
        o.tracked_ = false;
    }
    FlDeviceBuffer&
    operator=(FlDeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            ptr_ = o.ptr_;
            bytes_ = o.bytes_;
            free_fn_ = o.free_fn_;
            tracked_ = o.tracked_;
            o.ptr_ = nullptr;
            o.bytes_ = 0;
            o.free_fn_ = nullptr;
            o.tracked_ = false;
        }
        return *this;
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

    // True when this buffer is accounted in FlMemoryPool (paired TrackAlloc /
    // TrackFree). Transient buffers report false.
    bool
    tracked() const {
        return tracked_;
    }

 private:
    static void (*CaptureFreeFn())(void*) {
        const FlDeviceOps* ops = GetDeviceOps();
        return (ops != nullptr) ? ops->free : nullptr;
    }

    void* ptr_{nullptr};
    size_t bytes_{0};
    void (*free_fn_)(void*){nullptr};  // captured at construction
    bool tracked_{false};              // pool-accounted by the owner
};

}  // namespace fl
}  // namespace knowhere

#endif  // KNOWHERE_FL_DEVICE_H
