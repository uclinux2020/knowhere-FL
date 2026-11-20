// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#include "common/fl/adapter/ppu/fl_ppu_runtime.h"

#ifdef KNOWHERE_FL_WITH_PPU
#include <hggc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <syncstream>
#include <unordered_map>
#include <vector>

namespace knowhere::fl::ppu {
namespace {
void
Check(HGresult result) {
    if (result != HGGC_SUCCESS) {
        const char* text = nullptr;
        hgGetErrorString(result, &text);
        throw std::runtime_error(text != nullptr ? text : "HG driver error");
    }
}
HGdeviceptr
Ptr(const void* p) {
    return static_cast<HGdeviceptr>(reinterpret_cast<uintptr_t>(p));
}
HGstream
Stream(void* p) {
    return static_cast<HGstream>(p);
}
struct ContextState {
    std::mutex mutex;
    std::unordered_map<int, HGcontext> contexts;
    std::unordered_map<void*, HGcontext> allocations;
    std::unordered_map<void*, HGcontext> streams;
};
ContextState&
State() {
    // Index registrations can query DeviceCount during static initialization.
    // Function-local state avoids cross-translation-unit initialization order.
    // Keep contexts and ownership records alive through all static index
    // destructors; the driver releases retained primary contexts on exit.
    static auto* state = new ContextState;
    return *state;
}

// Restore the calling thread's context after resource cleanup.
struct ContextScope {
    HGcontext previous = nullptr;
    explicit ContextScope(HGcontext ctx) {
        Check(hgCtxGetCurrent(&previous));
        Check(hgCtxSetCurrent(ctx));
    }
    ~ContextScope() {
        hgCtxSetCurrent(previous);
    }
};
struct Buffer {
    HGdeviceptr ptr = 0;
    HGcontext ctx = nullptr;
    HGstream stream = nullptr;
    explicit Buffer(size_t bytes, HGstream work_stream) : stream(work_stream) {
        Check(hgCtxGetCurrent(&ctx));
        Check(hgMemAlloc(&ptr, std::max<size_t>(bytes, 1)));
    }
    Buffer(const Buffer&) = delete;
    Buffer&
    operator=(const Buffer&) = delete;
    ~Buffer() {
        HGcontext previous = nullptr;
        hgCtxGetCurrent(&previous);
        hgCtxSetCurrent(ctx);
        hgStreamSynchronize(stream);
        hgMemFree(ptr);
        hgCtxSetCurrent(previous);
    }
};
// Opt-in arena reuse. The budget includes both idle and in-flight pooled
// buffers across all contexts; requests beyond it use ordinary owned buffers.
// Keep state alive through static index destruction, like ContextState.
struct WorkspacePool {
    struct Entry {
        size_t bytes;
        std::unique_ptr<Buffer> buffer;
    };
    std::mutex mutex;
    std::vector<Entry> idle;
    size_t bytes = 0;
    size_t limit = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    WorkspacePool() {
        if (const char* raw = std::getenv("KNOWHERE_FL_PPU_WORKSPACE_CACHE_MB")) {
            char* end = nullptr;
            const auto mb = std::strtoull(raw, &end, 10);
            if (end == raw || *end != '\0' || *raw == '-' || mb > 1024)
                throw std::runtime_error("KNOWHERE_FL_PPU_WORKSPACE_CACHE_MB must be in [0, 1024]");
            limit = static_cast<size_t>(mb) << 20;
        }
    }
};
WorkspacePool&
ArenaPool() {
    static auto* pool = new WorkspacePool;
    return *pool;
}
struct Workspace {
    std::unique_ptr<Buffer> buffer;
    WorkspacePool* pool = nullptr;
    size_t bytes = 0;
    bool cache_hit = false;
    bool completed = false;
    explicit Workspace(size_t requested, HGstream stream) {
        auto& state = ArenaPool();
        if (state.limit == 0) {
            buffer = std::make_unique<Buffer>(requested, stream);
            return;
        }
        HGcontext ctx = nullptr;
        Check(hgCtxGetCurrent(&ctx));
        std::vector<WorkspacePool::Entry> retired;
        size_t retired_bytes = 0;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            auto best = state.idle.end();
            for (auto it = state.idle.begin(); it != state.idle.end(); ++it) {
                if (it->buffer->ctx == ctx && it->bytes >= requested &&
                    (best == state.idle.end() || it->bytes < best->bytes))
                    best = it;
            }
            if (best != state.idle.end()) {
                bytes = best->bytes;
                buffer = std::move(best->buffer);
                state.idle.erase(best);
                buffer->stream = stream;
                pool = &state;
                cache_hit = true;
                ++state.hits;
                return;
            }
            ++state.misses;
            // Reserve capacity for replacement while retired allocations still
            // count toward the limit. Never evict a lease currently in flight.
            if (requested <= state.limit && requested > state.limit - state.bytes) {
                size_t available = state.limit - state.bytes;
                for (const auto& entry : state.idle) available += entry.bytes;
                if (available >= requested) {
                    retired.reserve(state.idle.size());
                    while (requested > state.limit - state.bytes + retired_bytes) {
                        retired_bytes += state.idle.front().bytes;
                        retired.push_back(std::move(state.idle.front()));
                        state.idle.erase(state.idle.begin());
                        ++state.evictions;
                    }
                }
            }
            if (!retired_bytes && requested <= state.limit - state.bytes) {
                state.bytes += requested;
                bytes = requested;
                pool = &state;
            }
        }
        if (retired_bytes) {
            // Free in each entry's owning context, outside the pool mutex.
            // Keep the retiring reservation until the physical frees complete.
            retired.clear();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.bytes -= retired_bytes;
            if (requested <= state.limit - state.bytes) {
                state.bytes += requested;
                bytes = requested;
                pool = &state;
            }
        }
        try {
            buffer = std::make_unique<Buffer>(requested, stream);
        } catch (...) {
            if (pool) {
                std::lock_guard<std::mutex> lock(pool->mutex);
                pool->bytes -= bytes;
            }
            throw;
        }
    }
    Workspace(const Workspace&) = delete;
    Workspace&
    operator=(const Workspace&) = delete;
    void
    Release() noexcept {
        if (!buffer)
            return;
        if (pool && completed) {
            // Only a fully completed launch/D2H chain is reusable. Each lease
            // is removed from idle storage until its owning request returns it.
            try {
                ContextScope scope(buffer->ctx);
                Check(hgStreamSynchronize(buffer->stream));
                // An idle buffer must not retain a stream the caller may destroy.
                buffer->stream = nullptr;
                std::lock_guard<std::mutex> lock(pool->mutex);
                pool->idle.push_back({bytes, nullptr});
                pool->idle.back().buffer = std::move(buffer);
                return;
            } catch (...) {
                // Allocation or driver errors discard this entry, never cache it.
            }
        }
        buffer.reset();
        if (pool) {
            std::lock_guard<std::mutex> lock(pool->mutex);
            pool->bytes -= bytes;
        }
    }
    ~Workspace() {
        Release();
    }
};

struct Kernel {
    HGmodule module = nullptr;
    HGfunction function = nullptr;
    HGcontext context = nullptr;
    unsigned shared = 0;
    unsigned warps = 0;
    Kernel(const std::filesystem::path& root, const nlohmann::json& meta) {
        // This first implementation intentionally rejects nonzero scratch
        // and clustered launches rather than guessing their ABI/allocation.
        if (meta.value("num_ctas", 1) != 1 || meta.value("global_scratch_size", 0) != 0 ||
            meta.value("profile_scratch_size", 0) != 0) {
            throw std::runtime_error("PPU retrieval bundle requires unsupported scratch allocation");
        }
        shared = meta.at("shared").get<unsigned>();
        warps = meta.at("num_warps").get<unsigned>();
        if (warps == 0 || warps > 32)
            throw std::runtime_error("Invalid PPU warp count");
        Check(hgCtxGetCurrent(&context));
        Check(hgModuleLoad(&module, (root / meta.at("file").get<std::string>()).c_str()));
        auto result = hgModuleGetFunction(&function, module, meta.at("symbol").get<std::string>().c_str());
        if (result != HGGC_SUCCESS) {
            hgModuleUnload(module);
            module = nullptr;
            Check(result);
        }
        if (shared != 0) {
            result = hgFuncSetAttribute(function, HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared);
            if (result != HGGC_SUCCESS) {
                hgModuleUnload(module);
                module = nullptr;
                Check(result);
            }
        }
    }
    ~Kernel() {
        if (module) {
            HGcontext previous = nullptr;
            hgCtxGetCurrent(&previous);
            hgCtxSetCurrent(context);
            hgModuleUnload(module);
            hgCtxSetCurrent(previous);
        }
    }
    void
    Launch(std::vector<void*> params, unsigned grid, HGstream stream, unsigned queries = 1) const {
        HGdeviceptr global_scratch = 0, profile_scratch = 0;
        params.push_back(&global_scratch);
        params.push_back(&profile_scratch);
        HGlaunchConfig config{};
        config.gridDimX = queries;
        config.gridDimY = grid;
        config.gridDimZ = 1;
        config.blockDimX = warps * 32;
        config.blockDimY = config.blockDimZ = 1;
        config.sharedMemBytes = shared;
        config.hStream = stream;
        Check(hgLaunchKernelEx(&config, function, params.data(), nullptr));
    }
};
struct Bundle {
    int64_t dim, k, m, ksub, block, ivf_block, select_block, fused_block, queries_per_block;
    bool implicit_ids = false;
    int metric;
    std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels;
    explicit Bundle(const std::filesystem::path& root) {
        std::ifstream file(root / "manifest.json");
        nlohmann::json meta;
        file >> meta;
        const int schema = meta.at("schema");
        if ((schema != 1 && schema != 2) || meta.at("backend") != "flagtree-ppu" ||
            (meta.at("implementation") != "triton" && meta.at("implementation") != "triton-tle")) {
            throw std::runtime_error("Not a FlagTree PPU Triton bundle");
        }
        dim = meta.at("dim");
        k = meta.at("k");
        metric = meta.at("metric");
        m = meta.at("m");
        ksub = meta.at("ksub");
        block = meta.at("block");
        // Legacy bundles use one tile size for both dense and gathered scores.
        ivf_block = schema == 2 ? meta.at("ivf_block").get<int64_t>() : meta.value("ivf_block", block);
        if (schema == 1 && ivf_block != block)
            throw std::runtime_error("Split PPU score tiles require schema 2");
        select_block = meta.at("select_block");
        fused_block = meta.value("fused_block", 256);
        queries_per_block = meta.value("queries_per_block", 1);
        implicit_ids = meta.value("implicit_ids", false);
        if (dim <= 0 || k <= 0 || 2 * k > select_block || select_block > 65536 || block <= 0 || (block & (block - 1)) ||
            (select_block & (select_block - 1)) || ivf_block <= 0 || ivf_block > 65536 ||
            (ivf_block & (ivf_block - 1)) || (metric != 0 && metric != 1) ||
            (queries_per_block != 1 && queries_per_block != 2 && queries_per_block != 4 && queries_per_block != 16 &&
             queries_per_block != 32) ||
            fused_block <= 0 || (fused_block & (fused_block - 1))) {
            throw std::runtime_error("Invalid PPU retrieval bundle shape");
        }
        for (auto& item : meta.at("kernels").items()) {
            kernels.emplace(item.key(), std::make_unique<Kernel>(root, item.value()));
        }
    }
    Kernel&
    Get(const std::string& name) {
        return *kernels.at(name);
    }
};
std::mutex bundle_mutex;
// Modules are context-local, not process-wide singletons for the first card.
std::unordered_map<HGcontext, std::unordered_map<std::string, std::shared_ptr<Bundle>>> bundles;
std::shared_ptr<Bundle>
Load(int64_t dim, int64_t k, int metric, int64_t m, int64_t ksub) {
    const char* root = std::getenv("KNOWHERE_FL_PPU_KERNEL_DIR");
    if (!root || !*root)
        throw std::runtime_error("KNOWHERE_FL_PPU_KERNEL_DIR is unset");
    // One directory per specialization; bundles for arbitrary nb/nq reuse
    // dynamic scalar arguments and do not require recompilation.
    std::string name = "d" + std::to_string(dim) + "_k" + std::to_string(k) + "_" + (metric == 0 ? "l2" : "ip") + "_m" +
                       std::to_string(m) + "_s" + std::to_string(ksub);
    auto path = (std::filesystem::path(root) / name).string();
    HGcontext ctx = nullptr;
    Check(hgCtxGetCurrent(&ctx));
    if (!ctx)
        throw std::runtime_error("No current PPU context");
    std::lock_guard<std::mutex> lock(bundle_mutex);
    auto& bundle = bundles[ctx][path];
    if (!bundle)
        bundle = std::make_shared<Bundle>(path);
    if (bundle->dim != dim || bundle->k != k || bundle->metric != metric || bundle->m != m ||
        (m != 0 && bundle->ksub != ksub))
        throw std::runtime_error("PPU bundle specialization mismatch");
    return bundle;
}
}  // namespace

WorkspaceCacheStats
GetWorkspaceCacheStats() {
    auto& state = ArenaPool();
    std::lock_guard<std::mutex> lock(state.mutex);
    WorkspaceCacheStats result;
    result.limit_bytes = state.limit;
    result.retained_bytes = state.bytes;
    result.idle_entries = state.idle.size();
    for (const auto& entry : state.idle) result.idle_bytes += entry.bytes;
    result.hits = state.hits;
    result.misses = state.misses;
    result.evictions = state.evictions;
    return result;
}
void
TrimWorkspaceCache() {
    auto& state = ArenaPool();
    std::vector<WorkspacePool::Entry> retired;
    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        for (const auto& entry : state.idle) bytes += entry.bytes;
        retired.swap(state.idle);
        state.evictions += retired.size();
    }
    retired.clear();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.bytes -= bytes;
}

int
DeviceCount() {
    int count = 0;
    return hgInit(0) == HGGC_SUCCESS && hgDeviceGetCount(&count) == HGGC_SUCCESS ? count : 0;
}
int
SetDevice(int id) {
    try {
        Check(hgInit(0));
        std::lock_guard<std::mutex> lock(State().mutex);
        auto it = State().contexts.find(id);
        if (it == State().contexts.end()) {
            HGdevice device;
            HGcontext ctx = nullptr;
            Check(hgDeviceGet(&device, id));
            Check(hgDevicePrimaryCtxRetain(&ctx, device));
            it = State().contexts.emplace(id, ctx).first;
        }
        Check(hgCtxSetCurrent(it->second));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FL PPU set_device: " << e.what() << '\n';
        return -1;
    }
}
int
DeviceSync() {
    return static_cast<int>(hgCtxSynchronize());
}
void*
Alloc(size_t bytes) {
    HGcontext ctx = nullptr;
    HGdeviceptr ptr = 0;
    if (hgCtxGetCurrent(&ctx) != HGGC_SUCCESS || !ctx || hgMemAlloc(&ptr, bytes) != HGGC_SUCCESS)
        return nullptr;
    void* result = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    std::lock_guard<std::mutex> lock(State().mutex);
    State().allocations[result] = ctx;
    return result;
}
void
Free(void* ptr) {
    if (!ptr)
        return;
    std::lock_guard<std::mutex> lock(State().mutex);
    auto it = State().allocations.find(ptr);
    if (it == State().allocations.end())
        return;
    try {
        ContextScope scope(it->second);
        Check(hgMemFree(Ptr(ptr)));
        State().allocations.erase(it);
    } catch (const std::exception& e) {
        std::cerr << "FL PPU free: " << e.what() << '\n';
    }
}
int
CopyH2D(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0)
        return 0;
    if (stream && hgStreamSynchronize(Stream(stream)) != HGGC_SUCCESS)
        return -1;
    return static_cast<int>(hgMemcpyHtoD(Ptr(dst), src, bytes));
}
int
CopyD2H(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0)
        return 0;
    if (stream && hgStreamSynchronize(Stream(stream)) != HGGC_SUCCESS)
        return -1;
    return static_cast<int>(hgMemcpyDtoH(dst, Ptr(src), bytes));
}
int
CopyD2D(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0)
        return 0;
    if (stream && hgStreamSynchronize(Stream(stream)) != HGGC_SUCCESS)
        return -1;
    return static_cast<int>(hgMemcpyDtoD(Ptr(dst), Ptr(src), bytes));
}
void*
StreamCreate() {
    HGstream stream = nullptr;
    HGcontext ctx = nullptr;
    if (hgCtxGetCurrent(&ctx) != HGGC_SUCCESS || hgStreamCreate(&stream, 0) != HGGC_SUCCESS)
        return nullptr;
    std::lock_guard<std::mutex> lock(State().mutex);
    State().streams[stream] = ctx;
    return stream;
}
void
StreamDestroy(void* stream) {
    if (!stream)
        return;
    std::lock_guard<std::mutex> lock(State().mutex);
    auto it = State().streams.find(stream);
    if (it == State().streams.end())
        return;
    try {
        ContextScope scope(it->second);
        Check(hgStreamSynchronize(Stream(stream)));
        Check(hgStreamDestroy(Stream(stream)));
        State().streams.erase(it);
    } catch (const std::exception& e) {
        std::cerr << "FL PPU stream_destroy: " << e.what() << '\n';
    }
}
int
StreamSync(void* stream) {
    return static_cast<int>(hgStreamSynchronize(Stream(stream)));
}

int
Search(const void* xb, int64_t nb, const void* xq, int64_t nq, int64_t dim, int64_t k, int metric,
       const int64_t* candidates, const int64_t* offsets, float* distances, int64_t* ids, void* raw_stream,
       const void* codes, int64_t m, int64_t ksub, int64_t dsub) {
    HGstream stream = Stream(raw_stream);
    try {
        if (!xb || !xq || !distances || !ids || nb <= 0 || nq <= 0 || dim <= 0 || k <= 0 ||
            (metric != 0 && metric != 1) ||
            (codes && (m <= 0 || dsub <= 0 || m > dim || dim / m != dsub || dim % m || ksub <= 0 || ksub > 256)))
            return -1;
        const bool ivf = offsets != nullptr;
        if (codes && !ivf)
            return -1;
        if (ivf) {
            if (offsets[0] != 0 || offsets[nq] < 0 ||
                static_cast<uint64_t>(offsets[nq]) > std::numeric_limits<size_t>::max() / sizeof(int64_t))
                return -1;
            for (int64_t q = 0; q < nq; ++q)
                if (offsets[q] > offsets[q + 1])
                    return -1;
            if (offsets[nq] && !candidates)
                return -1;
        }
        const bool profile = std::getenv("KNOWHERE_FL_PPU_PROFILE") != nullptr;
        using Clock = std::chrono::steady_clock;
        const auto runtime_started = profile ? Clock::now() : Clock::time_point{};
        double candidate_h2d_ms = 0, d2h_ms = 0, launch_wait_ms = 0;
        int64_t batch_calls = 0;
        auto bundle = Load(dim, k, metric, codes ? m : 0, codes ? ksub : 0);
        // Batch query and candidate tiles on the device. Each level is a
        // single grid launch; never serialize one launch per candidate chunk.
        int64_t width = nb;
        if (ivf) {
            width = 0;
            for (int64_t q = 0; q < nq; ++q) width = std::max(width, offsets[q + 1] - offsets[q]);
        }
        if (width == 0) {
            std::fill_n(ids, nq * k, -1);
            std::fill_n(distances, nq * k, metric == 0 ? INFINITY : -INFINITY);
            return 0;
        }
        const int64_t score_block = ivf ? bundle->ivf_block : bundle->block;
        if (width > 65535 * score_block)
            throw std::runtime_error("PPU score grid exceeds supported width");
        const bool implicit = !ivf && !codes && bundle->implicit_ids;
        if (implicit && (!bundle->kernels.count("select_implicit") || !bundle->kernels.count("batched_distance"))) {
            throw std::runtime_error("Incomplete implicit ID kernel bundle");
        }
        const bool use_lut = codes && bundle->kernels.count("pq_lut") && bundle->kernels.count("pq_adc");
        const bool compact = !ivf && !codes && bundle->kernels.count("matrix_compact");
        if (compact && (k != 1 || !bundle->kernels.count("transpose_base"))) {
            throw std::runtime_error("Matrix compact kernel requires top-1 and packed base");
        }
        const std::string fused_name = compact ? "matrix_compact" : ivf ? "fused_ivf" : "fused_distance";
        const bool fused = !codes && bundle->kernels.count(fused_name) != 0;
        const int64_t local_block = compact ? bundle->block : fused ? bundle->fused_block : bundle->select_block;
        const int64_t tiles = (width + local_block - 1) / local_block;
        const int64_t compact_width = tiles * k;
        const uint64_t row_bytes = (fused ? 0 : static_cast<uint64_t>(width) * (implicit ? 4 : 12)) +
                                   static_cast<uint64_t>(2 * compact_width) * 12 +
                                   (use_lut ? static_cast<uint64_t>(m * ksub) * sizeof(float) : 0) +
                                   (ivf ? static_cast<uint64_t>(width) * sizeof(int64_t) + sizeof(int64_t) : 0);
        uint64_t workspace_limit = 64ULL << 20;
        if (const char* raw = std::getenv("KNOWHERE_FL_PPU_WORKSPACE_MB")) {
            char* end = nullptr;
            const auto mb = std::strtoull(raw, &end, 10);
            if (end == raw || *end != '\0' || *raw == '-' || mb == 0 || mb > 1024) {
                throw std::runtime_error("KNOWHERE_FL_PPU_WORKSPACE_MB must be in [1, 1024]");
            }
            workspace_limit = mb << 20;
        }
        // Include the final CSR offset and worst-case arena alignment in the
        // IVF budget rather than exceeding the limit at an exact row boundary.
        const uint64_t overhead = ivf ? sizeof(int64_t) + 9 * 255 : 0;
        if (row_bytes > workspace_limit - overhead)
            throw std::runtime_error("PPU workspace budget cannot fit one query");
        int64_t batch = std::min<int64_t>(nq, (workspace_limit - overhead) / row_bytes);
        // Avoid recomputing masked query lanes at every budget-limited batch
        // boundary. Keep the final partial batch and single-query requests valid.
        if (!ivf && !codes && batch < nq && batch >= bundle->queries_per_block) {
            batch -= batch % bundle->queries_per_block;
        }
        // Stream CSR in the same query batches as scores. Large requests must
        // not retain all nq candidate lists on device at once.
        int64_t max_batch_candidates = 0;
        if (ivf) {
            for (int64_t q = 0; q < nq; q += batch) {
                max_batch_candidates = std::max(max_batch_candidates, offsets[std::min(q + batch, nq)] - offsets[q]);
            }
        }
        // One request-owned allocation avoids repeated HG allocator and
        // destructor synchronizations. No cross-request result/data caching.
        const size_t sizes[] = {ivf ? static_cast<size_t>(max_batch_candidates) * sizeof(int64_t) : 1,
                                ivf ? static_cast<size_t>(batch + 1) * sizeof(int64_t) : 1,
                                use_lut ? static_cast<size_t>(batch * m * ksub) * sizeof(float) : 1,
                                fused ? 1 : static_cast<size_t>(batch * width) * sizeof(float),
                                fused || implicit ? 1 : static_cast<size_t>(batch * width) * sizeof(int64_t),
                                static_cast<size_t>(batch * compact_width) * sizeof(float),
                                static_cast<size_t>(batch * compact_width) * sizeof(int64_t),
                                static_cast<size_t>(batch * compact_width) * sizeof(float),
                                static_cast<size_t>(batch * compact_width) * sizeof(int64_t)};
        size_t total = 0;
        size_t starts[9];
        for (size_t i = 0; i < 9; ++i) {
            if (sizes[i] > std::numeric_limits<size_t>::max() - 255 ||
                total > std::numeric_limits<size_t>::max() - ((sizes[i] + 255) & ~size_t(255))) {
                throw std::runtime_error("PPU workspace size overflow");
            }
            starts[i] = total;
            total += (sizes[i] + 255) & ~size_t(255);
        }
        const auto alloc_started = profile ? Clock::now() : Clock::time_point{};
        Workspace workspace(total, stream);
        const double workspace_alloc_ms =
            profile ? std::chrono::duration<double, std::milli>(Clock::now() - alloc_started).count() : 0;
        struct Slice {
            HGdeviceptr ptr;
        };
        Slice candidate_buffer{workspace.buffer->ptr + starts[0]}, offset_buffer{workspace.buffer->ptr + starts[1]};
        Slice lookup{workspace.buffer->ptr + starts[2]}, scores{workspace.buffer->ptr + starts[3]};
        Slice members{workspace.buffer->ptr + starts[4]}, compact_a{workspace.buffer->ptr + starts[5]};
        Slice ids_a{workspace.buffer->ptr + starts[6]}, compact_b{workspace.buffer->ptr + starts[7]};
        Slice ids_b{workspace.buffer->ptr + starts[8]};
        std::vector<int64_t> batch_offsets(ivf ? static_cast<size_t>(batch + 1) : 0);
        std::unique_ptr<Buffer> packed;
        HGdeviceptr search_vectors = Ptr(xb);
        if (!ivf && !codes && bundle->kernels.count("transpose_base")) {
            const uint64_t max_elements = std::numeric_limits<size_t>::max() / sizeof(float);
            if (static_cast<uint64_t>(nb) > max_elements / static_cast<uint64_t>(dim)) {
                throw std::runtime_error("PPU packed base size overflow");
            }
            const size_t packed_bytes = static_cast<size_t>(nb) * static_cast<size_t>(dim) * sizeof(float);
            // Packing is opt-in and has a separate hard cap; do not conceal
            // this extra allocation behind the score-workspace budget.
            if (packed_bytes > (1024ULL << 20))
                throw std::runtime_error("PPU packed base exceeds 1024 MiB cap");
            packed = std::make_unique<Buffer>(packed_bytes, stream);
            HGdeviceptr source = Ptr(xb), output = packed->ptr;
            bundle->Get("transpose_base")
                .Launch({&source, &output, &nb}, static_cast<unsigned>((dim + 31) / 32), stream,
                        static_cast<unsigned>((nb + 31) / 32));
            search_vectors = packed->ptr;
        }
        struct Events {
            HGevent begin = nullptr, scored = nullptr, selected = nullptr;
            explicit Events(bool enabled) {
                if (!enabled)
                    return;
                Check(hgEventCreate(&begin, 0));
                try {
                    Check(hgEventCreate(&scored, 0));
                    Check(hgEventCreate(&selected, 0));
                } catch (...) {
                    if (scored)
                        hgEventDestroy(scored);
                    hgEventDestroy(begin);
                    throw;
                }
            }
            ~Events() {
                if (selected)
                    hgEventDestroy(selected);
                if (scored)
                    hgEventDestroy(scored);
                if (begin)
                    hgEventDestroy(begin);
            }
        } events(profile);
        double score_ms = 0, select_ms = 0;
        uint64_t scored_slots = 0;
        const int64_t request_width = width;
        for (int64_t q = 0; q < nq; q += batch) {
            unsigned query_count = static_cast<unsigned>(std::min(batch, nq - q));
            int64_t width = request_width;
            if (ivf) {
                width = 0;
                for (unsigned i = 0; i < query_count; ++i) {
                    width = std::max(width, offsets[q + i + 1] - offsets[q + i]);
                }
                if (width == 0) {
                    std::fill_n(ids + q * k, query_count * k, -1);
                    std::fill_n(distances + q * k, query_count * k, metric == 0 ? INFINITY : -INFINITY);
                    continue;
                }
                const auto copy_started = profile ? Clock::now() : Clock::time_point{};
                const int64_t first = offsets[q], count = offsets[q + query_count] - first;
                for (unsigned i = 0; i <= query_count; ++i) batch_offsets[i] = offsets[q + i] - first;
                if (count)
                    Check(hgMemcpyHtoD(candidate_buffer.ptr, candidates + first, count * sizeof(int64_t)));
                Check(hgMemcpyHtoD(offset_buffer.ptr, batch_offsets.data(), (query_count + 1) * sizeof(int64_t)));
                if (profile)
                    candidate_h2d_ms += std::chrono::duration<double, std::milli>(Clock::now() - copy_started).count();
            }
            ++batch_calls;
            const auto launch_started = profile ? Clock::now() : Clock::time_point{};
            if (profile)
                Check(hgEventRecord(events.begin, stream));
            scored_slots += static_cast<uint64_t>(query_count) * width;
            const int64_t tiles = (width + local_block - 1) / local_block;
            const int64_t compact_width = tiles * k;
            HGdeviceptr query_ptr = Ptr(xq) + q * dim * sizeof(float);
            HGdeviceptr vectors = search_vectors, encoded_codes = Ptr(codes);
            HGdeviceptr off_ptr = offset_buffer.ptr;
            int64_t base = 0;
            unsigned score_grid = static_cast<unsigned>((width + score_block - 1) / score_block);
            int64_t fused_tiles = tiles;
            if (compact) {
                int64_t queries = query_count;
                bundle->Get(fused_name)
                    .Launch({&vectors, &query_ptr, &compact_a.ptr, &ids_a.ptr, &nb, &width, &fused_tiles, &queries},
                            static_cast<unsigned>(tiles), stream,
                            (query_count + bundle->queries_per_block - 1) / bundle->queries_per_block);
            } else if (fused) {
                bundle->Get(fused_name)
                    .Launch({&vectors, &query_ptr, &candidate_buffer.ptr, &off_ptr, &compact_a.ptr, &ids_a.ptr, &nb,
                             &width, &fused_tiles},
                            static_cast<unsigned>(tiles), stream, query_count);
            } else if (use_lut) {
                bundle->Get("pq_lut").Launch({&vectors, &query_ptr, &lookup.ptr}, static_cast<unsigned>(m), stream,
                                             query_count);
                bundle->Get("pq_adc").Launch({&lookup.ptr, &encoded_codes, &candidate_buffer.ptr, &off_ptr, &scores.ptr,
                                              &members.ptr, &nb, &width},
                                             score_grid, stream, query_count);
            } else if (codes) {
                bundle->Get("pq").Launch({&vectors, &encoded_codes, &query_ptr, &candidate_buffer.ptr, &off_ptr,
                                          &scores.ptr, &members.ptr, &nb, &width, &base},
                                         score_grid, stream, query_count);
            } else if (!ivf && bundle->kernels.count("batched_distance")) {
                int64_t queries = query_count;
                bundle->Get("batched_distance")
                    .Launch({&vectors, &query_ptr, &scores.ptr, &members.ptr, &nb, &width, &queries}, score_grid,
                            stream, (query_count + bundle->queries_per_block - 1) / bundle->queries_per_block);
            } else {
                bundle->Get(ivf ? "ivf_flat" : "distance")
                    .Launch({&vectors, &query_ptr, &candidate_buffer.ptr, &off_ptr, &scores.ptr, &members.ptr, &nb,
                             &width, &base},
                            score_grid, stream, query_count);
            }
            if (profile)
                Check(hgEventRecord(events.scored, stream));
            HGdeviceptr in_score = fused ? compact_a.ptr : scores.ptr;
            HGdeviceptr in_ids = fused ? ids_a.ptr : members.ptr;
            HGdeviceptr out_score = fused ? compact_b.ptr : compact_a.ptr;
            HGdeviceptr out_ids = fused ? ids_b.ptr : ids_a.ptr;
            int64_t input_width = fused ? compact_width : width;
            bool use_a = !fused;
            bool first_level = true;
            while (true) {
                int64_t level_tiles = (input_width + bundle->select_block - 1) / bundle->select_block;
                bundle->Get(implicit && !fused && first_level ? "select_implicit" : "select")
                    .Launch({&in_score, &in_ids, &out_score, &out_ids, &input_width, &level_tiles},
                            static_cast<unsigned>(level_tiles), stream, query_count);
                first_level = false;
                if (level_tiles == 1)
                    break;
                input_width = level_tiles * k;
                in_score = out_score;
                in_ids = out_ids;
                use_a = !use_a;
                out_score = use_a ? compact_a.ptr : compact_b.ptr;
                out_ids = use_a ? ids_a.ptr : ids_b.ptr;
            }
            if (profile)
                Check(hgEventRecord(events.selected, stream));
            Check(hgStreamSynchronize(stream));
            if (profile) {
                launch_wait_ms += std::chrono::duration<double, std::milli>(Clock::now() - launch_started).count();
                float elapsed = 0;
                Check(hgEventElapsedTime(&elapsed, events.begin, events.scored));
                score_ms += elapsed;
                Check(hgEventElapsedTime(&elapsed, events.scored, events.selected));
                select_ms += elapsed;
            }
            const auto d2h_started = profile ? Clock::now() : Clock::time_point{};
            Check(hgMemcpyDtoH(distances + q * k, out_score, query_count * k * sizeof(float)));
            Check(hgMemcpyDtoH(ids + q * k, out_ids, query_count * k * sizeof(int64_t)));
            if (profile)
                d2h_ms += std::chrono::duration<double, std::milli>(Clock::now() - d2h_started).count();
        }
        workspace.completed = true;
        const auto release_started = profile ? Clock::now() : Clock::time_point{};
        workspace.Release();
        const double workspace_release_ms =
            profile ? std::chrono::duration<double, std::milli>(Clock::now() - release_started).count() : 0;
        const auto cache_stats = profile ? GetWorkspaceCacheStats() : WorkspaceCacheStats{};
        if (profile)
            std::osyncstream(std::cerr)
                << "PPU_PROFILE score_ms=" << score_ms << " select_ms=" << select_ms << " query_batch=" << batch
                << " candidate_count=" << (ivf ? offsets[nq] : 0) << " candidate_width=" << width
                << " workspace_bytes=" << total << " candidate_device_bytes=" << (ivf ? sizes[0] + sizes[1] : 0)
                << " scored_slots=" << scored_slots << " base_ptr=" << Ptr(xb)
                << " candidate_h2d_ms=" << candidate_h2d_ms << " d2h_ms=" << d2h_ms
                << " workspace_alloc_ms=" << workspace_alloc_ms << " workspace_release_ms=" << workspace_release_ms
                << " workspace_cache_hit=" << workspace.cache_hit << " workspace_lease_bytes=" << workspace.bytes
                << " workspace_cache_retained_bytes=" << cache_stats.retained_bytes
                << " workspace_cache_idle_bytes=" << cache_stats.idle_bytes
                << " workspace_cache_evictions=" << cache_stats.evictions << " launch_wait_ms=" << launch_wait_ms
                << " batch_calls=" << batch_calls
                << " runtime_ms=" << std::chrono::duration<double, std::milli>(Clock::now() - runtime_started).count()
                << '\n';
        return 0;
    } catch (const std::exception& e) {
        // Never free workspace while a failed launch chain may still use it.
        hgStreamSynchronize(stream);
        std::cerr << "FL PPU Triton search failed (CPU fallback required): " << e.what() << '\n';
        return -1;
    }
}
int
Assign(const void* xb, int64_t nb, int64_t dim, const void* centroids, int64_t nlist, int metric, int64_t* assign,
       void* stream) {
    if (nb <= 0)
        return -1;
    std::vector<float> distances(static_cast<size_t>(nb));
    return Search(centroids, nlist, xb, nb, dim, 1, metric, nullptr, nullptr, distances.data(), assign, stream);
}
}  // namespace knowhere::fl::ppu
#endif
