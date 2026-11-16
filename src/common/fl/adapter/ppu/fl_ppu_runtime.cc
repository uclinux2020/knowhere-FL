// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#include "common/fl/adapter/ppu/fl_ppu_runtime.h"

#ifdef KNOWHERE_FL_WITH_PPU
#include <hggc.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

namespace knowhere::fl::ppu {
namespace {
void Check(HGresult result) {
    if (result != HGGC_SUCCESS) {
        const char* text = nullptr;
        hgGetErrorString(result, &text);
        throw std::runtime_error(text != nullptr ? text : "HG driver error");
    }
}
HGdeviceptr Ptr(const void* p) { return static_cast<HGdeviceptr>(reinterpret_cast<uintptr_t>(p)); }
HGstream Stream(void* p) { return static_cast<HGstream>(p); }
struct ContextState {
    std::mutex mutex;
    std::unordered_map<int, HGcontext> contexts;
    std::unordered_map<void*, HGcontext> allocations;
    std::unordered_map<void*, HGcontext> streams;
};
ContextState& State() {
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
    ~ContextScope() { hgCtxSetCurrent(previous); }
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
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        HGcontext previous = nullptr;
        hgCtxGetCurrent(&previous);
        hgCtxSetCurrent(ctx);
        hgStreamSynchronize(stream);
        hgMemFree(ptr);
        hgCtxSetCurrent(previous);
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
        if (warps == 0 || warps > 32) throw std::runtime_error("Invalid PPU warp count");
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
    void Launch(std::vector<void*> params, unsigned grid, HGstream stream, unsigned queries = 1) const {
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
    int64_t dim, k, m, ksub, block, select_block, fused_block, queries_per_block;
    bool implicit_ids = false;
    int metric;
    std::unordered_map<std::string, std::unique_ptr<Kernel>> kernels;
    explicit Bundle(const std::filesystem::path& root) {
        std::ifstream file(root / "manifest.json");
        nlohmann::json meta;
        file >> meta;
        if (meta.at("schema") != 1 || meta.at("backend") != "flagtree-ppu" ||
            (meta.at("implementation") != "triton" && meta.at("implementation") != "triton-tle")) {
            throw std::runtime_error("Not a FlagTree PPU Triton bundle");
        }
        dim = meta.at("dim"); k = meta.at("k"); metric = meta.at("metric");
        m = meta.at("m"); ksub = meta.at("ksub");
        block = meta.at("block"); select_block = meta.at("select_block");
        fused_block = meta.value("fused_block", 256);
        queries_per_block = meta.value("queries_per_block", 1);
        implicit_ids = meta.value("implicit_ids", false);
        if (dim <= 0 || k <= 0 || 2 * k > select_block || select_block > 65536 || block <= 0 ||
            (block & (block - 1)) || (select_block & (select_block - 1)) || (metric != 0 && metric != 1) ||
            (queries_per_block != 1 && queries_per_block != 2 && queries_per_block != 4 &&
             queries_per_block != 16 && queries_per_block != 32) ||
            fused_block <= 0 || (fused_block & (fused_block - 1))) {
            throw std::runtime_error("Invalid PPU retrieval bundle shape");
        }
        for (auto& item : meta.at("kernels").items()) {
            kernels.emplace(item.key(), std::make_unique<Kernel>(root, item.value()));
        }
    }
    Kernel& Get(const std::string& name) { return *kernels.at(name); }
};
std::mutex bundle_mutex;
// Modules are context-local, not process-wide singletons for the first card.
std::unordered_map<HGcontext, std::unordered_map<std::string, std::shared_ptr<Bundle>>> bundles;
std::shared_ptr<Bundle> Load(int64_t dim, int64_t k, int metric, int64_t m, int64_t ksub) {
    const char* root = std::getenv("KNOWHERE_FL_PPU_KERNEL_DIR");
    if (!root || !*root) throw std::runtime_error("KNOWHERE_FL_PPU_KERNEL_DIR is unset");
    // One directory per specialization; bundles for arbitrary nb/nq reuse
    // dynamic scalar arguments and do not require recompilation.
    std::string name = "d" + std::to_string(dim) + "_k" + std::to_string(k) + "_" +
                       (metric == 0 ? "l2" : "ip") + "_m" + std::to_string(m) + "_s" + std::to_string(ksub);
    auto path = (std::filesystem::path(root) / name).string();
    HGcontext ctx = nullptr;
    Check(hgCtxGetCurrent(&ctx));
    if (!ctx) throw std::runtime_error("No current PPU context");
    std::lock_guard<std::mutex> lock(bundle_mutex);
    auto& bundle = bundles[ctx][path];
    if (!bundle) bundle = std::make_shared<Bundle>(path);
    if (bundle->dim != dim || bundle->k != k || bundle->metric != metric || bundle->m != m ||
        (m != 0 && bundle->ksub != ksub)) throw std::runtime_error("PPU bundle specialization mismatch");
    return bundle;
}
}  // namespace

int DeviceCount() {
    int count = 0;
    return hgInit(0) == HGGC_SUCCESS && hgDeviceGetCount(&count) == HGGC_SUCCESS ? count : 0;
}
int SetDevice(int id) {
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
int DeviceSync() { return static_cast<int>(hgCtxSynchronize()); }
void* Alloc(size_t bytes) {
    HGcontext ctx = nullptr;
    HGdeviceptr ptr = 0;
    if (hgCtxGetCurrent(&ctx) != HGGC_SUCCESS || !ctx || hgMemAlloc(&ptr, bytes) != HGGC_SUCCESS) return nullptr;
    void* result = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    std::lock_guard<std::mutex> lock(State().mutex);
    State().allocations[result] = ctx;
    return result;
}
void Free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(State().mutex);
    auto it = State().allocations.find(ptr);
    if (it == State().allocations.end()) return;
    try { ContextScope scope(it->second); Check(hgMemFree(Ptr(ptr))); State().allocations.erase(it); }
    catch (const std::exception& e) { std::cerr << "FL PPU free: " << e.what() << '\n'; }
}
int CopyH2D(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0) return 0;
    if (stream && hgStreamSynchronize(Stream(stream)) != HGGC_SUCCESS) return -1;
    return static_cast<int>(hgMemcpyHtoD(Ptr(dst), src, bytes));
}
int CopyD2H(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0) return 0;
    if (stream && hgStreamSynchronize(Stream(stream)) != HGGC_SUCCESS) return -1;
    return static_cast<int>(hgMemcpyDtoH(dst, Ptr(src), bytes));
}
int CopyD2D(void* dst, const void* src, size_t bytes, void* stream) {
    if (bytes == 0) return 0;
    if (stream && hgStreamSynchronize(Stream(stream)) != HGGC_SUCCESS) return -1;
    return static_cast<int>(hgMemcpyDtoD(Ptr(dst), Ptr(src), bytes));
}
void* StreamCreate() {
    HGstream stream = nullptr;
    HGcontext ctx = nullptr;
    if (hgCtxGetCurrent(&ctx) != HGGC_SUCCESS || hgStreamCreate(&stream, 0) != HGGC_SUCCESS) return nullptr;
    std::lock_guard<std::mutex> lock(State().mutex);
    State().streams[stream] = ctx;
    return stream;
}
void StreamDestroy(void* stream) {
    if (!stream) return;
    std::lock_guard<std::mutex> lock(State().mutex);
    auto it = State().streams.find(stream);
    if (it == State().streams.end()) return;
    try { ContextScope scope(it->second); Check(hgStreamSynchronize(Stream(stream)));
          Check(hgStreamDestroy(Stream(stream))); State().streams.erase(it); }
    catch (const std::exception& e) { std::cerr << "FL PPU stream_destroy: " << e.what() << '\n'; }
}
int StreamSync(void* stream) { return static_cast<int>(hgStreamSynchronize(Stream(stream))); }

int Search(const void* xb, int64_t nb, const void* xq, int64_t nq, int64_t dim, int64_t k, int metric,
           const int64_t* candidates, const int64_t* offsets, float* distances, int64_t* ids, void* raw_stream,
           const void* codes, int64_t m, int64_t ksub, int64_t dsub) {
    HGstream stream = Stream(raw_stream);
    try {
        if (!xb || !xq || !distances || !ids || nb <= 0 || nq <= 0 || dim <= 0 || k <= 0 ||
            (metric != 0 && metric != 1) || (codes && (m <= 0 || dsub <= 0 || m > dim || dim / m != dsub ||
                                                   dim % m || ksub <= 0 || ksub > 256))) return -1;
        const bool ivf = offsets != nullptr;
        if (codes && !ivf) return -1;
        if (ivf) {
            if (offsets[0] != 0 || offsets[nq] < 0 ||
                static_cast<uint64_t>(offsets[nq]) > std::numeric_limits<size_t>::max() / sizeof(int64_t)) return -1;
            for (int64_t q = 0; q < nq; ++q) if (offsets[q] > offsets[q + 1]) return -1;
            if (offsets[nq] && !candidates) return -1;
        }
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
        if (width > 65535 * bundle->block) throw std::runtime_error("PPU score grid exceeds supported width");
        const bool implicit = !ivf && !codes && bundle->implicit_ids;
        if (implicit && (!bundle->kernels.count("select_implicit") || !bundle->kernels.count("batched_distance"))) {
            throw std::runtime_error("Incomplete implicit ID kernel bundle");
        }
        const bool use_lut = codes && bundle->kernels.count("pq_lut") && bundle->kernels.count("pq_adc");
        const std::string fused_name = ivf ? "fused_ivf" : "fused_distance";
        const bool fused = !codes && bundle->kernels.count(fused_name) != 0;
        const int64_t local_block = fused ? bundle->fused_block : bundle->select_block;
        const int64_t tiles = (width + local_block - 1) / local_block;
        const int64_t compact_width = tiles * k;
        const uint64_t row_bytes = (fused ? 0 : static_cast<uint64_t>(width) * (implicit ? 4 : 12)) +
                                   static_cast<uint64_t>(2 * compact_width) * 12 +
                                   (use_lut ? static_cast<uint64_t>(m * ksub) * sizeof(float) : 0);
        uint64_t workspace_limit = 64ULL << 20;
        if (const char* raw = std::getenv("KNOWHERE_FL_PPU_WORKSPACE_MB")) {
            char* end = nullptr;
            const auto mb = std::strtoull(raw, &end, 10);
            if (end == raw || *end != '\0' || *raw == '-' || mb == 0 || mb > 1024) {
                throw std::runtime_error("KNOWHERE_FL_PPU_WORKSPACE_MB must be in [1, 1024]");
            }
            workspace_limit = mb << 20;
        }
        if (row_bytes > workspace_limit) throw std::runtime_error("PPU workspace budget cannot fit one query");
        const int64_t batch = std::min<int64_t>(nq, workspace_limit / row_bytes);
        Buffer candidate_buffer(ivf ? static_cast<size_t>(offsets[nq]) * sizeof(int64_t) : 1, stream);
        Buffer offset_buffer(ivf ? (nq + 1) * sizeof(int64_t) : 1, stream);
        Buffer lookup(use_lut ? batch * m * ksub * sizeof(float) : 1, stream);
        Buffer scores(fused ? 1 : batch * width * sizeof(float), stream);
        Buffer members(fused || implicit ? 1 : batch * width * sizeof(int64_t), stream);
        Buffer compact_a(batch * compact_width * sizeof(float), stream);
        Buffer ids_a(batch * compact_width * sizeof(int64_t), stream);
        Buffer compact_b(batch * compact_width * sizeof(float), stream);
        Buffer ids_b(batch * compact_width * sizeof(int64_t), stream);
        if (ivf) {
            if (offsets[nq]) Check(hgMemcpyHtoD(candidate_buffer.ptr, candidates, offsets[nq] * sizeof(int64_t)));
            Check(hgMemcpyHtoD(offset_buffer.ptr, offsets, (nq + 1) * sizeof(int64_t)));
        }
        for (int64_t q = 0; q < nq; q += batch) {
            unsigned query_count = static_cast<unsigned>(std::min(batch, nq - q));
            HGdeviceptr query_ptr = Ptr(xq) + q * dim * sizeof(float);
            HGdeviceptr vectors = Ptr(xb), encoded_codes = Ptr(codes);
            HGdeviceptr off_ptr = offset_buffer.ptr + (ivf ? q * sizeof(int64_t) : 0);
            int64_t base = 0;
            unsigned score_grid = static_cast<unsigned>((width + bundle->block - 1) / bundle->block);
            int64_t fused_tiles = tiles;
            if (fused) {
                bundle->Get(fused_name).Launch({&vectors, &query_ptr, &candidate_buffer.ptr, &off_ptr,
                                               &compact_a.ptr, &ids_a.ptr, &nb, &width, &fused_tiles},
                                              static_cast<unsigned>(tiles), stream, query_count);
            } else if (use_lut) {
                bundle->Get("pq_lut").Launch({&vectors, &query_ptr, &lookup.ptr},
                                              static_cast<unsigned>(m), stream, query_count);
                bundle->Get("pq_adc").Launch({&lookup.ptr, &encoded_codes, &candidate_buffer.ptr, &off_ptr,
                                              &scores.ptr, &members.ptr, &nb, &width}, score_grid, stream, query_count);
            } else if (codes) {
                bundle->Get("pq").Launch({&vectors, &encoded_codes, &query_ptr, &candidate_buffer.ptr, &off_ptr,
                                          &scores.ptr, &members.ptr, &nb, &width, &base}, score_grid, stream, query_count);
            } else if (!ivf && bundle->kernels.count("batched_distance")) {
                int64_t queries = query_count;
                bundle->Get("batched_distance").Launch({&vectors, &query_ptr, &scores.ptr, &members.ptr,
                                                        &nb, &width, &queries}, score_grid, stream,
                                                       (query_count + bundle->queries_per_block - 1) / bundle->queries_per_block);
            } else {
                bundle->Get(ivf ? "ivf_flat" : "distance").Launch(
                    {&vectors, &query_ptr, &candidate_buffer.ptr, &off_ptr, &scores.ptr, &members.ptr, &nb, &width, &base},
                    score_grid, stream, query_count);
            }
            HGdeviceptr in_score = fused ? compact_a.ptr : scores.ptr;
            HGdeviceptr in_ids = fused ? ids_a.ptr : members.ptr;
            HGdeviceptr out_score = fused ? compact_b.ptr : compact_a.ptr;
            HGdeviceptr out_ids = fused ? ids_b.ptr : ids_a.ptr;
            int64_t input_width = fused ? compact_width : width;
            bool use_a = !fused;
            bool first_level = true;
            while (true) {
                int64_t level_tiles = (input_width + bundle->select_block - 1) / bundle->select_block;
                bundle->Get(implicit && first_level ? "select_implicit" : "select").Launch({&in_score, &in_ids, &out_score, &out_ids, &input_width, &level_tiles},
                                             static_cast<unsigned>(level_tiles), stream, query_count);
                first_level = false;
                if (level_tiles == 1) break;
                input_width = level_tiles * k;
                in_score = out_score; in_ids = out_ids;
                use_a = !use_a;
                out_score = use_a ? compact_a.ptr : compact_b.ptr;
                out_ids = use_a ? ids_a.ptr : ids_b.ptr;
            }
            Check(hgStreamSynchronize(stream));
            Check(hgMemcpyDtoH(distances + q * k, out_score, query_count * k * sizeof(float)));
            Check(hgMemcpyDtoH(ids + q * k, out_ids, query_count * k * sizeof(int64_t)));
        }
        return 0;
    } catch (const std::exception& e) {
        // Never free workspace while a failed launch chain may still use it.
        hgStreamSynchronize(stream);
        std::cerr << "FL PPU Triton search failed (CPU fallback required): " << e.what() << '\n';
        return -1;
    }
}
int Assign(const void* xb, int64_t nb, int64_t dim, const void* centroids, int64_t nlist, int metric,
           int64_t* assign, void* stream) {
    if (nb <= 0) return -1;
    std::vector<float> distances(static_cast<size_t>(nb));
    return Search(centroids, nlist, xb, nb, dim, 1, metric, nullptr, nullptr, distances.data(), assign, stream);
}
}  // namespace knowhere::fl::ppu
#endif
