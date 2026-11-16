// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#include "common/fl/adapter/ppu/fl_ppu_runtime.h"

#ifdef KNOWHERE_FL_WITH_PPU
#include <hggc.h>

#include <algorithm>
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
    void Launch(std::vector<void*> params, unsigned grid, HGstream stream) const {
        HGdeviceptr global_scratch = 0, profile_scratch = 0;
        params.push_back(&global_scratch);
        params.push_back(&profile_scratch);
        HGlaunchConfig config{};
        config.gridDimX = 1;
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
    int64_t dim, k, m, ksub, block, select_block;
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
        if (dim <= 0 || k <= 0 || 2 * k > select_block || select_block > 65536 || block <= 0 ||
            (block & (block - 1)) || (select_block & (select_block - 1)) || (metric != 0 && metric != 1)) {
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
void Select(Bundle& bundle, HGdeviceptr scores, HGdeviceptr ids, HGdeviceptr out_scores,
            HGdeviceptr out_ids, int64_t width, HGstream stream) {
    int64_t tiles = 1;
    bundle.Get("select").Launch({&scores, &ids, &out_scores, &out_ids, &width, &tiles}, 1, stream);
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
        const int64_t width_limit = bundle->select_block;
        Buffer cand(ivf ? static_cast<size_t>(offsets[nq]) * sizeof(int64_t) : 1, stream);
        Buffer off(2 * sizeof(int64_t), stream);
        Buffer score(width_limit * sizeof(float), stream), member(width_limit * sizeof(int64_t), stream);
        Buffer local_score(k * sizeof(float), stream), local_ids(k * sizeof(int64_t), stream);
        Buffer merge_score(2 * k * sizeof(float), stream), merge_ids(2 * k * sizeof(int64_t), stream);
        Buffer best_score(k * sizeof(float), stream), best_ids(k * sizeof(int64_t), stream);
        if (ivf && offsets[nq]) Check(hgMemcpyHtoD(cand.ptr, candidates, offsets[nq] * sizeof(int64_t)));
        for (int64_t q = 0; q < nq; ++q) {
            const int64_t count = ivf ? offsets[q + 1] - offsets[q] : nb;
            if (count == 0) {
                std::fill_n(ids + q * k, k, -1);
                std::fill_n(distances + q * k, k, metric == 0 ? std::numeric_limits<float>::infinity() :
                                                                              -std::numeric_limits<float>::infinity());
                continue;
            }
            if (ivf) Check(hgMemcpyHtoD(off.ptr, offsets + q, 2 * sizeof(int64_t)));
            HGdeviceptr query = Ptr(xq) + q * dim * sizeof(float);
            HGdeviceptr base_vectors = Ptr(xb), encoded = Ptr(codes);
            bool first = true;
            for (int64_t base = 0; base < count; base += width_limit) {
                int64_t width = std::min(width_limit, count - base);
                unsigned grid = static_cast<unsigned>((width + bundle->block - 1) / bundle->block);
                if (codes) {
                    bundle->Get("pq").Launch({&base_vectors, &encoded, &query, &cand.ptr, &off.ptr,
                                              &score.ptr, &member.ptr, &nb, &width, &base}, grid, stream);
                } else {
                    bundle->Get(ivf ? "ivf_flat" : "distance").Launch(
                        {&base_vectors, &query, &cand.ptr, &off.ptr, &score.ptr, &member.ptr, &nb, &width, &base},
                        grid, stream);
                }
                Select(*bundle, score.ptr, member.ptr, local_score.ptr, local_ids.ptr, width, stream);
                Check(hgStreamSynchronize(stream));
                if (first) {
                    Check(hgMemcpyDtoD(best_score.ptr, local_score.ptr, k * sizeof(float)));
                    Check(hgMemcpyDtoD(best_ids.ptr, local_ids.ptr, k * sizeof(int64_t)));
                    first = false;
                } else {
                    Check(hgMemcpyDtoD(merge_score.ptr, best_score.ptr, k * sizeof(float)));
                    Check(hgMemcpyDtoD(merge_score.ptr + k * sizeof(float), local_score.ptr, k * sizeof(float)));
                    Check(hgMemcpyDtoD(merge_ids.ptr, best_ids.ptr, k * sizeof(int64_t)));
                    Check(hgMemcpyDtoD(merge_ids.ptr + k * sizeof(int64_t), local_ids.ptr, k * sizeof(int64_t)));
                    Select(*bundle, merge_score.ptr, merge_ids.ptr, best_score.ptr, best_ids.ptr, 2 * k, stream);
                    Check(hgStreamSynchronize(stream));
                }
            }
            Check(hgMemcpyDtoH(distances + q * k, best_score.ptr, k * sizeof(float)));
            Check(hgMemcpyDtoH(ids + q * k, best_ids.ptr, k * sizeof(int64_t)));
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
