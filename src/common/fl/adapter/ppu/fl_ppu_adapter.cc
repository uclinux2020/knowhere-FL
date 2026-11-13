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

// ---------------------------------------------------------------------------
// PPU (Zhenwu/ZW810E) adapter for the FL backend. Implements FlDeviceOps on
// top of the PPU runtime, reusing the launch path validated in the FlagFit
// project (HIDDEN_PARAM ABI fix / hgLaunchKernelEx / HGlaunchConfig / device
// context binding - the BUG-H resolution chain).
//
// Retrieval kernels are FlagGems-composed Triton kernels AOT-compiled by
// FlagTree for the PPU backend; this adapter loads the compiled artifacts and
// launches them through the FlagFit launch wrapper.
//
// This file only compiles when the PPU SDK is available:
//   cmake .. -DWITH_FL=ON -DKNOWHERE_FL_WITH_PPU=ON -DPPU_SDK_ROOT=/opt/ppu
// Without the SDK the file compiles to an empty TU so the rest of the FL
// backend stays buildable everywhere.
// ---------------------------------------------------------------------------

#include "common/fl/fl_device.h"

#ifdef KNOWHERE_FL_WITH_PPU

#include <cstdlib>
#include <cstring>

#include "knowhere/log.h"

// Integration points to the FlagFit launch wrapper (see README-FL.md).
// The wrapper owns: device context binding, HGlaunchConfig construction,
// HIDDEN_PARAM injection, hgLaunchKernelEx invocation.
// TODO(PPU): replace these forward decls with the real FlagFit headers when
// integrating on the PPU dev box (bastion.aiops.baai.ac.cn environment).
extern "C" {
int hgDeviceCount();
int hgSetDevice(int device_id);
int hgDeviceSynchronize();
void* hgMemAlloc(size_t bytes);
int hgMemFree(void* ptr);
int hgMemcpyH2D(void* dst, const void* src, size_t bytes, void* stream);
int hgMemcpyD2H(void* dst, const void* src, size_t bytes, void* stream);
int hgMemcpyD2D(void* dst, const void* src, size_t bytes, void* stream);
void* hgStreamCreate();
int hgStreamDestroy(void* stream);
int hgStreamSynchronize(void* stream);
// AOT kernel registry produced by the FlagTree build step (fl_kernels.json):
// loads a compiled FlagGems-composition kernel by name.
void* hgAotKernelLoad(const char* name, const char* artifact_path);
int hgAotLaunchTopK(const void* kernel, const void* xb, long nb, const void* xq, long nq, long dim, long k,
                    int metric, float* out_dist, long* out_ids, void* stream);
}

namespace knowhere {
namespace fl {
namespace {

// TODO(PPU): artifact locations come from the FlagTree AOT build; wire into
// milvus.yaml-adjacent config once the kernel set stabilizes (Phase 1).
constexpr const char* kBruteForceTopKArtifact = "fl_kernels/brute_force_topk.ppuo";
constexpr const char* kKmeansAssignArtifact = "fl_kernels/kmeans_assign.ppuo";

int
ppu_brute_force_topk(const void* xb, int64_t nb, const void* xq, int64_t nq, int64_t dim, int64_t k, int metric,
                     float* out_dist, int64_t* out_ids, void* stream) {
    static void* kernel = hgAotKernelLoad("brute_force_topk", kBruteForceTopKArtifact);
    if (kernel == nullptr) {
        LOG_KNOWHERE_ERROR_ << "PPU adapter: brute_force_topk kernel not loaded";
        return -1;
    }
    return hgAotLaunchTopK(kernel, xb, nb, xq, nq, dim, k, metric, out_dist, out_ids, stream);
}

int
ppu_ivf_flat_scan(const void* xb, int64_t nb, int64_t dim, const void* xq, int64_t nq, const int64_t* cand_ids,
                  const int64_t* cand_offsets, int64_t k, int metric, float* out_dist, int64_t* out_ids,
                  void* stream) {
    // TODO(PPU): upload cand_ids/cand_offsets (H2D), then launch the
    // gather+distance+topk kernel. Until ivf_scan lands (Phase 1.3), return
    // -1 so the caller falls back to the CPU scan path (design D5).
    (void)xb; (void)nb; (void)dim; (void)xq; (void)nq;
    (void)cand_ids; (void)cand_offsets; (void)k; (void)metric;
    (void)out_dist; (void)out_ids; (void)stream;
    return -1;
}

int
ppu_kmeans_assign(const void* xb, int64_t nb, int64_t dim, const void* centroids, int64_t nlist, int metric,
                  int64_t* assign, void* stream) {
    // TODO(PPU): load the FlagTree AOT artifact of the FlagGems mm+argmin
    // composition (see python/fl_kernels/kmeans_composition.py --emit) and
    // launch it here via hgAotLaunchAssign. Until wired (Phase 1.2), return
    // -1 so Train/Add fall back to CPU routing (design D5).
    (void)xb; (void)nb; (void)dim; (void)centroids; (void)nlist; (void)metric;
    (void)assign; (void)stream;
    (void)kKmeansAssignArtifact;
    return -1;
}

const FlDeviceOps g_ppu_ops = {
    hgDeviceCount,     hgSetDevice,      hgDeviceSynchronize,
    hgMemAlloc,        [](void* p) { (void)hgMemFree(p); },
    hgMemcpyH2D,       hgMemcpyD2H,      hgMemcpyD2D,
    hgStreamCreate,    [](void* s) { (void)hgStreamDestroy(s); }, hgStreamSynchronize,
    ppu_brute_force_topk,
    ppu_ivf_flat_scan,
    ppu_kmeans_assign,
};

}  // namespace

// Called by Milvus/knowhere init glue when WITH_FL + PPU are enabled.
bool
RegisterPpuAdapter() {
    RegisterDeviceOps(&g_ppu_ops);
    LOG_KNOWHERE_INFO_ << "FL PPU adapter registered";
    return true;
}

}  // namespace fl
}  // namespace knowhere

#endif  // KNOWHERE_FL_WITH_PPU
