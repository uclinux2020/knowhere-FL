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

#include "common/fl/fl_host_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "common/fl/fl_device.h"
#include "common/fl/fl_graph_search_cpu.h"
#include "knowhere/log.h"

namespace knowhere {
namespace fl {
namespace {

// ---------------------------------------------------------------------------
// Host implementation of the FlDeviceOps contract.
// metric encoding (see fl_device.h): 0 = L2 (smaller is better), 1 = IP
// (larger is better).
// ---------------------------------------------------------------------------

int
host_device_count() {
    // Pretend there is exactly one "device" so the device path is exercised.
    return 1;
}

int
host_set_device(int /*device_id*/) {
    return 0;
}

int
host_device_sync() {
    return 0;
}

void*
host_malloc(size_t bytes) {
    return std::malloc(bytes == 0 ? 1 : bytes);
}

void
host_free(void* ptr) {
    std::free(ptr);
}

int
host_memcpy(void* dst, const void* src, size_t bytes, void* /*stream*/) {
    std::memcpy(dst, src, bytes);
    return 0;
}

void*
host_stream_create() {
    return nullptr;
}

void
host_stream_destroy(void* /*stream*/) {}

int
host_stream_sync(void* /*stream*/) {
    return 0;
}

inline float
host_distance(const float* a, const float* b, int64_t dim, int metric) {
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int64_t d = 0; d < dim; ++d) {
        dot += a[d] * b[d];
        if (metric == 0) {
            na += a[d] * a[d];
            nb += b[d] * b[d];
        }
    }
    return metric == 1 ? dot : (na + nb - 2.f * dot);
}

// topk over a candidate list defined by a lambda id accessor. Shared by
// brute_force_topk (all ids) and ivf_flat_scan (subset).
template <typename IdAt>
void
host_topk_over_candidates(const float* xb, int64_t dim, const float* xq, int64_t nq, const int64_t* cand_counts,
                          int64_t k, int metric, IdAt&& id_at, float* out_dist, int64_t* out_ids) {
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < nq; ++i) {
        const float* q = xq + i * dim;
        const int64_t nc = cand_counts[i];
        using Cand = std::pair<float, int64_t>;
        const bool is_ip = (metric == 1);
        // simple selection: keep an unsorted buffer of k best
        std::vector<Cand> best;
        best.reserve(static_cast<size_t>(k) + 1);
        float worst = is_ip ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (int64_t c = 0; c < nc; ++c) {
            const int64_t id = id_at(i, c);
            const float dist = host_distance(q, xb + id * dim, dim, metric);
            if ((int64_t)best.size() < k) {
                best.emplace_back(dist, id);
                if (is_ip) {
                    worst = std::min(worst, dist);
                } else {
                    worst = std::max(worst, dist);
                }
            } else if ((is_ip && dist > worst) || (!is_ip && dist < worst)) {
                // replace current worst
                auto it = std::min_element(best.begin(), best.end(),
                                           [is_ip](const Cand& a, const Cand& b) {
                                               return is_ip ? (a.first < b.first) : (a.first > b.first);
                                           });
                *it = {dist, id};
                worst = is_ip ? std::min_element(best.begin(), best.end(),
                                                 [](const Cand& a, const Cand& b) { return a.first < b.first; })
                                    ->first
                              : std::max_element(best.begin(), best.end(),
                                                 [](const Cand& a, const Cand& b) { return a.first < b.first; })
                                    ->first;
            }
        }
        std::sort(best.begin(), best.end(), [is_ip](const Cand& a, const Cand& b) {
            return is_ip ? (a.first > b.first) : (a.first < b.first);
        });
        for (int64_t t = 0; t < k; ++t) {
            if (t < (int64_t)best.size()) {
                out_dist[i * k + t] = best[t].first;
                out_ids[i * k + t] = best[t].second;
            } else {
                out_dist[i * k + t] = is_ip ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
                out_ids[i * k + t] = -1;
            }
        }
    }
}

int
host_brute_force_topk(const void* xb, int64_t nb, const void* xq, int64_t nq, int64_t dim, int64_t k, int metric,
                      float* out_dist, int64_t* out_ids, void* /*stream*/) {
    if (xb == nullptr || xq == nullptr || out_dist == nullptr || out_ids == nullptr) {
        return -1;
    }
    std::vector<int64_t> counts(nq, nb);
    host_topk_over_candidates(static_cast<const float*>(xb), dim, static_cast<const float*>(xq), nq, counts.data(), k,
                              metric, [](int64_t, int64_t c) { return c; }, out_dist, out_ids);
    return 0;
}

int
host_ivf_flat_scan(const void* xb, int64_t /*nb*/, int64_t dim, const void* xq, int64_t nq, const int64_t* cand_ids,
                   const int64_t* cand_offsets, int64_t k, int metric, float* out_dist, int64_t* out_ids,
                   void* /*stream*/) {
    if (xb == nullptr || xq == nullptr || cand_ids == nullptr || cand_offsets == nullptr || out_dist == nullptr ||
        out_ids == nullptr) {
        return -1;
    }
    std::vector<int64_t> counts(nq);
    for (int64_t i = 0; i < nq; ++i) {
        counts[i] = cand_offsets[i + 1] - cand_offsets[i];
    }
    host_topk_over_candidates(static_cast<const float*>(xb), dim, static_cast<const float*>(xq), nq, counts.data(), k,
                              metric,
                              [cand_ids, cand_offsets](int64_t i, int64_t c) { return cand_ids[cand_offsets[i] + c]; },
                              out_dist, out_ids);
    return 0;
}

int
host_kmeans_assign(const void* xb, int64_t nb, int64_t dim, const void* centroids, int64_t nlist, int metric,
                   int64_t* assign, void* /*stream*/) {
    if (xb == nullptr || centroids == nullptr || assign == nullptr || nlist <= 0) {
        return -1;
    }
    const float* base = static_cast<const float*>(xb);
    const float* cent = static_cast<const float*>(centroids);
    // Reference semantics for the device kernel: per-vector argmin over
    // centroids (argmax for IP). Pure mm + argmin composition, no atomics.
#pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < nb; ++j) {
        const float* v = base + j * dim;
        float best = metric == 1 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        int64_t best_id = 0;
        for (int64_t c = 0; c < nlist; ++c) {
            const float d = host_distance(v, cent + c * dim, dim, metric);
            if ((metric == 1 && d > best) || (metric == 0 && d < best)) {
                best = d;
                best_id = c;
            }
        }
        assign[j] = best_id;
    }
    return 0;
}

// Reference semantics for the device ADC kernel:
//   1) per-query LUT: lut[m][ks] = distance(q_sub_m, codebook[m][ks])
//      (device side computes this via mm against reshaped codebooks)
//   2) scan: dist(q, v) = sum_m lut[m][codes[v * m + m]]
// L2 LUT stores squared sub-distances; IP LUT stores sub dot products and the
// scan maximizes the sum.
int
host_ivf_pq_scan(const void* codebooks, int64_t m, int64_t ksub, int64_t dsub, const void* codes, int64_t /*nb*/,
                 const void* xq, int64_t nq, int64_t /*dim*/, const int64_t* cand_ids, const int64_t* cand_offsets,
                 int64_t k, int metric, float* out_dist, int64_t* out_ids, void* /*stream*/) {
    if (codebooks == nullptr || codes == nullptr || xq == nullptr || cand_ids == nullptr || cand_offsets == nullptr ||
        out_dist == nullptr || out_ids == nullptr || m <= 0 || ksub <= 0 || dsub <= 0) {
        return -1;
    }
    const float* cb = static_cast<const float*>(codebooks);
    const uint8_t* pq_codes = static_cast<const uint8_t*>(codes);
    const float* queries = static_cast<const float*>(xq);

#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < nq; ++i) {
        const float* q = queries + i * m * dsub;
        // 1) build LUT for this query
        std::vector<float> lut(static_cast<size_t>(m) * ksub);
        for (int64_t s = 0; s < m; ++s) {
            const float* q_sub = q + s * dsub;
            const float* cb_s = cb + s * ksub * dsub;
            for (int64_t ks = 0; ks < ksub; ++ks) {
                lut[static_cast<size_t>(s) * ksub + ks] = host_distance(q_sub, cb_s + ks * dsub, dsub, metric);
            }
        }
        // 2) ADC scan over candidates
        using Cand = std::pair<float, int64_t>;
        const bool is_ip = (metric == 1);
        std::vector<Cand> best;
        best.reserve(static_cast<size_t>(k) + 1);
        for (int64_t c = cand_offsets[i]; c < cand_offsets[i + 1]; ++c) {
            const int64_t id = cand_ids[c];
            const uint8_t* vc = pq_codes + static_cast<size_t>(id) * m;
            float dist = 0.f;
            for (int64_t s = 0; s < m; ++s) {
                dist += lut[static_cast<size_t>(s) * ksub + vc[s]];
            }
            best.emplace_back(dist, id);
        }
        const int64_t nk = std::min<int64_t>(k, static_cast<int64_t>(best.size()));
        std::partial_sort(best.begin(), best.begin() + nk, best.end(), [is_ip](const Cand& a, const Cand& b) {
            return is_ip ? (a.first > b.first) : (a.first < b.first);
        });
        for (int64_t t = 0; t < k; ++t) {
            if (t < nk) {
                out_dist[i * k + t] = best[t].first;
                out_ids[i * k + t] = best[t].second;
            } else {
                out_dist[i * k + t] = is_ip ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
                out_ids[i * k + t] = -1;
            }
        }
    }
    return 0;
}

// Reference greedy graph traversal (CAGRA-like search). The semantics
// contract for device kernels lives in common/fl/fl_graph_search_cpu.h; the
// device graph_search kernel must reproduce it exactly.
int
host_graph_search(const void* xb, int64_t nb, int64_t dim, const int64_t* graph_ids, const int64_t* graph_offsets,
                  int64_t max_degree, const void* xq, int64_t nq, int64_t k, int64_t itopk, int metric,
                  float* out_dist, int64_t* out_ids, void* /*stream*/) {
    if (xb == nullptr || graph_ids == nullptr || graph_offsets == nullptr || xq == nullptr || out_dist == nullptr ||
        out_ids == nullptr || nb <= 0 || dim <= 0 || k <= 0 || max_degree <= 0) {
        return -1;
    }
    FlGraphSearchCpu(static_cast<const float*>(xb), nb, dim, graph_ids, graph_offsets, max_degree,
                     static_cast<const float*>(xq), nq, k, itopk, metric, out_dist, out_ids);
    return 0;
}

const FlDeviceOps g_host_ops = {
    host_device_count, host_set_device, host_device_sync,
    host_malloc,       host_free,       host_memcpy,
    host_memcpy,       host_memcpy,  // d2h, d2d
    host_stream_create, host_stream_destroy, host_stream_sync,
    host_brute_force_topk,
    host_ivf_flat_scan,
    host_kmeans_assign,
    host_ivf_pq_scan,
    host_graph_search,
};

}  // namespace

bool
RegisterHostAdapter() {
    if (GetDeviceOps() != nullptr) {
        LOG_KNOWHERE_WARNING_ << "FL adapter already registered; host adapter not installed";
        return false;
    }
    RegisterDeviceOps(&g_host_ops);
    LOG_KNOWHERE_INFO_ << "FL host adapter registered (device path emulated on CPU)";
    return true;
}

void
UnregisterHostAdapter() {
    if (GetDeviceOps() == &g_host_ops) {
        RegisterDeviceOps(nullptr);
    }
}

}  // namespace fl
}  // namespace knowhere
