// Copyright (C) 2026 knowhere-FL contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

#ifndef KNOWHERE_FL_GRAPH_SEARCH_CPU_H
#define KNOWHERE_FL_GRAPH_SEARCH_CPU_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace knowhere {
namespace fl {

// ---------------------------------------------------------------------------
// Shared CPU greedy graph traversal (CAGRA-like search), header-only so both
// the host adapter (device-ops reference implementation) and the FL_CAGRA
// index node (no-adapter fallback path) use the exact same semantics. Keep
// this as the single source of truth for what the device graph_search kernel
// must reproduce.
//
// Contract: maintain a per-query candidate frontier of at most
// max(itopk, k) (dist, id) entries; repeatedly expand the best unexpanded
// candidate's graph neighbors; prune the frontier to bound memory; emit the
// top-k of the final frontier. metric: 0 = L2 (minimize), 1 = IP (maximize).
// ---------------------------------------------------------------------------

inline float
FlGraphDistance(const float* a, const float* b, int64_t dim, int metric) {
    if (metric == 1) {  // IP
        float s = 0.f;
        for (int64_t d = 0; d < dim; ++d) {
            s += a[d] * b[d];
        }
        return s;
    }
    float s = 0.f;  // L2 squared
    for (int64_t d = 0; d < dim; ++d) {
        const float diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
}

// out_dist/out_ids: nq*k, padded with worst-distance/-1 when the frontier
// yields fewer than k results. graph CSR rows may be padded with -1.
inline void
FlGraphSearchCpu(const float* xb, int64_t nb, int64_t dim, const int64_t* graph_ids, const int64_t* graph_offsets,
                 int64_t max_degree, const float* xq, int64_t nq, int64_t k, int64_t itopk, int metric,
                 float* out_dist, int64_t* out_ids) {
    const bool is_ip = (metric == 1);
    const float kWorst = is_ip ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
    const int64_t frontier_cap = std::max<int64_t>(itopk, k);

#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < nq; ++i) {
        if (nb <= 0) {  // empty graph: pad the whole row
            for (int64_t t = 0; t < k; ++t) {
                out_dist[i * k + t] = kWorst;
                out_ids[i * k + t] = -1;
            }
            continue;
        }
        const float* q = xq + i * dim;
        struct Cand {
            float dist;
            int64_t id;
            bool expanded;
        };
        std::vector<Cand> frontier;
        frontier.reserve(static_cast<size_t>(frontier_cap) + max_degree);
        std::vector<char> visited(static_cast<size_t>(nb), 0);

        auto better = [is_ip](const Cand& a, const Cand& b) {
            // deterministic tie-break on id so results are reproducible
            // (std::sort is not stable)
            if (a.dist != b.dist) {
                return is_ip ? (a.dist > b.dist) : (a.dist < b.dist);
            }
            return a.id < b.id;
        };
        auto try_push = [&](int64_t id, float dist) {
            if (visited[id]) {
                return;
            }
            visited[id] = 1;
            frontier.push_back({dist, id, false});
        };

        // deterministic seed: node 0 (mirrors CAGRA's fixed entry point when
        // no dedicated entry-point training is done)
        try_push(0, FlGraphDistance(q, xb, dim, metric));

        while (true) {
            int64_t best_idx = -1;
            for (size_t t = 0; t < frontier.size(); ++t) {
                if (frontier[t].expanded) {
                    continue;
                }
                if (best_idx < 0 || better(frontier[t], frontier[static_cast<size_t>(best_idx)])) {
                    best_idx = static_cast<int64_t>(t);
                }
            }
            if (best_idx < 0) {
                break;
            }
            Cand& cur = frontier[static_cast<size_t>(best_idx)];
            cur.expanded = true;
            const int64_t row_beg = graph_offsets[cur.id];
            const int64_t row_end = std::min<int64_t>(graph_offsets[cur.id + 1], row_beg + max_degree);
            for (int64_t e = row_beg; e < row_end; ++e) {
                const int64_t nid = graph_ids[e];
                if (nid < 0 || nid >= nb || visited[nid]) {
                    continue;
                }
                try_push(nid, FlGraphDistance(q, xb + nid * dim, dim, metric));
            }
            if (static_cast<int64_t>(frontier.size()) > frontier_cap * 2) {
                std::sort(frontier.begin(), frontier.end(), better);
                // dropped tail stays visited=1 -> effectively pruned; this is
                // the intended bounded-memory behavior (CAGRA prunes too)
                frontier.resize(static_cast<size_t>(frontier_cap));
            }
        }

        std::sort(frontier.begin(), frontier.end(), better);
        for (int64_t t = 0; t < k; ++t) {
            if (t < static_cast<int64_t>(frontier.size())) {
                out_dist[i * k + t] = frontier[static_cast<size_t>(t)].dist;
                out_ids[i * k + t] = frontier[static_cast<size_t>(t)].id;
            } else {
                out_dist[i * k + t] = kWorst;
                out_ids[i * k + t] = -1;
            }
        }
    }
}

}  // namespace fl
}  // namespace knowhere

#endif  // KNOWHERE_FL_GRAPH_SEARCH_CPU_H
