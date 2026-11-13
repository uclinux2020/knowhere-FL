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

#ifndef FL_KMEANS_CPU_H
#define FL_KMEANS_CPU_H

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "knowhere/log.h"

namespace knowhere {
namespace fl {

// ---------------------------------------------------------------------------
// Generic CPU Lloyd kmeans over row-major fp32 data.
//
// Shared by FL index nodes for training (IVF_PQ subspace codebooks, and any
// path where the device kmeans_assign route is not wired). Assignment uses
// the metric encoded as: 0 = L2 (squared), 1 = IP. Empty clusters are
// re-seeded from a rotated sample. Deterministic for a given seed.
//
//   x         : nb x dim row-major input
//   centroids : out, nlist x dim (must be preallocated)
//   owner     : optional out, nb nearest-centroid ids (may be nullptr)
// ---------------------------------------------------------------------------
inline void
LloydKmeansCpu(const float* x, int64_t nb, int64_t dim, int64_t nlist, int metric_code, float* centroids,
               int64_t* owner_out = nullptr, int max_iters = 25, uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::vector<int64_t> perm(nb);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int64_t l = 0; l < nlist; ++l) {
        std::memcpy(centroids + l * dim, x + perm[l] * dim, dim * sizeof(float));
    }

    // Assignment cost model: for L2, argmin_l ||v-c_l||^2 = argmin_l (||c_l||^2
    // - 2*v.c_l) -- the query norm ||v||^2 is a constant offset across l and
    // never changes the selection, so it is eliminated outright. Centroid
    // norms are recomputed once per iteration (they change after each
    // update). This mirrors exactly what the device kmeans_assign kernel
    // computes via its mm+argmin composition.
    std::vector<float> cent_norms(static_cast<size_t>(nlist), 0.f);
    auto refresh_centroid_norms = [&] {
        if (metric_code != 0) {
            return;
        }
#pragma omp parallel for schedule(static)
        for (int64_t l = 0; l < nlist; ++l) {
            const float* c = centroids + l * dim;
            float s = 0.f;
            for (int64_t d = 0; d < dim; ++d) {
                s += c[d] * c[d];
            }
            cent_norms[static_cast<size_t>(l)] = s;
        }
    };

    auto nearest = [&](const float* v) {
        int64_t best = 0;
        float best_score = metric_code == 1 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (int64_t l = 0; l < nlist; ++l) {
            const float* c = centroids + l * dim;
            float dot = 0.f;
            for (int64_t d = 0; d < dim; ++d) {
                dot += v[d] * c[d];
            }
            const float score =
                metric_code == 1 ? dot : (cent_norms[static_cast<size_t>(l)] - 2.f * dot);
            if ((metric_code == 1 && score > best_score) || (metric_code == 0 && score < best_score)) {
                best_score = score;
                best = l;
            }
        }
        return best;
    };

    std::vector<int64_t> owner(nb, -1);
    std::vector<int64_t> cur(nb, -1);
    for (int iter = 0; iter < max_iters; ++iter) {
        refresh_centroid_norms();
#pragma omp parallel for schedule(dynamic)
        for (int64_t j = 0; j < nb; ++j) {
            cur[j] = nearest(x + j * dim);
        }
        bool changed = false;
#pragma omp parallel for schedule(static) reduction(| : changed)
        for (int64_t j = 0; j < nb; ++j) {
            if (cur[j] != owner[j]) {
                owner[j] = cur[j];
                changed = true;
            }
        }
        std::vector<double> sums(static_cast<size_t>(nlist) * dim, 0.0);
        std::vector<int64_t> counts(nlist, 0);
        for (int64_t j = 0; j < nb; ++j) {
            const int64_t c = owner[j];
            ++counts[c];
            for (int64_t d = 0; d < dim; ++d) {
                sums[static_cast<size_t>(c) * dim + d] += x[j * dim + d];
            }
        }
        for (int64_t l = 0; l < nlist; ++l) {
            if (counts[l] == 0) {
                std::memcpy(centroids + l * dim, x + perm[(l + iter + 1) % nb] * dim, dim * sizeof(float));
            } else {
                for (int64_t d = 0; d < dim; ++d) {
                    centroids[l * dim + d] = static_cast<float>(sums[static_cast<size_t>(l) * dim + d] / counts[l]);
                }
            }
        }
        if (!changed) {
            break;
        }
    }
    if (owner_out != nullptr) {
        std::memcpy(owner_out, owner.data(), nb * sizeof(int64_t));
    }
}

}  // namespace fl
}  // namespace knowhere

#endif  // FL_KMEANS_CPU_H
