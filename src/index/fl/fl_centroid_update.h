// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace knowhere::fl {

// Stable grouping preserves the original row order of every cluster. Each
// independent dimension therefore keeps the serial double accumulation order,
// unlike thread-local partial sums or floating-point atomics.
inline void
AccumulateCentroids(const float* x, const int64_t* owner, int64_t nb, int64_t dim, int64_t nlist,
                    std::vector<double>& sums, std::vector<int64_t>& counts) {
    sums.assign(static_cast<size_t>(nlist) * dim, 0.0);
    counts.assign(nlist, 0);
    // Avoid grouping/allocation overhead for small or single-threaded updates.
    bool parallel = false;
#ifdef _OPENMP
    parallel = omp_get_max_threads() > 1 && !omp_in_parallel() && nb > 0 && dim >= 16 && nb >= 1048576 / dim;
#endif
    if (!parallel) {
        for (int64_t j = 0; j < nb; ++j) {
            const int64_t c = owner[j];
            ++counts[c];
            for (int64_t d = 0; d < dim; ++d) {
                sums[static_cast<size_t>(c) * dim + d] += x[j * dim + d];
            }
        }
        return;
    }
    for (int64_t j = 0; j < nb; ++j) ++counts[owner[j]];
    std::vector<int64_t> offsets(nlist + 1, 0);
    for (int64_t c = 0; c < nlist; ++c) offsets[c + 1] = offsets[c] + counts[c];
    auto cursor = offsets;
    std::vector<int64_t> rows(nb);
    for (int64_t j = 0; j < nb; ++j) rows[cursor[owner[j]]++] = j;

    // Split wide/skewed clusters across dimension tiles, without splitting a
    // dimension's reduction. Bound teams to avoid nested segment oversubscription.
    constexpr int64_t tile = 128;
    const int64_t tiles = (dim + tile - 1) / tile;
#ifdef _OPENMP
    const int threads = static_cast<int>(std::min<int64_t>({8, omp_get_max_threads(), nlist * tiles}));
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
    for (int64_t task = 0; task < nlist * tiles; ++task) {
        const int64_t c = task / tiles;
        const int64_t start = (task % tiles) * tile;
        const int64_t end = std::min(dim, start + tile);
        double* sum = sums.data() + c * dim;
        for (int64_t p = offsets[c]; p < offsets[c + 1]; ++p) {
            const float* row = x + rows[p] * dim;
            for (int64_t d = start; d < end; ++d) sum[d] += row[d];
        }
    }
}

}  // namespace knowhere::fl
