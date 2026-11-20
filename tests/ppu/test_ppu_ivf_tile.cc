// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Real-device gathered score tile tests; no CPU fallback.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/fl/adapter/ppu/fl_ppu_runtime.h"

namespace ppu = knowhere::fl::ppu;
namespace {
void
Require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
struct Stream {
    void* ptr = ppu::StreamCreate();
    Stream() {
        Require(ptr != nullptr, "Stream creation failed");
    }
    ~Stream() {
        ppu::StreamDestroy(ptr);
    }
};
struct Buffer {
    void* ptr;
    explicit Buffer(size_t bytes) : ptr(ppu::Alloc(bytes)) {
        Require(ptr != nullptr, "Device allocation failed");
    }
    ~Buffer() {
        ppu::Free(ptr);
    }
};
}  // namespace

int
main(int argc, char** argv) {
    try {
        Require(argc == 5, "Args: device dim k L2|IP");
        const int device = std::stoi(argv[1]);
        const int64_t dim = std::stoll(argv[2]), k = std::stoll(argv[3]);
        const std::string metric_name = argv[4];
        Require(dim > 0 && k > 0 && k <= 67 && (metric_name == "L2" || metric_name == "IP"), "Invalid shape");
        const int metric = metric_name == "IP";
        Require(ppu::SetDevice(device) == 0, "Device binding failed");
        constexpr int64_t nb = 2051, nq = 17;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> random(-1, 1);
        std::vector<float> xb(nb * dim), xq(nq * dim), distances(nq * k);
        std::vector<int64_t> ids(nq * k), perm(nb);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        Buffer base(xb.size() * sizeof(float)), queries(xq.size() * sizeof(float));
        Stream stream;
        int cases = 0;
        // Large-offset L2 probes cancellation. IP at ~1e11 can round distinct
        // double scores to the same FP32 value, so strict double ordering is not
        // an appropriate contract for that case; retain bounded IP and exact ties.
        for (int pattern = 0; pattern < (metric == 0 ? 3 : 2); ++pattern) {
            for (auto& v : xb) v = pattern == 1 ? 1.f : random(rng) + (pattern == 2 ? 10000.f : 0.f);
            for (auto& v : xq) v = random(rng) + (pattern == 2 ? 10000.f : 0.f);
            Require(ppu::CopyH2D(base.ptr, xb.data(), xb.size() * sizeof(float), stream.ptr) == 0,
                    "Base upload failed");
            Require(ppu::CopyH2D(queries.ptr, xq.data(), xq.size() * sizeof(float), stream.ptr) == 0,
                    "Query upload failed");
            for (bool explicit_stream : {false, true}) {
                for (int64_t width : {int64_t{0}, int64_t{1}, k - 1, k, int64_t{31}, int64_t{32}, int64_t{33},
                                      int64_t{127}, int64_t{128}, int64_t{129}, int64_t{1025}, nb}) {
                    std::vector<int64_t> candidates, offsets{0};
                    for (int64_t q = 0; q < nq; ++q) {
                        // Empty and short rows share batches with longer rows.
                        const int64_t count = q % 3 == 0 ? 0 : q % 3 == 1 ? std::min<int64_t>(width, 2) : width;
                        candidates.insert(candidates.end(), perm.begin(), perm.begin() + count);
                        offsets.push_back(candidates.size());
                    }
                    Require(
                        ppu::Search(base.ptr, nb, queries.ptr, nq, dim, k, metric, candidates.data(), offsets.data(),
                                    distances.data(), ids.data(), explicit_stream ? stream.ptr : nullptr) == 0,
                        "Device IVF failed; no fallback permitted");
                    for (int64_t q = 0; q < nq; ++q) {
                        std::vector<std::pair<double, int64_t>> truth;
                        for (int64_t pos = offsets[q]; pos < offsets[q + 1]; ++pos) {
                            const int64_t id = candidates[pos];
                            double score = 0;
                            for (int64_t d = 0; d < dim; ++d) {
                                const double a = xq[q * dim + d], b = xb[id * dim + d];
                                score += metric ? a * b : (a - b) * (a - b);
                            }
                            truth.emplace_back(metric ? -score : score, id);
                        }
                        std::sort(truth.begin(), truth.end());
                        for (int64_t rank = 0; rank < k; ++rank) {
                            const int64_t pos = q * k + rank;
                            if (rank >= static_cast<int64_t>(truth.size())) {
                                Require(ids[pos] == -1 && std::isinf(distances[pos]) &&
                                            (metric ? distances[pos] < 0 : distances[pos] > 0),
                                        "Invalid padding");
                            } else {
                                Require(ids[pos] == truth[rank].second, "Ordered ID differs from double oracle");
                                const double score = metric ? -truth[rank].first : truth[rank].first;
                                Require(std::isfinite(distances[pos]) &&
                                            std::abs(distances[pos] - score) <= 1e-4 * (1 + std::abs(score)),
                                        "Score differs from double oracle");
                            }
                        }
                    }
                    ++cases;
                }
            }
        }
        std::cout << "PASS: " << cases
                  << " IVF cases, random/ties (L2 also common offsets), empty/short/multi-tile, both streams\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
