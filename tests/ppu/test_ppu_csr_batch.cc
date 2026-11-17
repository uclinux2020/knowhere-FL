// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Real-device CSR streaming regression; no CPU fallback.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
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
struct Buffer {
    void* ptr;
    explicit Buffer(size_t bytes) : ptr(ppu::Alloc(bytes)) {
        Require(ptr != nullptr, "device allocation failed");
    }
    ~Buffer() {
        ppu::Free(ptr);
    }
};
}  // namespace

int
main(int argc, char** argv) {
    try {
        const int device = argc > 1 ? std::stoi(argv[1]) : 0;
        Require(ppu::SetDevice(device) == 0, "device binding failed");
        constexpr int64_t nb = 5003, nq = 83, dim = 5, k = 3, ksub = 8;
        // Force multiple batches, with nonzero CSR origins and a partial tail.
        setenv("KNOWHERE_FL_PPU_WORKSPACE_MB", "1", 1);
        setenv("KNOWHERE_FL_PPU_PROFILE", "1", 1);
        std::vector<float> xb(nb * dim), xq(nq * dim), book(ksub * dim), decoded(nb * dim), distances(nq * k);
        std::vector<uint8_t> codes(nb);
        std::vector<int64_t> candidates, offsets{0}, ids(nq * k);
        for (size_t i = 0; i < xb.size(); ++i) xb[i] = std::sin(static_cast<float>(i) * 1.37f);
        for (size_t i = 0; i < xq.size(); ++i) xq[i] = std::cos(static_cast<float>(i) * .73f);
        for (size_t i = 0; i < book.size(); ++i) book[i] = std::sin(static_cast<float>(i) * .37f);
        for (int64_t id = 0; id < nb; ++id) {
            codes[id] = id % ksub;
            std::copy_n(book.data() + codes[id] * dim, dim, decoded.data() + id * dim);
        }
        for (int64_t q = 0; q < nq; ++q) {
            const int64_t count = q < 22 ? 0 : q < 44 ? 2 : q < 66 ? 17 : 4097;
            for (int64_t j = 0; j < count; ++j) candidates.push_back((j * 37 + q * 13) % nb);
            offsets.push_back(candidates.size());
        }
        Buffer base(xb.size() * sizeof(float)), query(xq.size() * sizeof(float));
        Buffer codebook(book.size() * sizeof(float)), encoded(codes.size());
        Require(ppu::CopyH2D(base.ptr, xb.data(), xb.size() * sizeof(float), nullptr) == 0, "base copy failed");
        Require(ppu::CopyH2D(query.ptr, xq.data(), xq.size() * sizeof(float), nullptr) == 0, "query copy failed");
        Require(ppu::CopyH2D(codebook.ptr, book.data(), book.size() * sizeof(float), nullptr) == 0, "book copy failed");
        Require(ppu::CopyH2D(encoded.ptr, codes.data(), codes.size(), nullptr) == 0, "code copy failed");
        for (bool explicit_stream : {false, true}) {
            void* stream = explicit_stream ? ppu::StreamCreate() : nullptr;
            Require(!explicit_stream || stream != nullptr, "stream creation failed");
            for (bool pq : {false, true}) {
                for (int metric : {0, 1}) {
                    Require(ppu::Search(pq ? codebook.ptr : base.ptr, nb, query.ptr, nq, dim, k, metric,
                                        candidates.data(), offsets.data(), distances.data(), ids.data(), stream,
                                        pq ? encoded.ptr : nullptr, pq ? 1 : 0, pq ? ksub : 0, pq ? dim : 0) == 0,
                            "batched CSR search failed");
                    const auto& vectors = pq ? decoded : xb;
                    for (int64_t q = 0; q < nq; ++q) {
                        std::vector<std::pair<float, int64_t>> truth;
                        for (int64_t j = offsets[q]; j < offsets[q + 1]; ++j) {
                            const int64_t id = candidates[j];
                            float score = 0;
                            for (int64_t d = 0; d < dim; ++d) {
                                const float a = xq[q * dim + d], b = vectors[id * dim + d];
                                score += metric == 0 ? (a - b) * (a - b) : a * b;
                            }
                            truth.emplace_back(metric == 0 ? score : -score, id);
                        }
                        std::sort(truth.begin(), truth.end());
                        for (int64_t rank = 0; rank < k; ++rank) {
                            const auto pos = q * k + rank;
                            if (rank >= static_cast<int64_t>(truth.size())) {
                                Require(ids[pos] == -1 && std::isinf(distances[pos]), "invalid empty/tail padding");
                            } else {
                                const float expected = metric == 0 ? truth[rank].first : -truth[rank].first;
                                Require(ids[pos] == truth[rank].second, "rebased CSR returned wrong ID");
                                Require(std::abs(distances[pos] - expected) <= 1e-4f * (1 + std::abs(expected)),
                                        "rebased CSR returned wrong score");
                            }
                        }
                    }
                }
            }
            ppu::StreamDestroy(stream);
        }
        std::cout << "PASS: bounded CSR batches, IVF/PQ L2/IP, uneven/empty lists, partial tail, both streams\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
