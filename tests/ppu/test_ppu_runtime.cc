// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Standalone real-PPU contract tests: no simulated device and no CPU fallback.
#include "common/fl/adapter/ppu/fl_ppu_runtime.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ppu = knowhere::fl::ppu;
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct DeviceBuffer {
    void* ptr;
    explicit DeviceBuffer(size_t bytes) : ptr(ppu::Alloc(bytes)) { Require(ptr != nullptr, "device allocation failed"); }
    ~DeviceBuffer() { ppu::Free(ptr); }
};
float Score(const float* a, const float* b, int64_t dim, int metric) {
    float result = 0;
    for (int64_t d = 0; d < dim; ++d) result += metric == 0 ? (a[d] - b[d]) * (a[d] - b[d]) : a[d] * b[d];
    return result;
}
void Verify(const std::vector<float>& xb, const std::vector<float>& xq, int64_t nb, int64_t nq, int64_t dim,
            int64_t k, int metric, const std::vector<int64_t>& candidates, const std::vector<int64_t>& offsets,
            const std::vector<float>& distances, const std::vector<int64_t>& ids) {
    for (int64_t q = 0; q < nq; ++q) {
        std::vector<std::pair<float, int64_t>> truth;
        int64_t count = offsets.empty() ? nb : offsets[q + 1] - offsets[q];
        for (int64_t j = 0; j < count; ++j) {
            int64_t id = offsets.empty() ? j : candidates[offsets[q] + j];
            truth.emplace_back(Score(xq.data() + q * dim, xb.data() + id * dim, dim, metric), id);
        }
        std::sort(truth.begin(), truth.end(), [metric](auto a, auto b) {
            return a.first == b.first ? a.second < b.second : metric == 0 ? a.first < b.first : a.first > b.first;
        });
        for (int64_t rank = 0; rank < k; ++rank) {
            if (rank >= count) {
                Require(ids[q * k + rank] == -1 && std::isinf(distances[q * k + rank]), "invalid padding");
            } else {
                if (ids[q * k + rank] != truth[rank].second ||
                    std::abs(distances[q * k + rank] - truth[rank].first) > 1e-4f * (1 + std::abs(truth[rank].first))) {
                    std::cerr << "q=" << q << " rank=" << rank << " got " << ids[q * k + rank] << ':'
                              << distances[q * k + rank] << " expected " << truth[rank].second << ':' << truth[rank].first << '\n';
                    throw std::runtime_error("PPU scores/IDs mismatch");
                }
            }
        }
    }
}
int main(int argc, char** argv) {
    try {
        int device = argc > 1 ? std::stoi(argv[1]) : 0;
        Require(ppu::DeviceCount() > device, "PPU device unavailable");
        Require(ppu::SetDevice(device) == 0, "PPU context binding failed");
        const bool default_stream = argc > 5 && std::string(argv[5]) == "default-stream";
        void* stream = default_stream ? nullptr : ppu::StreamCreate();
        Require(default_stream || stream != nullptr, "PPU stream creation failed");
        const int64_t nb = 67, nq = 3;
        const int64_t dim = argc > 2 ? std::stoll(argv[2]) : 5;
        const int64_t k = argc > 3 ? std::stoll(argv[3]) : 3;
        const int64_t m = argc > 4 ? std::stoll(argv[4]) : 1;
        Require(dim > 0 && k > 0 && k <= nb && m > 0 && dim % m == 0, "invalid test shape");
        const int64_t dsub = dim / m;
        std::vector<float> xb(nb * dim), xq(nq * dim), distances(nq * k);
        std::vector<int64_t> ids(nq * k);
        for (size_t i = 0; i < xb.size(); ++i) xb[i] = std::sin(static_cast<float>(i) * 1.37f);
        for (size_t i = 0; i < xq.size(); ++i) xq[i] = std::cos(static_cast<float>(i) * .73f);
        DeviceBuffer base(xb.size() * sizeof(float)), query(xq.size() * sizeof(float));
        Require(ppu::CopyH2D(base.ptr, xb.data(), xb.size() * sizeof(float), stream) == 0, "base H2D failed");
        Require(ppu::CopyH2D(query.ptr, xq.data(), xq.size() * sizeof(float), stream) == 0, "query H2D failed");
        for (int metric : {0, 1}) {
            Require(ppu::Search(base.ptr, nb, query.ptr, nq, dim, k, metric, nullptr, nullptr,
                                distances.data(), ids.data(), stream) == 0, "brute-force device search failed");
            Verify(xb, xq, nb, nq, dim, k, metric, {}, {}, distances, ids);
            // Large common offsets must not break L2 norm decomposition.
            if (metric == 0) {
                for (size_t i = 0; i < xb.size(); ++i) xb[i] = 10000.f + static_cast<float>(i % 31) * .125f;
                for (size_t i = 0; i < xq.size(); ++i) xq[i] = 10000.f + static_cast<float>(i % 29) * .125f;
                Require(ppu::CopyH2D(base.ptr, xb.data(), xb.size() * sizeof(float), stream) == 0, "offset base H2D failed");
                Require(ppu::CopyH2D(query.ptr, xq.data(), xq.size() * sizeof(float), stream) == 0, "offset query H2D failed");
                Require(ppu::Search(base.ptr, nb, query.ptr, nq, dim, k, metric, nullptr, nullptr,
                                    distances.data(), ids.data(), stream) == 0, "large-offset search failed");
                Verify(xb, xq, nb, nq, dim, k, metric, {}, {}, distances, ids);
                for (size_t i = 0; i < xb.size(); ++i) xb[i] = std::sin(static_cast<float>(i) * 1.37f);
                for (size_t i = 0; i < xq.size(); ++i) xq[i] = std::cos(static_cast<float>(i) * .73f);
                Require(ppu::CopyH2D(base.ptr, xb.data(), xb.size() * sizeof(float), stream) == 0, "restore base failed");
                Require(ppu::CopyH2D(query.ptr, xq.data(), xq.size() * sizeof(float), stream) == 0, "restore query failed");
            }
            std::vector<int64_t> cand = {4, 1, 66, 2, 13, 41};
            std::vector<int64_t> offsets = {0, 0, 2, 6};
            Require(ppu::Search(base.ptr, nb, query.ptr, nq, dim, k, metric, cand.data(), offsets.data(),
                                distances.data(), ids.data(), stream) == 0, "IVF device search failed");
            Verify(xb, xq, nb, nq, dim, k, metric, cand, offsets, distances, ids);
            // Invalid CSR must fail, never silently launch on unrelated data.
            int64_t invalid[] = {0, 3, 2, 6};
            Require(ppu::Search(base.ptr, nb, query.ptr, nq, dim, k, metric, cand.data(), invalid,
                                distances.data(), ids.data(), stream) != 0, "invalid CSR accepted");
            // Tie-breaking must survive multiple score chunks and merge stages.
            std::fill(xb.begin(), xb.end(), 1);
            Require(ppu::CopyH2D(base.ptr, xb.data(), xb.size() * sizeof(float), stream) == 0, "tie H2D failed");
            Require(ppu::Search(base.ptr, nb, query.ptr, nq, dim, k, metric, nullptr, nullptr,
                                distances.data(), ids.data(), stream) == 0, "tie device search failed");
            Verify(xb, xq, nb, nq, dim, k, metric, {}, {}, distances, ids);
            for (size_t i = 0; i < xb.size(); ++i) xb[i] = std::sin(static_cast<float>(i) * 1.37f);
            Require(ppu::CopyH2D(base.ptr, xb.data(), xb.size() * sizeof(float), stream) == 0, "restore H2D failed");
            // PQ scan correctness against explicitly decoded codebooks.
            constexpr int64_t ksub = 8;
            std::vector<float> cb(m * ksub * dsub), decoded(nb * dim);
            std::vector<uint8_t> codes(nb * m);
            for (size_t i = 0; i < cb.size(); ++i) cb[i] = std::sin(static_cast<float>(i) * .37f);
            for (int64_t i = 0; i < nb; ++i) {
                for (int64_t s = 0; s < m; ++s) {
                    codes[i * m + s] = (i + s) % ksub;
                    std::copy_n(cb.data() + (s * ksub + codes[i * m + s]) * dsub, dsub,
                                decoded.data() + i * dim + s * dsub);
                }
            }
            DeviceBuffer book(cb.size() * sizeof(float)), encoded(codes.size());
            Require(ppu::CopyH2D(book.ptr, cb.data(), cb.size() * sizeof(float), stream) == 0, "codebook H2D failed");
            Require(ppu::CopyH2D(encoded.ptr, codes.data(), codes.size(), stream) == 0, "codes H2D failed");
            Require(ppu::Search(book.ptr, nb, query.ptr, nq, dim, k, metric, cand.data(), offsets.data(),
                                distances.data(), ids.data(), stream, encoded.ptr, m, ksub, dsub) == 0, "PQ search failed");
            Verify(decoded, xq, nb, nq, dim, k, metric, cand, offsets, distances, ids);
            // LUTs must be regenerated, not cached solely by device address.
            for (auto& value : cb) value += .25f;
            for (int64_t i = 0; i < nb; ++i) {
                for (int64_t s = 0; s < m; ++s) {
                    std::copy_n(cb.data() + (s * ksub + codes[i * m + s]) * dsub, dsub,
                                decoded.data() + i * dim + s * dsub);
                }
            }
            Require(ppu::CopyH2D(book.ptr, cb.data(), cb.size() * sizeof(float), stream) == 0,
                    "updated codebook H2D failed");
            Require(ppu::Search(book.ptr, nb, query.ptr, nq, dim, k, metric, cand.data(), offsets.data(),
                                distances.data(), ids.data(), stream, encoded.ptr, m, ksub, dsub) == 0,
                    "updated PQ search failed");
            Verify(decoded, xq, nb, nq, dim, k, metric, cand, offsets, distances, ids);
            std::vector<int64_t> assign(nq);
            Require(ppu::Assign(query.ptr, nq, dim, base.ptr, nb, metric, assign.data(), stream) == 0, "assignment failed");
            std::vector<float> assignment_distances(nq);
            for (int64_t q = 0; q < nq; ++q) assignment_distances[q] = Score(xq.data() + q * dim, xb.data() + assign[q] * dim, dim, metric);
            Verify(xb, xq, nb, nq, dim, 1, metric, {}, {}, assignment_distances, assign);
        }
        ppu::StreamDestroy(stream);
        ppu::Free(nullptr);
        std::cout << "PASS: real PPU L2/IP brute-force, chunk merge, IVF, PQ, assignment, ties, padding, invalid CSR, large-offset L2\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
