// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Real-device regression for k -> smallest compiled k' >= k resolution.
// Usage: knowhere_ppu_kernel_k_test <device> <bucket_dir> <exact_dir>
//   bucket_dir: bundles whose k differ from the requested k (e.g. k20/64/128/256)
//   exact_dir:  bundles compiled for exactly the requested k (oracle, e.g. k50/100/200)
// A truncated k' result must equal the exact-k bundle bit for bit; no CPU fallback.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/fl/adapter/ppu/fl_ppu_runtime.h"

namespace ppu = knowhere::fl::ppu;
namespace {
constexpr int64_t kDim = 1024;
constexpr int kIp = 1;

void
Require(bool ok, const std::string& message) {
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
struct Result {
    std::vector<float> d;
    std::vector<int64_t> i;
};

Result
Run(const char* dir, const Buffer& base, int64_t nb, const Buffer& query, int64_t nq, int64_t k,
    const std::vector<int64_t>* cand = nullptr, const std::vector<int64_t>* off = nullptr, int* rc_out = nullptr) {
    setenv("KNOWHERE_FL_PPU_KERNEL_DIR", dir, 1);
    Result r{std::vector<float>(nq * k, 12345.f), std::vector<int64_t>(nq * k, -7)};
    const int rc = ppu::Search(base.ptr, nb, query.ptr, nq, kDim, k, kIp, cand ? cand->data() : nullptr,
                               off ? off->data() : nullptr, r.d.data(), r.i.data(), nullptr);
    if (rc_out) {
        *rc_out = rc;
    } else {
        Require(rc == 0, "device search failed for k=" + std::to_string(k) + " dir=" + dir);
    }
    return r;
}

bool
SameBits(const Result& a, const Result& b) {
    return a.i == b.i && std::memcmp(a.d.data(), b.d.data(), a.d.size() * sizeof(float)) == 0;
}

// Ranks must follow the kernel's total order: score descending (IP), then smaller ID.
void
CheckOrder(const Result& r, int64_t nq, int64_t k, const std::string& tag) {
    for (int64_t q = 0; q < nq; ++q)
        for (int64_t j = 1; j < k; ++j) {
            const float a = r.d[q * k + j - 1], b = r.d[q * k + j];
            const int64_t ia = r.i[q * k + j - 1], ib = r.i[q * k + j];
            if (ib < 0)
                continue;
            Require(ia >= 0 && (a > b || (a == b && ia < ib)), tag + ": rank order violated");
        }
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        Require(argc == 4, "usage: <device> <bucket_dir> <exact_dir>");
        Require(ppu::SetDevice(std::stoi(argv[1])) == 0, "device binding failed");
        const char* bucket = argv[2];
        const char* exact = argv[3];
        constexpr int64_t nb = 118592, nq = 37;
        std::vector<float> xb(nb * kDim), xq(nq * kDim);
        uint64_t state = 0x9E3779B97F4A7C15ULL;
        auto next = [&]() {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0f - 0.5f;
        };
        for (auto& v : xb) v = next();
        for (auto& v : xq) v = next();
        // Exact duplicates force score ties that only the ID tie-break resolves;
        // put them near query 0 so they land inside its top ranks.
        for (int64_t r = 0; r < 64; ++r) {
            float* row = xb.data() + (1000 + 997 * r) * kDim;
            for (int64_t c = 0; c < kDim; ++c) row[c] = xq[c] * (r < 32 ? 1.0f : 0.75f);
        }
        Buffer base(xb.size() * sizeof(float)), query(xq.size() * sizeof(float));
        Require(ppu::CopyH2D(base.ptr, xb.data(), xb.size() * sizeof(float), nullptr) == 0, "base copy failed");
        Require(ppu::CopyH2D(query.ptr, xq.data(), xq.size() * sizeof(float), nullptr) == 0, "query copy failed");

        int checks = 0;
        // 1) Brute force: truncated bucket result == exact-k bundle, bit for bit.
        for (int64_t k : {50, 100, 200}) {
            auto t = Run(bucket, base, nb, query, nq, k);
            auto e = Run(exact, base, nb, query, nq, k);
            Require(SameBits(t, e), "brute-force k=" + std::to_string(k) + " differs from exact bundle");
            CheckOrder(t, nq, k, "bf k=" + std::to_string(k));
            ++checks;
        }
        // 2) Prefix property within the bucket set, incl. k between buckets.
        for (int64_t k : {21, 33, 64, 65, 129, 256}) {
            auto t = Run(bucket, base, nb, query, nq, k);
            CheckOrder(t, nq, k, "bucket k=" + std::to_string(k));
            auto full = Run(bucket, base, nb, query, nq, 256);
            for (int64_t q = 0; q < nq; ++q)
                for (int64_t j = 0; j < k; ++j)
                    Require(t.i[q * k + j] == full.i[q * 256 + j] && t.d[q * k + j] == full.d[q * 256 + j],
                            "k=" + std::to_string(k) + " is not a prefix of k=256");
            ++checks;
        }
        // 3) Ties: query 0 must rank its 32 duplicate rows by ascending ID.
        {
            auto t = Run(bucket, base, nb, query, nq, 50);
            for (int64_t j = 0; j < 32; ++j) Require(t.i[j] == 1000 + 997 * j, "tie-break by ID violated");
            ++checks;
        }
        // 4) IVF CSR (filtered) path, incl. lists shorter than k (padding).
        {
            std::vector<int64_t> cand, off{0};
            const int64_t counts[] = {0, 7, 49, 50, 1000, 30000};
            for (int64_t q = 0; q < nq; ++q) {
                const int64_t count = counts[q % 6];
                for (int64_t j = 0; j < count; ++j) cand.push_back((j * 7919 + q * 104729) % nb);
                off.push_back(static_cast<int64_t>(cand.size()));
            }
            for (int64_t k : {50, 100}) {
                auto t = Run(bucket, base, nb, query, nq, k, &cand, &off);
                auto e = Run(exact, base, nb, query, nq, k, &cand, &off);
                Require(SameBits(t, e), "ivf k=" + std::to_string(k) + " differs from exact bundle");
                for (int64_t q = 0; q < nq; ++q) {
                    const int64_t count = off[q + 1] - off[q];
                    for (int64_t j = count; j < k; ++j)
                        Require(t.i[q * k + j] == -1 && std::isinf(t.d[q * k + j]) && t.d[q * k + j] < 0,
                                "ivf padding violated");
                }
                ++checks;
            }
        }
        // 5) Above the largest compiled k: refuse (caller falls back), never wrong rows.
        {
            int rc = 0;
            Run(bucket, base, nb, query, nq, 257, nullptr, nullptr, &rc);
            Require(rc != 0, "k=257 above every bundle must fail");
            ++checks;
        }
        // 6) Single-query device latency per requested k (diagnostic, not a gate).
        for (int64_t k : {20, 50, 100, 256}) {
            for (int w = 0; w < 5; ++w) Run(bucket, base, nb, query, 1, k);
            std::vector<double> ms;
            for (int r = 0; r < 30; ++r) {
                const auto s = std::chrono::steady_clock::now();
                Run(bucket, base, nb, query, 1, k);
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count());
            }
            std::sort(ms.begin(), ms.end());
            std::cout << "LATENCY k=" << k << " nq=1 n=30 p50_ms=" << ms[15] << " min_ms=" << ms[0]
                      << " max_ms=" << ms.back() << '\n';
        }
        // 7) Same, on the filtered IVF route production takes (owner bitset makes
        //    full_scan false): one query whose CSR list is every row.
        {
            std::vector<int64_t> all(nb), off{0, nb};
            std::iota(all.begin(), all.end(), 0);
            for (int64_t k : {20, 50, 100, 256}) {
                for (int w = 0; w < 5; ++w) Run(bucket, base, nb, query, 1, k, &all, &off);
                std::vector<double> ms;
                for (int r = 0; r < 30; ++r) {
                    const auto s = std::chrono::steady_clock::now();
                    Run(bucket, base, nb, query, 1, k, &all, &off);
                    ms.push_back(
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count());
                }
                std::sort(ms.begin(), ms.end());
                std::cout << "LATENCY_IVF k=" << k << " nq=1 n=30 p50_ms=" << ms[15] << " min_ms=" << ms[0]
                          << " max_ms=" << ms.back() << '\n';
            }
        }
        std::cout << "PASS checks=" << checks << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
