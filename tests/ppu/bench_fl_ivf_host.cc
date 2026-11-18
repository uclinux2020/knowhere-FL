// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Host routing/CSR benchmark by default; opt-in native mode includes real PPU scans.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/fl/fl_host_adapter.h"
#include "index/fl/fl_ivf_flat_node.h"
#include "knowhere/version.h"

namespace {
const knowhere::fl::FlDeviceOps* host_ops;
bool native_scan = false;
int native_calls = 0;
std::vector<int64_t> expected_ids;
int
Scan(const void* xb, int64_t nb, int64_t dim, const void* queries, int64_t nq, const int64_t* candidates,
     const int64_t* offsets, int64_t k, int metric, float* scores, int64_t* ids, void* stream) {
    if (native_scan) {
        const int status =
            host_ops->ivf_flat_scan(xb, nb, dim, queries, nq, candidates, offsets, k, metric, scores, ids, stream);
        if (status != 0)
            throw std::runtime_error("Native IVF scan failed; CPU fallback is not allowed in this benchmark");
        ++native_calls;
        // The zero base fixture makes all L2 distances exactly equal. Verify
        // deterministic top-k IDs directly from each query's selected CSR.
        if (expected_ids.empty()) {
            expected_ids.assign(nq * k, -1);
            for (int64_t q = 0; q < nq; ++q) {
                std::vector<int64_t> row(candidates + offsets[q], candidates + offsets[q + 1]);
                const auto count = std::min<int64_t>(k, row.size());
                std::partial_sort(row.begin(), row.begin() + count, row.end());
                std::copy_n(row.begin(), count, expected_ids.data() + q * k);
            }
        }
        return status;
    }
    if (offsets[nq] <= 0)
        throw std::runtime_error("Expected nonempty CSR");
    std::fill_n(scores, nq * k, 0.f);
    std::fill_n(ids, nq * k, -1);
    return 0;
}
}  // namespace

int
main(int argc, char** argv) {
    if (argc != 4 && argc != 5)
        throw std::runtime_error("Args: nb nprobe repeats [nlist]");
    const int64_t nb = std::stoll(argv[1]), nprobe = std::stoll(argv[2]);
    const int repeats = std::stoi(argv[3]);
    const int64_t dim = 128, nlist = argc == 5 ? std::stoll(argv[4]) : 128, nq = 100, metric = 0;
    if (nlist <= 0 || nb < nlist || nprobe <= 0 || nprobe >= nlist || repeats < 3)
        throw std::runtime_error("Invalid benchmark shape");
    native_scan = std::getenv("KNOWHERE_IVF_BENCH_NATIVE") != nullptr;
    const auto* native_ops = knowhere::fl::GetDeviceOps();
    if (native_scan) {
        if (native_ops == nullptr || native_ops->ivf_flat_scan == nullptr || native_ops->device_count() <= 0)
            throw std::runtime_error("Native IVF adapter is unavailable");
    } else {
        // Consume lazy native registration before installing the host adapter.
        knowhere::fl::RegisterDeviceOps(nullptr);
        if (!knowhere::fl::RegisterHostAdapter())
            throw std::runtime_error("Host adapter registration failed");
    }
    host_ops = knowhere::fl::GetDeviceOps();
    auto ops = *host_ops;
    ops.ivf_flat_scan = Scan;
    knowhere::fl::RegisterDeviceOps(&ops);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<float> centroids(nlist * dim), xb(nb * dim), queries(nq * dim);
    for (auto& x : centroids) x = random(rng);
    for (auto& x : queries) x = random(rng);
    knowhere::MemoryIOWriter writer;
    knowhere::fl::WriteFormatHeader(writer);
    writer(&dim, sizeof(dim), 1);
    writer(&nb, sizeof(nb), 1);
    writer(&nlist, sizeof(nlist), 1);
    const int32_t metric_code = metric;
    writer(&metric_code, sizeof(metric_code), 1);
    writer(centroids.data(), sizeof(float), centroids.size());
    writer(xb.data(), sizeof(float), xb.size());
    for (int64_t list = 0; list < nlist; ++list) {
        std::vector<int64_t> members;
        for (int64_t id = list; id < nb; id += nlist) members.push_back(id);
        const int64_t count = members.size();
        writer(&count, sizeof(count), 1);
        writer(members.data(), sizeof(int64_t), members.size());
    }
    knowhere::BinarySet binary;
    binary.Append(knowhere::IndexEnum::INDEX_FL_IVFFLAT, std::shared_ptr<uint8_t[]>(writer.data()), writer.tellg());
    {
        knowhere::FlIvfFlatIndexNode<knowhere::fp32> node(knowhere::Version::GetCurrentVersion().VersionNumber(),
                                                          nullptr);
        if (node.Deserialize(binary, nullptr) != knowhere::Status::success)
            throw std::runtime_error("Load failed");
        auto query = knowhere::GenDataSet(nq, dim, queries.data());
        std::vector<double> times;
        for (int repeat = -2; repeat < repeats; ++repeat) {
            auto config = std::make_unique<knowhere::FlIvfFlatConfig>();
            config->k = 10;
            config->nprobe = nprobe;
            const auto start = std::chrono::steady_clock::now();
            auto result = node.Search(query, std::move(config), nullptr);
            if (!result.has_value())
                throw std::runtime_error("Search failed");
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (repeat >= 0)
                times.push_back(ms);
            if (native_scan) {
                if (native_calls != repeat + 3 ||
                    !std::equal(expected_ids.begin(), expected_ids.end(), result.value()->GetIds()))
                    throw std::runtime_error("Native scan missing or ordered IDs differ from CSR oracle");
                for (int64_t q = 0; q < nq; ++q) {
                    float norm = 0.f;
                    for (int64_t d = 0; d < dim; ++d) norm += queries[q * dim + d] * queries[q * dim + d];
                    for (int64_t rank = 0; rank < 10; ++rank) {
                        const float got = result.value()->GetDistance()[q * 10 + rank];
                        if (!std::isfinite(got) || std::abs(got - norm) > 1e-4f * (1.f + norm))
                            throw std::runtime_error("Native scores differ from zero-base L2 oracle");
                    }
                }
            }
        }
        std::cout << (native_scan ? "NATIVE_SAMPLES_MS" : "HOST_SAMPLES_MS");
        for (double ms : times) std::cout << ' ' << ms;
        std::cout << '\n';
        std::sort(times.begin(), times.end());
        std::cout << (native_scan ? "NATIVE_IVF" : "HOST_ONLY") << " nb=" << nb << " nq=" << nq << " nlist=" << nlist
                  << " nprobe=" << nprobe << " p50_ms=" << times[times.size() / 2] << " min_ms=" << times.front()
                  << " max_ms=" << times.back() << '\n';
    }
    knowhere::fl::RegisterDeviceOps(nullptr);
}
