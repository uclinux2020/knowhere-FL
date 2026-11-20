// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Native Add boundary: final staging, assignment and inverted-list construction.
#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/fl/fl_ivf_flat_node.h"
#include "knowhere/version.h"

namespace {
const knowhere::fl::FlDeviceOps* native_ops;
size_t base_bytes, uploaded_bytes = 0;
int base_uploads = 0, assignments = 0;
double assign_ms = 0.0;
int
Copy(void* dst, const void* src, size_t bytes, void* stream) {
    uploaded_bytes += bytes;
    if (bytes == base_bytes)
        ++base_uploads;
    const int status = native_ops->memcpy_h2d(dst, src, bytes, stream);
    if (status != 0)
        throw std::runtime_error("Native upload failed");
    return status;
}
int
Assign(const void* x, int64_t nb, int64_t dim, const void* c, int64_t nlist, int metric, int64_t* out, void* stream) {
    const auto start = std::chrono::steady_clock::now();
    const int status = native_ops->kmeans_assign(x, nb, dim, c, nlist, metric, out, stream);
    assign_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (status != 0)
        throw std::runtime_error("Native assignment failed; fallback invalidates benchmark");
    ++assignments;
    return status;
}
}  // namespace

int
main(int argc, char** argv) {
    if (argc < 2 || argc > 4)
        throw std::runtime_error("Args: nb [add|build|quality [nlist]]");
    const bool quality = argc >= 3 && std::string(argv[2]) == "quality";
    const bool full_build = quality || (argc >= 3 && std::string(argv[2]) == "build");
    const int64_t nb = std::stoll(argv[1]), dim = 128, nlist = argc >= 4 ? std::stoll(argv[3]) : 128;
    const int64_t train_rows = full_build ? nb : 4096;
    if (nb < train_rows)
        throw std::runtime_error("Insufficient rows");
    native_ops = knowhere::fl::GetDeviceOps();
    if (!native_ops || !native_ops->kmeans_assign)
        throw std::runtime_error("Native adapter unavailable");
    auto ops = *native_ops;
    ops.memcpy_h2d = Copy;
    // Add-only mode keeps CPU training untimed; build mode includes native
    // assignment throughout Train and measures Train + Add wall time.
    ops.kmeans_assign = full_build ? Assign : nullptr;
    knowhere::fl::RegisterDeviceOps(&ops);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> value(-1.f, 1.f);
    std::vector<float> xb(nb * dim);
    for (auto& v : xb) v = value(rng);
    base_bytes = xb.size() * sizeof(float);
    const size_t used_before = knowhere::fl::FlMemoryPool::Instance().Used();
    {
        knowhere::FlIvfFlatIndexNode<knowhere::fp32> node(knowhere::Version::GetCurrentVersion().VersionNumber(),
                                                          nullptr);
        auto config = std::make_shared<knowhere::FlIvfFlatConfig>();
        config->metric_type = "L2";
        config->nlist = nlist;
        config->use_device_assign = true;
        const auto build_start = std::chrono::steady_clock::now();
        if (node.Train(knowhere::GenDataSet(train_rows, dim, xb.data()), config, false) != knowhere::Status::success)
            throw std::runtime_error("Train failed");
        const double train_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count();
        const double train_assign_ms = assign_ms;
        const int train_assignments = assignments;
        ops.kmeans_assign = Assign;
        uploaded_bytes = 0;
        base_uploads = assignments = 0;
        assign_ms = 0;
        const auto start = std::chrono::steady_clock::now();
        if (node.Add(knowhere::GenDataSet(nb, dim, xb.data()), config, false) != knowhere::Status::success)
            throw std::runtime_error("Add failed");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        rusage usage{};
        getrusage(RUSAGE_SELF, &usage);  // Before Search/Serialize, includes input and untimed training.
        if (assignments != 1 || knowhere::fl::FlMemoryPool::Instance().Used() != used_before + base_bytes)
            throw std::runtime_error("Missing assignment or incorrect resident accounting");
        std::cout << "ADD nb=" << nb << " ms=" << ms << " base_uploads=" << base_uploads
                  << " h2d_bytes=" << uploaded_bytes << " build_process_highwater_kib=" << usage.ru_maxrss << '\n';
        if (full_build) {
            if (train_assignments <= 0)
                throw std::runtime_error("Native training assignment missing");
            std::cout << "BUILD nb=" << nb << " nlist=" << nlist << " train_ms=" << train_ms << " add_ms=" << ms
                      << " total_ms=" << train_ms + ms << " train_assign_ms=" << train_assign_ms
                      << " add_assign_ms=" << assign_ms << " train_assignments=" << train_assignments << '\n';
        }
        knowhere::BinarySet binary;
        const auto serialize_start = std::chrono::steady_clock::now();
        if (node.Serialize(binary) != knowhere::Status::success)
            throw std::runtime_error("Serialize failed");
        std::cout
            << "SERIALIZE ms="
            << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - serialize_start).count()
            << '\n';
        auto artifact = binary.GetByName(node.Type());
        uint64_t hash = 14695981039346656037ULL;
        for (size_t i = 0; i < artifact->size; ++i) hash = (hash ^ artifact->data[i]) * 1099511628211ULL;
        std::cout << "ARTIFACT bytes=" << artifact->size << " fnv1a64=" << hash << '\n';
        if (quality) {
            // Use independent queries and double direct-distance ground truth.
            // Oracle construction and per-sample validation stay outside timing.
            constexpr int64_t nq = 100, k = 10;
            std::mt19937 query_rng(43);
            std::vector<float> queries(nq * dim);
            for (auto& v : queries) v = value(query_rng);
            std::vector<std::vector<int64_t>> truth(nq);
#pragma omp parallel for schedule(static)
            for (int64_t q = 0; q < nq; ++q) {
                std::vector<std::pair<double, int64_t>> scores(nb);
                for (int64_t j = 0; j < nb; ++j) {
                    double score = 0;
                    for (int64_t d = 0; d < dim; ++d) {
                        const double delta = static_cast<double>(queries[q * dim + d]) - xb[j * dim + d];
                        score += delta * delta;
                    }
                    scores[j] = {score, j};
                }
                std::partial_sort(scores.begin(), scores.begin() + k, scores.end());
                for (int64_t rank = 0; rank < k; ++rank) truth[q].push_back(scores[rank].second);
            }
            // Instrument callbacks through static storage, rejecting fallback.
            static int scan_calls = 0;
            static const knowhere::fl::FlDeviceOps* quality_ops;
            quality_ops = native_ops;
            ops.ivf_flat_scan = [](const void* b, int64_t rows, int64_t width, const void* q, int64_t count,
                                   const int64_t* cand, const int64_t* offsets, int64_t topk, int metric,
                                   float* distances, int64_t* ids, void* stream) {
                ++scan_calls;
                const int status = quality_ops->ivf_flat_scan(b, rows, width, q, count, cand, offsets, topk, metric,
                                                              distances, ids, stream);
                if (status != 0)
                    throw std::runtime_error("Quality scan failed; fallback forbidden");
                return status;
            };
            ops.brute_force_topk = [](const void* b, int64_t rows, const void* q, int64_t count, int64_t width,
                                      int64_t topk, int metric, float* distances, int64_t* ids, void* stream) {
                ++scan_calls;
                const int status =
                    quality_ops->brute_force_topk(b, rows, q, count, width, topk, metric, distances, ids, stream);
                if (status != 0)
                    throw std::runtime_error("Quality exact scan failed; fallback forbidden");
                return status;
            };
            std::set<int64_t> probes = {std::min<int64_t>(16, nlist), std::min<int64_t>(64, nlist), nlist / 2,
                                        3 * nlist / 4, nlist};
            nlohmann::json report = {{"nb", nb},     {"nlist", nlist},        {"nq", nq},
                                     {"k", k},       {"artifact_hash", hash}, {"train_ms", train_ms},
                                     {"add_ms", ms}, {"truth", truth},        {"rows", nlohmann::json::array()}};
            auto query = knowhere::GenDataSet(nq, dim, queries.data());
            for (const int64_t probe : probes) {
                std::vector<double> samples;
                std::vector<int64_t> first_ids;
                std::vector<int64_t> hits(nq);
                for (int repeat = -2; repeat < 7; ++repeat) {
                    auto search = std::make_unique<knowhere::FlIvfFlatConfig>();
                    search->k = k;
                    search->nprobe = probe;
                    const int calls_before = scan_calls;
                    const auto start = std::chrono::steady_clock::now();
                    auto result = node.Search(query, std::move(search), nullptr);
                    const double elapsed =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                    if (!result.has_value() || scan_calls != calls_before + 1)
                        throw std::runtime_error("Quality search missing native callback");
                    if (repeat >= 0)
                        samples.push_back(elapsed);
                    std::vector<int64_t> ids(result.value()->GetIds(), result.value()->GetIds() + nq * k);
                    if (!first_ids.empty() && ids != first_ids)
                        throw std::runtime_error("Quality IDs changed across repeats");
                    first_ids = ids;
                    for (int64_t q = 0; q < nq; ++q) {
                        std::set<int64_t> seen;
                        hits[q] = 0;
                        for (int64_t rank = 0; rank < k; ++rank) {
                            const int64_t id = ids[q * k + rank];
                            if (id < 0 || id >= nb || !seen.insert(id).second)
                                throw std::runtime_error("Quality invalid or duplicate ID");
                            double score = 0;
                            for (int64_t d = 0; d < dim; ++d) {
                                double delta = static_cast<double>(queries[q * dim + d]) - xb[id * dim + d];
                                score += delta * delta;
                            }
                            const float got = result.value()->GetDistance()[q * k + rank];
                            if (!std::isfinite(got) || std::abs(score - got) > 1e-4 * (1 + score))
                                throw std::runtime_error("Quality score differs from double oracle");
                            hits[q] += std::find(truth[q].begin(), truth[q].end(), id) != truth[q].end();
                        }
                        if (probe == nlist && hits[q] != k)
                            throw std::runtime_error("All-probe quality search not exact");
                    }
                }
                const double recall = std::accumulate(hits.begin(), hits.end(), 0.0) / (nq * k);
                report["rows"].push_back({{"nprobe", probe},
                                          {"recall", recall},
                                          {"hits", hits},
                                          {"ids", first_ids},
                                          {"samples_ms", samples}});
                std::cout << "QUALITY nprobe=" << probe << " recall=" << recall << '\n';
            }
            if (const char* path = std::getenv("KNOWHERE_IVF_QUALITY_OUTPUT")) {
                std::ofstream file(path);
                file << report.dump(2);
                if (!file)
                    throw std::runtime_error("Cannot write quality report");
            } else {
                throw std::runtime_error("KNOWHERE_IVF_QUALITY_OUTPUT is required");
            }
        }
    }
    if (knowhere::fl::FlMemoryPool::Instance().Used() != used_before)
        throw std::runtime_error("Pool reservation leaked");
    knowhere::fl::RegisterDeviceOps(nullptr);
}
