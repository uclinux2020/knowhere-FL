// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Native PPU operator benchmark. The CPU comparator is an independent direct
// FP32 distance + heap implementation, NOT the Knowhere CPU index or CUDA-C.
#include "common/fl/adapter/ppu/fl_ppu_runtime.h"

#include <omp.h>
#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <hggc.h>
#include <cstdlib>

namespace ppu = knowhere::fl::ppu;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
double Seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Buffer {
    void* ptr;
    explicit Buffer(size_t bytes) : ptr(ppu::Alloc(std::max<size_t>(1, bytes))) {
        Require(ptr != nullptr, "PPU allocation failed");
    }
    ~Buffer() { ppu::Free(ptr); }
};
struct StreamGuard {
    void* ptr = ppu::StreamCreate();
    StreamGuard() { Require(ptr != nullptr, "PPU stream creation failed"); }
    ~StreamGuard() { ppu::StreamDestroy(ptr); }
};
struct Result {
    std::vector<float> scores;
    std::vector<int64_t> ids;
    Result(int64_t nq, int64_t k) : scores(nq * k), ids(nq * k) {}
};
float Distance(const float* q, const float* b, int64_t dim, int metric) {
    float value = 0;
#pragma omp simd reduction(+:value)
    for (int64_t d = 0; d < dim; ++d) {
        float diff = q[d] - b[d];
        value += metric == 0 ? diff * diff : q[d] * b[d];
    }
    return value;
}
void CpuSearch(const std::vector<float>& xb, const std::vector<float>& q, int64_t nb, int64_t nq,
               int64_t dim, int64_t k, int metric, const std::vector<int64_t>& cand,
               const std::vector<int64_t>& offsets, Result& result) {
    using Candidate = std::pair<float, int64_t>;
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < nq; ++i) {
        // Max-heap of lower-is-better keys, including deterministic ID ties.
        std::priority_queue<Candidate> heap;
        int64_t count = offsets.empty() ? nb : offsets[i + 1] - offsets[i];
        for (int64_t c = 0; c < count; ++c) {
            int64_t id = offsets.empty() ? c : cand[offsets[i] + c];
            float score = Distance(q.data() + i * dim, xb.data() + id * dim, dim, metric);
            Candidate entry(metric == 0 ? score : -score, id);
            if (static_cast<int64_t>(heap.size()) < k) heap.push(entry);
            else if (entry < heap.top()) { heap.pop(); heap.push(entry); }
        }
        std::fill_n(result.ids.data() + i * k, k, -1);
        std::fill_n(result.scores.data() + i * k, k, metric == 0 ? INFINITY : -INFINITY);
        int64_t rank = static_cast<int64_t>(heap.size()) - 1;
        while (!heap.empty()) {
            auto entry = heap.top(); heap.pop();
            result.ids[i * k + rank] = entry.second;
            result.scores[i * k + rank] = metric == 0 ? entry.first : -entry.first;
            --rank;
        }
    }
}
Json Compare(const Result& expected, const Result& actual, int64_t nq, int64_t k) {
    size_t hit = 0, valid = 0, wrong_ids = 0, wrong_scores = 0;
    double max_error = 0;
    for (int64_t q = 0; q < nq; ++q) {
        for (int64_t t = 0; t < k; ++t) {
            auto pos = q * k + t;
            if (expected.ids[pos] >= 0) {
                ++valid;
                for (int64_t j = 0; j < k; ++j) if (actual.ids[pos] == expected.ids[q * k + j]) { ++hit; break; }
                double error = std::abs(static_cast<double>(expected.scores[pos]) - actual.scores[pos]);
                max_error = std::max(max_error, error);
                if (!std::isfinite(actual.scores[pos]) || error > 1e-4 * (1 + std::abs(expected.scores[pos]))) ++wrong_scores;
            } else if (!std::isinf(actual.scores[pos])) ++wrong_scores;
            if (expected.ids[pos] != actual.ids[pos]) ++wrong_ids;
        }
    }
    return {{"recall_at_k_same_representation", valid ? static_cast<double>(hit) / valid : 1.0},
            {"ordered_id_mismatches", wrong_ids}, {"score_mismatches", wrong_scores},
            {"max_score_abs_error", max_error}, {"eligible_truth_slots", valid}};
}
Json Summary(std::vector<double> times, int64_t nq) {
    auto raw = times;
    std::sort(times.begin(), times.end());
    auto percentile = [&](double p) { return times[static_cast<size_t>(std::ceil(p * times.size())) - 1]; };
    double median = percentile(.5);
    return {{"samples_s", raw}, {"batch_p50_s", median}, {"batch_p95_s", percentile(.95)},
            {"min_s", times.front()}, {"max_s", times.back()}, {"qps", nq / median}};
}
int main(int argc, char** argv) {
    try {
        // output mode nb nq dim k metric candidate_fraction distribution iterations threads device
        if (argc != 13) throw std::runtime_error("Args: output bf|ivf|pq|assign nb nq dim k L2|IP fraction random|clustered iters threads device");
        std::string output = argv[1], mode = argv[2], distribution = argv[9];
        int64_t nb = std::stoll(argv[3]), nq = std::stoll(argv[4]), dim = std::stoll(argv[5]), k = std::stoll(argv[6]);
        std::string metric_name = argv[7];
        int metric = metric_name == "L2" ? 0 : 1;
        double fraction = std::stod(argv[8]);
        int iters = std::stoi(argv[10]), threads = std::stoi(argv[11]), device = std::stoi(argv[12]);
        Require(nb > 0 && nq > 0 && dim > 0 && k > 0 && k <= nb && iters >= 3 && threads > 0 &&
                fraction >= 0 && fraction <= 1 && (metric_name == "L2" || metric_name == "IP") &&
                (mode == "bf" || mode == "ivf" || mode == "pq" || mode == "assign") &&
                (distribution == "random" || distribution == "clustered"), "Invalid benchmark configuration");
        Require(mode != "assign" || k == 1, "Assign requires k=1");
        omp_set_dynamic(0); omp_set_num_threads(threads);
        Require(ppu::SetDevice(device) == 0, "PPU context binding failed");
        StreamGuard stream;
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> random(-1, 1);
        std::vector<float> xb(nb * dim), queries(nq * dim);
        for (auto& x : xb) x = random(rng);
        for (auto& x : queries) x = random(rng);
        std::vector<int64_t> cand, offsets;
        if (mode == "ivf" || mode == "pq") {
            int64_t nc = static_cast<int64_t>(nb * fraction);
            offsets.push_back(0);
            std::vector<int64_t> perm(nb); std::iota(perm.begin(), perm.end(), 0);
            for (int64_t q = 0; q < nq; ++q) {
                if (distribution == "random") std::shuffle(perm.begin(), perm.end(), rng);
                else std::iota(perm.begin(), perm.end(), 0);
                cand.insert(cand.end(), perm.begin(), perm.begin() + nc);
                offsets.push_back(cand.size());
            }
        }
        int64_t m = mode == "pq" ? 16 : 0, ksub = mode == "pq" ? 256 : 0;
        Require(mode != "pq" || dim % m == 0, "PQ requires dim divisible by 16");
        std::vector<float> books;
        std::vector<uint8_t> codes;
        if (mode == "pq") {
            int64_t ds = dim / m;
            books.resize(m * ksub * ds); codes.resize(nb * m);
            for (auto& x : books) x = random(rng);
            for (int64_t id = 0; id < nb; ++id) for (int64_t s = 0; s < m; ++s) {
                uint8_t code = static_cast<uint8_t>(rng() % ksub); codes[id * m + s] = code;
                std::copy_n(books.data() + (s * ksub + code) * ds, ds, xb.data() + id * dim + s * ds);
            }
        }
        Result truth(nq, k), actual(nq, k);
        CpuSearch(xb, queries, nb, nq, dim, k, metric, cand, offsets, truth);
        std::vector<double> cpu_times;
        CpuSearch(xb, queries, nb, nq, dim, k, metric, cand, offsets, actual);
        for (int repeat = 0; repeat < iters; ++repeat) {
            auto start = Clock::now(); CpuSearch(xb, queries, nb, nq, dim, k, metric, cand, offsets, actual);
            cpu_times.push_back(Seconds(start));
        }
        const auto& base_host = mode == "pq" ? books : xb;
        auto allocation_start = Clock::now();
        Buffer base(base_host.size() * sizeof(float)), query(queries.size() * sizeof(float)), encoded(codes.size());
        double allocation_s = Seconds(allocation_start);
        auto stage_start = Clock::now();
        Require(ppu::CopyH2D(base.ptr, base_host.data(), base_host.size() * sizeof(float), stream.ptr) == 0, "base H2D failed");
        Require(ppu::CopyH2D(query.ptr, queries.data(), queries.size() * sizeof(float), stream.ptr) == 0, "query H2D failed");
        if (!codes.empty()) Require(ppu::CopyH2D(encoded.ptr, codes.data(), codes.size(), stream.ptr) == 0, "codes H2D failed");
        double stage_s = Seconds(stage_start);
        // Optional historical CUDA-C artifact is a comparison-only baseline;
        // never used by the Knowhere adapter or the optimized implementation.
        const char* legacy_path = std::getenv("KNOWHERE_PPU_LEGACY_CUBIN");
        HGmodule legacy_module = nullptr;
        HGfunction legacy_function = nullptr;
        Buffer legacy_scores(nq * k * sizeof(float)), legacy_ids(nq * k * sizeof(int64_t));
        if (legacy_path) {
            Require(mode == "bf" && dim == 128 && k == 10 && metric == 0, "Legacy artifact only supports BF d128 k10 L2");
            Require(hgModuleLoad(&legacy_module, legacy_path) == HGGC_SUCCESS, "Legacy artifact load failed");
            Require(hgModuleGetFunction(&legacy_function, legacy_module, "brute_force_topk") == HGGC_SUCCESS, "Legacy function missing");
        }
        auto search = [&] {
            int status;
            if (legacy_path) {
                HGdeviceptr xb_ptr = reinterpret_cast<uintptr_t>(base.ptr), xq_ptr = reinterpret_cast<uintptr_t>(query.ptr);
                HGdeviceptr od = reinterpret_cast<uintptr_t>(legacy_scores.ptr), oi = reinterpret_cast<uintptr_t>(legacy_ids.ptr), scratch = 0;
                void* params[] = {&xb_ptr, &nb, &xq_ptr, &nq, &dim, &k, &metric, &od, &oi, &scratch};
                HGlaunchConfig config{};
                config.gridDimX = nq; config.gridDimY = config.gridDimZ = 1;
                config.blockDimX = 256; config.blockDimY = config.blockDimZ = 1;
                config.hStream = static_cast<HGstream>(stream.ptr);
                Require(hgLaunchKernelEx(&config, legacy_function, params, nullptr) == HGGC_SUCCESS, "Legacy launch failed");
                Require(ppu::StreamSync(stream.ptr) == 0, "Legacy sync failed");
                Require(ppu::CopyD2H(actual.scores.data(), legacy_scores.ptr, nq * k * sizeof(float), stream.ptr) == 0, "Legacy score D2H failed");
                Require(ppu::CopyD2H(actual.ids.data(), legacy_ids.ptr, nq * k * sizeof(int64_t), stream.ptr) == 0, "Legacy ID D2H failed");
                return;
            }
            if (mode == "assign") {
                status = ppu::Assign(query.ptr, nq, dim, base.ptr, nb, metric, actual.ids.data(), stream.ptr);
                // Assign returns IDs only. Gather reference scores outside timed region below.
            } else {
                status = ppu::Search(base.ptr, nb, query.ptr, nq, dim, k, metric,
                                    offsets.empty() ? nullptr : cand.data(), offsets.empty() ? nullptr : offsets.data(),
                                    actual.scores.data(), actual.ids.data(), stream.ptr,
                                    codes.empty() ? nullptr : encoded.ptr, m, ksub, m ? dim / m : 0);
            }
            Require(status == 0, "PPU search failed; CPU fallback is NOT allowed in this benchmark");
        };
        auto first_start = Clock::now(); search(); double first_s = Seconds(first_start);
        search();
        std::vector<double> times;
        for (int repeat = 0; repeat < iters; ++repeat) { auto start = Clock::now(); search(); times.push_back(Seconds(start)); }
        if (mode == "assign") for (int64_t q = 0; q < nq; ++q) {
            Require(actual.ids[q] >= 0 && actual.ids[q] < nb, "invalid assigned centroid");
            actual.scores[q] = Distance(queries.data() + q * dim, xb.data() + actual.ids[q] * dim, dim, metric);
        }
        Json correctness = Compare(truth, actual, nq, k);
        Json cpu = Summary(cpu_times, nq), ppu_times = Summary(times, nq);
        struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        Json report = {{"schema", 1}, {"path", legacy_path ? "historical CUDA-C artifact via native HG (comparison only)" : "FlagOS Triton with Triton-TLE -> FlagTree AOT -> native HG -> PPU"},
                       {"mode", mode}, {"nb", nb}, {"nq", nq}, {"dim", dim}, {"k", k}, {"metric", metric_name},
                       {"seed", 42}, {"candidate_fraction", fraction}, {"excluded_fraction", offsets.empty() ? 0 : 1 - fraction},
                       {"candidate_distribution", distribution}, {"candidate_count_per_query", offsets.empty() ? nb : offsets[1]},
                       {"device", device}, {"cpu_threads", threads}, {"concurrency", 1}, {"warmup_calls", 2},
                       {"first_call_including_module_load_s", first_s}, {"input_device_allocation_s", allocation_s},
                       {"input_h2d_s", stage_s}, {"cpu_reference", cpu}, {"ppu_search", ppu_times},
                       {"cpu_over_ppu_speedup", cpu.at("batch_p50_s").get<double>() / ppu_times.at("batch_p50_s").get<double>()},
                       {"correctness", correctness}, {"process_lifetime_peak_rss_kib", usage.ru_maxrss},
                       {"input_device_bytes", base_host.size() * sizeof(float) + queries.size() * sizeof(float) + codes.size()},
                       {"host_candidate_bytes", cand.size() * sizeof(int64_t) + offsets.size() * sizeof(int64_t)},
                       {"timing_boundary", "resident base/query; includes runtime workspace alloc/free, candidate H2D, all launches/syncs, result D2H; excludes initial base/query H2D and AOT compilation"},
                       {"cpu_baseline", "independent direct FP32 distance + deterministic heap, OpenMP/SIMD; PQ uses decoded vectors, NOT a CPU LUT-ADC implementation; NOT Knowhere/Faiss index"}};
        std::ofstream file(output); Require(file.good(), "Cannot open report output"); file << report.dump(2) << '\n';
        std::cout << report.dump() << '\n';
        Require(correctness.at("ordered_id_mismatches") == 0 && correctness.at("score_mismatches") == 0, "PPU correctness check failed");
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
