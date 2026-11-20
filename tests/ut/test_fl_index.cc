// Copyright (C) 2026 knowhere-FL contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// FL (FlagOS domestic-GPU) backend tests. The whole file is compiled only in
// WITH_FL builds; device behavior is emulated through the host adapter, so no
// GPU is required.
#ifdef KNOWHERE_WITH_FL

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/fl/fl_device.h"
#include "common/fl/fl_device_router.h"
#include "common/fl/fl_host_adapter.h"
#include "index/fl/fl_cagra_node.h"
#include "index/fl/fl_format.h"
#include "index/fl/fl_index_node.h"
#include "index/fl/fl_ivf_flat_node.h"
#include "index/fl/fl_ivf_pq_node.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace knowhere::fl {
// Serial test-only reset: no index or worker may outlive this fixture.
struct FlDeviceRouterTestAccess {
    static void
    Clear() {
        auto& router = FlDeviceRouter::Instance();
        router.initialized_.~once_flag();
        new (&router.initialized_) std::once_flag;
        router.ids_.clear();
        router.error_.clear();
        router.configured_ = false;
        router.Reset();
    }
};
}  // namespace knowhere::fl

namespace {

struct DeviceAllowlistFixture {
    const knowhere::fl::FlDeviceOps* saved_ops{knowhere::fl::GetDeviceOps()};
    bool had_env{std::getenv("KNOWHERE_FL_DEVICE_IDS") != nullptr};
    std::string saved_env{had_env ? std::getenv("KNOWHERE_FL_DEVICE_IDS") : ""};
    knowhere::fl::FlDeviceOps ops{};
    inline static thread_local int current_device = -1;
    inline static std::mutex observed_mutex;
    inline static std::vector<int> observed;
    inline static std::vector<int> search_devices;
    inline static knowhere::fl::FlDeviceOps host_ops{};

    template <typename Fn, typename... Args>
    static int
    ObserveSearchContext(Fn fn, Args... args) {
        {
            std::lock_guard<std::mutex> lock(observed_mutex);
            search_devices.push_back(current_device);
        }
        const int result = fn(args...);
        // A subsequent search must bind its context, not inherit a worker's previous one.
        current_device = -1;
        return result;
    }

    DeviceAllowlistFixture() {
        unsetenv("KNOWHERE_FL_DEVICE_IDS");
        knowhere::fl::FlDeviceRouterTestAccess::Clear();
        knowhere::fl::RegisterHostAdapter();
        ops = *knowhere::fl::GetDeviceOps();
        host_ops = ops;
        ops.brute_force_topk = [](const void* xb, int64_t nb, const void* xq, int64_t nq, int64_t dim, int64_t k,
                                  int metric, float* distances, int64_t* ids, void* stream) {
            return ObserveSearchContext(host_ops.brute_force_topk, xb, nb, xq, nq, dim, k, metric, distances, ids,
                                        stream);
        };
        ops.ivf_flat_scan = [](const void* xb, int64_t nb, int64_t dim, const void* xq, int64_t nq,
                               const int64_t* candidates, const int64_t* offsets, int64_t k, int metric,
                               float* distances, int64_t* ids, void* stream) {
            return ObserveSearchContext(host_ops.ivf_flat_scan, xb, nb, dim, xq, nq, candidates, offsets, k, metric,
                                        distances, ids, stream);
        };
        ops.ivf_pq_scan = [](const void* codebooks, int64_t m, int64_t ksub, int64_t dsub, const void* codes,
                             int64_t nb, const void* xq, int64_t nq, int64_t dim, const int64_t* candidates,
                             const int64_t* offsets, int64_t k, int metric, float* distances, int64_t* ids,
                             void* stream) {
            return ObserveSearchContext(host_ops.ivf_pq_scan, codebooks, m, ksub, dsub, codes, nb, xq, nq, dim,
                                        candidates, offsets, k, metric, distances, ids, stream);
        };
        ops.graph_search = [](const void* xb, int64_t nb, int64_t dim, const int64_t* graph_ids,
                              const int64_t* graph_offsets, int64_t degree, const void* xq, int64_t nq, int64_t k,
                              int64_t itopk, int metric, float* distances, int64_t* ids, void* stream) {
            return ObserveSearchContext(host_ops.graph_search, xb, nb, dim, graph_ids, graph_offsets, degree, xq, nq, k,
                                        itopk, metric, distances, ids, stream);
        };
        ops.device_count = [] { return 8; };
        ops.set_device = [](int id) {
            current_device = id;
            std::lock_guard<std::mutex> lock(observed_mutex);
            observed.push_back(id);
            return id >= 0 && id < 8 ? 0 : -1;
        };
        observed.clear();
        knowhere::fl::RegisterDeviceOps(&ops);
    }
    void
    Configure(const char* value) {
        knowhere::fl::FlDeviceRouterTestAccess::Clear();
        if (value) {
            setenv("KNOWHERE_FL_DEVICE_IDS", value, 1);
        } else {
            unsetenv("KNOWHERE_FL_DEVICE_IDS");
        }
        observed.clear();
    }
    ~DeviceAllowlistFixture() {
        knowhere::fl::FlDeviceRouterTestAccess::Clear();
        if (had_env) {
            setenv("KNOWHERE_FL_DEVICE_IDS", saved_env.c_str(), 1);
        } else {
            unsetenv("KNOWHERE_FL_DEVICE_IDS");
        }
        knowhere::fl::RegisterDeviceOps(saved_ops);
    }
};

constexpr int64_t kNb = 2000;
constexpr int64_t kDim = 16;
constexpr int64_t kNq = 5;
constexpr int64_t kK = 10;

// exact brute-force KNN as ground truth (recall gold standard)
std::vector<std::vector<int64_t>>
ExactKnn(const float* xb, const float* xq, int64_t nb, int64_t nq, int64_t dim, int64_t k, bool is_ip) {
    std::vector<std::vector<int64_t>> out(nq);
    for (int64_t i = 0; i < nq; ++i) {
        std::vector<std::pair<float, int64_t>> ds(nb);
        for (int64_t j = 0; j < nb; ++j) {
            float dot = 0.f, na = 0.f, nbsum = 0.f;
            for (int64_t d = 0; d < dim; ++d) {
                dot += xq[i * dim + d] * xb[j * dim + d];
                na += xq[i * dim + d] * xq[i * dim + d];
                nbsum += xb[j * dim + d] * xb[j * dim + d];
            }
            ds[j] = {is_ip ? dot : (na + nbsum - 2.f * dot), j};
        }
        std::partial_sort(ds.begin(), ds.begin() + k, ds.end(), [is_ip](const auto& a, const auto& b) {
            return is_ip ? (a.first > b.first) : (a.first < b.first);
        });
        for (int64_t t = 0; t < k; ++t) {
            out[i].push_back(ds[t].second);
        }
    }
    return out;
}

float
RecallAtK(const knowhere::DataSetPtr& res, const std::vector<std::vector<int64_t>>& truth, int64_t nq, int64_t k) {
    const int64_t* ids = res->GetIds();
    size_t hit = 0;
    for (int64_t i = 0; i < nq; ++i) {
        std::set<int64_t> gt(truth[i].begin(), truth[i].end());
        for (int64_t t = 0; t < k; ++t) {
            if (gt.count(ids[i * k + t]) > 0) {
                ++hit;
            }
        }
    }
    return static_cast<float>(hit) / static_cast<float>(nq * k);
}

knowhere::Index<knowhere::IndexNode>
CreateFlIndex(const std::string& name) {
    auto version = knowhere::Version::GetCurrentVersion();
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version.VersionNumber());
    REQUIRE(idx.has_value());
    return std::move(idx.value());
}

}  // namespace

TEST_CASE("FL host adapter registration", "[fl]") {
    knowhere::fl::GetDeviceOps();              // Complete native lazy registration before the test override.
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    REQUIRE(knowhere::fl::DeviceAvailable());
    REQUIRE(knowhere::fl::DeviceCount() == 1);
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    REQUIRE_FALSE(knowhere::fl::DeviceAvailable());
}

TEST_CASE("FL_BRUTE_FORCE build/search matches exact KNN (CPU fallback)", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.  // no
                                               // adapter -> CPU fallback path
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);

    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);
    REQUIRE(idx.Count() == kNb);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10})");
    auto res = idx.Search(query, search, nullptr);
    REQUIRE(res.has_value());

    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    // brute force must be exact regardless of path
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);
}

TEST_CASE("FL_BRUTE_FORCE device path via host adapter matches exact KNN", "[fl]") {
    knowhere::fl::GetDeviceOps();              // Complete native lazy registration before the test override.
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);  // unlimited

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_GPU_BRUTEFORCE);  // user-facing name

    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"IP","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"IP","k":10})");
    auto res = idx.Search(query, search, nullptr);
    REQUIRE(res.has_value());

    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/true);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
}

TEST_CASE("FL_BRUTE_FORCE OOM pre-check falls back to CPU with correct results", "[fl]") {
    knowhere::fl::GetDeviceOps();              // Complete native lazy registration before the test override.
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    knowhere::fl::FlMemoryPool::Instance().Configure(0, 1);  // 1-byte ceiling -> staging rejected (D5)

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);

    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10})");
    auto res = idx.Search(query, search, nullptr);
    REQUIRE(res.has_value());

    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);

    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
}

TEST_CASE("FL_BRUTE_FORCE serialize/deserialize round trip", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);

    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);

    auto idx2 = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
    knowhere::Json load_cfg = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx2.Deserialize(binset, load_cfg) == knowhere::Status::success);
    REQUIRE(idx2.Count() == kNb);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10})");
    auto res = idx2.Search(query, search, nullptr);
    REQUIRE(res.has_value());
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);
}

TEST_CASE("FL_BRUTE_FORCE rejects COSINE with guidance (design D4)", "[fl]") {
    auto base = GenDataSet(kNb, kDim, 42);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"COSINE","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::invalid_metric_type);
}

TEST_CASE("FL_IVF_FLAT build/search recall on clustered data", "[fl]") {
    knowhere::fl::GetDeviceOps();              // Complete native lazy registration before the test override.
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_GPU_IVFFLAT);  // user-facing name

    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);
    REQUIRE(idx.Count() == kNb);

    // nprobe == nlist -> scans everything -> must be exact
    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"nprobe":16})");
    auto res = idx.Search(query, search, nullptr);
    REQUIRE(res.has_value());
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);

    // nprobe < nlist -> approximate; on uniform random data recall should stay high
    knowhere::Json search2 = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"nprobe":8})");
    auto res2 = idx.Search(query, search2, nullptr);
    REQUIRE(res2.has_value());
    REQUIRE(RecallAtK(res2.value(), truth, kNq, kK) >= 0.8f);

    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
}

TEST_CASE("FL_IVF_FLAT full scan preserves filters padding and device failure fallback", "[fl][ivf-full]") {
    knowhere::fl::GetDeviceOps();
    knowhere::fl::RegisterDeviceOps(nullptr);
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    const auto* host_ops = knowhere::fl::GetDeviceOps();
    auto base = GenDataSet(67, 16, 42);
    auto query = GenDataSet(3, 16, 7);
    for (const std::string metric : {"L2", "IP"}) {
        auto index = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
        REQUIRE(index.Build(base, {{"metric_type", metric}, {"dim", 16}, {"nlist", 8}}) == knowhere::Status::success);
        for (int64_t k : {1, 10, 70}) {
            CAPTURE(metric, k);
            const knowhere::Json cfg = {{"metric_type", metric}, {"k", k}, {"nprobe", 8}};
            auto device = index.Search(query, cfg, nullptr);
            REQUIRE(device.has_value());
            // A failed BF specialization must rebuild candidates and scan CPU.
            auto failed_ops = *host_ops;
            failed_ops.brute_force_topk = [](const void*, int64_t, const void*, int64_t, int64_t, int64_t, int, float*,
                                             int64_t*, void*) { return -1; };
            knowhere::fl::RegisterDeviceOps(&failed_ops);
            auto fallback = index.Search(query, cfg, nullptr);
            knowhere::fl::RegisterDeviceOps(host_ops);
            REQUIRE(fallback.has_value());
            for (int64_t q = 0; q < 3; ++q) {
                for (int64_t rank = 0; rank < k; ++rank) {
                    REQUIRE(device.value()->GetIds()[q * k + rank] == fallback.value()->GetIds()[q * k + rank]);
                    if (rank >= 67)
                        REQUIRE(device.value()->GetIds()[q * k + rank] == -1);
                }
            }
            std::vector<uint8_t> bits((67 + 7) / 8, 0xff);
            bits[0] &= ~uint8_t(1);        // Only row zero remains eligible.
            bits.back() &= uint8_t(0x07);  // Padding bits are outside the 67-row filter.
            auto filtered = index.Search(query, cfg, knowhere::BitsetView(bits.data(), 67));
            REQUIRE(filtered.has_value());
            for (int64_t q = 0; q < 3; ++q) {
                REQUIRE(filtered.value()->GetIds()[q * k] == 0);
                for (int64_t rank = 1; rank < k; ++rank) REQUIRE(filtered.value()->GetIds()[q * k + rank] == -1);
            }
        }
    }
    knowhere::fl::RegisterDeviceOps(nullptr);
}

TEST_CASE("FL_IVF_FLAT large CSR copy preserves candidate order across thread limits", "[fl][ivf-csr]") {
    knowhere::fl::GetDeviceOps();
    knowhere::fl::RegisterDeviceOps(nullptr);
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    const auto* host_ops = knowhere::fl::GetDeviceOps();
    struct Restore {
        const knowhere::fl::FlDeviceOps* ops;
        int threads = omp_get_max_threads();
        ~Restore() {
            omp_set_num_threads(threads);
            knowhere::fl::RegisterDeviceOps(ops);
        }
    } restore{host_ops};
    static std::vector<int64_t> captured_ids, captured_offsets;
    auto capture_ops = *host_ops;
    capture_ops.ivf_flat_scan = [](const void*, int64_t, int64_t, const void*, int64_t nq, const int64_t* ids,
                                   const int64_t* offsets, int64_t k, int, float* scores, int64_t* results, void*) {
        captured_ids.assign(ids, ids + offsets[nq]);
        captured_offsets.assign(offsets, offsets + nq + 1);
        std::fill_n(scores, nq * k, 0.f);
        std::fill_n(results, nq * k, -1);
        return 0;
    };
    // The adapter outlives all buffers owned by the node.
    knowhere::fl::RegisterDeviceOps(&capture_ops);
    constexpr int64_t nq = 100;
    const int64_t nlist = GENERATE(64, 65);
    const int64_t nb = nlist * 256;
    // Include a partial SIMD block and a dimension not divisible by the lane count.
    const int64_t dim = GENERATE(1, 17, 128);
    const int32_t metric = GENERATE(0, 1);
    const std::string metric_name = metric == 0 ? "L2" : "IP";
    std::vector<float> centroids(nlist * dim, 0.f), xb(nb * dim), queries(nq * dim, 0.f);
    // Permuted, tied scores exercise both selection algorithms and their boundary.
    for (int64_t list = 0; list < nlist; ++list) {
        const float value = static_cast<float>((list * 17 % nlist) / 3);
        centroids[list * dim] = metric == 0 ? value : -value;
        for (int64_t d = 1; d < dim; ++d) centroids[list * dim + d] = static_cast<float>((list + d) % 7 - 3) * 0.125f;
    }
    for (int64_t q = 0; q < nq; ++q) {
        queries[q * dim] = metric == 1 ? 1.f : 0.f;
        for (int64_t d = 1; d < dim; ++d) queries[q * dim + d] = 0.125f;
    }
    std::vector<std::pair<float, int64_t>> route_order;
    for (int64_t list = 0; list < nlist; ++list) {
        float norm = 0.f, dot = 0.f;
        for (int64_t d = 0; d < dim; ++d) {
            const float c = centroids[list * dim + d];
            norm += c * c;
            dot += queries[d] * c;
        }
        route_order.emplace_back(metric == 0 ? norm - 2.f * dot : dot, list);
    }
    std::sort(route_order.begin(), route_order.end(), [metric](const auto& a, const auto& b) {
        return a.first == b.first ? a.second < b.second : metric == 1 ? a.first > b.first : a.first < b.first;
    });
    knowhere::MemoryIOWriter writer;
    knowhere::fl::WriteFormatHeader(writer);
    writer(&dim, sizeof(dim), 1);
    writer(&nb, sizeof(nb), 1);
    writer(&nlist, sizeof(nlist), 1);
    writer(&metric, sizeof(metric), 1);
    writer(centroids.data(), sizeof(float), centroids.size());
    writer(xb.data(), sizeof(float), xb.size());
    for (int64_t list = 0; list < nlist; ++list) {
        const int64_t count = nb / nlist;
        writer(&count, sizeof(count), 1);
        for (int64_t id = list; id < nb; id += nlist) writer(&id, sizeof(id), 1);
    }
    knowhere::BinarySet binary;
    binary.Append(knowhere::IndexEnum::INDEX_FL_IVFFLAT, std::shared_ptr<uint8_t[]>(writer.data()), writer.tellg());
    auto index = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    REQUIRE(index.Deserialize(binary, {{"metric_type", metric_name}, {"dim", dim}, {"nlist", nlist}}) ==
            knowhere::Status::success);
    auto query = knowhere::GenDataSet(nq, dim, queries.data());
    // Cover small/large probe prefixes; 40/42 straddle parallel copying.
    for (int64_t probe : {1, 8, 9, 40, 42, 63}) {
        const knowhere::Json config = {{"metric_type", metric_name}, {"k", 1}, {"nprobe", probe}};
        omp_set_num_threads(1);
        REQUIRE(index.Search(query, config, nullptr).has_value());
        const auto serial_ids = captured_ids;
        const auto serial_offsets = captured_offsets;
        REQUIRE(serial_ids.size() == static_cast<size_t>(nq * probe * nb / nlist));
        for (int threads : {2, 8, 88}) {
            omp_set_num_threads(threads);
            REQUIRE(index.Search(query, config, nullptr).has_value());
            REQUIRE(captured_ids == serial_ids);
            REQUIRE(captured_offsets == serial_offsets);
        }
        for (int64_t q = 0; q < nq; ++q) {
            const auto first = serial_offsets[q];
            for (int64_t list = 0; list < probe; ++list) {
                for (int64_t row = 0; row < nb / nlist; ++row) {
                    if (serial_ids[first + list * (nb / nlist) + row] != row * nlist + route_order[list].second) {
                        FAIL("CSR member order changed");
                    }
                }
            }
        }
        std::vector<uint8_t> bits((nb + 7) / 8, 0xaa);  // Exclude odd row IDs.
        const auto mask = knowhere::BitsetView(bits.data(), nb);
        omp_set_num_threads(1);
        REQUIRE(index.Search(query, config, mask).has_value());
        const auto filtered_ids = captured_ids;
        const auto filtered_offsets = captured_offsets;
        std::vector<int64_t> expected;
        std::copy_if(serial_ids.begin(), serial_ids.end(), std::back_inserter(expected),
                     [](int64_t id) { return id % 2 == 0; });
        REQUIRE(filtered_ids == expected);
        for (int threads : {2, 8, 88}) {
            omp_set_num_threads(threads);
            REQUIRE(index.Search(query, config, mask).has_value());
            REQUIRE(captured_ids == filtered_ids);
            REQUIRE(captured_offsets == filtered_offsets);
        }
    }
    // A failed device scan must read the same fully initialized CSR on CPU.
    capture_ops.ivf_flat_scan = [](const void*, int64_t, int64_t, const void*, int64_t, const int64_t*, const int64_t*,
                                   int64_t, int, float*, int64_t*, void*) { return -1; };
    const auto fallback = index.Search(query, {{"metric_type", metric_name}, {"k", 1}, {"nprobe", 6}}, nullptr);
    REQUIRE(fallback.has_value());
    int64_t best_id = nb;
    for (int64_t p = 0; p < 6; ++p) best_id = std::min(best_id, route_order[p].second);
    float query_norm = 0.f;
    for (int64_t d = 0; d < dim; ++d) query_norm += queries[d] * queries[d];
    for (int64_t q = 0; q < nq; ++q) {
        REQUIRE(fallback.value()->GetIds()[q] == best_id);
        REQUIRE(fallback.value()->GetDistance()[q] == (metric == 0 ? query_norm : 0.f));
    }
}

TEST_CASE("FL_IVF_FLAT search config preserves probes above the train-only default", "[fl][ivf-nprobe]") {
    knowhere::FlIvfFlatConfig config;
    config.metric_type = "L2";
    config.nprobe = 1024;
    REQUIRE(config.CheckAndAdjust(knowhere::PARAM_TYPE::SEARCH, nullptr) == knowhere::Status::success);
    REQUIRE(config.nprobe.value() == 1024);
}

TEST_CASE("FL_IVF_FLAT full probes use the trained list count above 128", "[fl][ivf-nprobe]") {
    struct RestoreDeviceOps {
        const knowhere::fl::FlDeviceOps* saved = knowhere::fl::GetDeviceOps();
        RestoreDeviceOps() {
            knowhere::fl::RegisterDeviceOps(nullptr);
        }
        ~RestoreDeviceOps() {
            knowhere::fl::RegisterDeviceOps(saved);
        }
    } restore;
    constexpr int64_t nb = 256, dim = 1, nlist = 256;
    const int32_t metric = 0;
    std::vector<float> centroids(nlist), xb(nb, 2.f);
    std::iota(centroids.begin(), centroids.end(), 0.f);
    xb.back() = 0.f;  // The exact neighbor belongs to the last probed list.
    knowhere::MemoryIOWriter writer;
    knowhere::fl::WriteFormatHeader(writer);
    writer(&dim, sizeof(dim), 1);
    writer(&nb, sizeof(nb), 1);
    writer(&nlist, sizeof(nlist), 1);
    writer(&metric, sizeof(metric), 1);
    writer(centroids.data(), sizeof(float), centroids.size());
    writer(xb.data(), sizeof(float), xb.size());
    for (int64_t id = 0; id < nb; ++id) {
        const int64_t count = 1;
        writer(&count, sizeof(count), 1);
        writer(&id, sizeof(id), 1);
    }
    knowhere::BinarySet binary;
    binary.Append(knowhere::IndexEnum::INDEX_FL_IVFFLAT, std::shared_ptr<uint8_t[]>(writer.data()), writer.tellg());
    auto index = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    REQUIRE(index.Deserialize(binary, {{"metric_type", "L2"}, {"dim", dim}, {"nlist", nlist}}) ==
            knowhere::Status::success);
    float q = 0.f;
    auto query = knowhere::GenDataSet(1, dim, &q);
    for (int64_t probe : {128, 256, 512}) {
        auto result = index.Search(query, {{"metric_type", "L2"}, {"k", 1}, {"nprobe", probe}}, nullptr);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetIds()[0] == (probe == 128 ? 0 : 255));
        REQUIRE(result.value()->GetDistance()[0] == (probe == 128 ? 4.f : 0.f));
    }
}

TEST_CASE("FL_IVF_FLAT tied probes remain nested and full probes are exact", "[fl][ivf-nprobe]") {
    // A serialized fixture makes routing ties reproducible without depending on kmeans training.
    struct RestoreDeviceOps {
        const knowhere::fl::FlDeviceOps* saved = knowhere::fl::GetDeviceOps();
        RestoreDeviceOps() {
            knowhere::fl::RegisterDeviceOps(nullptr);
        }
        ~RestoreDeviceOps() {
            knowhere::fl::RegisterDeviceOps(saved);
        }
    } restore_device_ops;
    constexpr int64_t nb = 15, dim = 1, nlist = 15;
    const float route[] = {2, 1, 1, 0, 2, 1, 2, 1, 2, 1, 1, 2, 0, 0, 1};
    for (bool is_ip : {false, true}) {
        const int metric = is_ip ? 1 : 0;
        const std::string metric_name = is_ip ? "IP" : "L2";
        for (bool tied_vectors : {false, true}) {
            CAPTURE(metric_name, tied_vectors);
            std::vector<float> xb(nb), centroids(nb);
            float q = is_ip ? 1.f : 0.f;
            for (int64_t id = 0; id < nb; ++id) {
                centroids[id] = is_ip ? -route[id] : std::sqrt(route[id]);
                float value = tied_vectors ? static_cast<float>(id % 3) : static_cast<float>(id + 1);
                if (id == 5)
                    value = 0.f;  // This neighbor used to disappear between probes 5 and 6.
                xb[id] = is_ip ? -value : value;
            }
            knowhere::MemoryIOWriter writer;
            knowhere::fl::WriteFormatHeader(writer);
            writer(&dim, sizeof(dim), 1);
            writer(&nb, sizeof(nb), 1);
            writer(&nlist, sizeof(nlist), 1);
            writer(&metric, sizeof(metric), 1);
            writer(centroids.data(), sizeof(float), centroids.size());
            writer(xb.data(), sizeof(float), xb.size());
            for (int64_t id = 0; id < nb; ++id) {
                const int64_t size = 1;
                writer(&size, sizeof(size), 1);
                writer(&id, sizeof(id), 1);
            }
            knowhere::BinarySet binset;
            binset.Append(knowhere::IndexEnum::INDEX_FL_IVFFLAT, std::shared_ptr<uint8_t[]>(writer.data()),
                          writer.tellg());
            auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
            knowhere::Json cfg = {{"metric_type", metric_name}, {"dim", dim}, {"nlist", nlist}};
            REQUIRE(idx.Deserialize(binset, cfg) == knowhere::Status::success);
            auto query = knowhere::GenDataSet(1, dim, &q);
            std::set<int64_t> previous;
            int64_t previous_hits = 0;
            std::vector<std::pair<float, int64_t>> exact;
            for (int64_t id = 0; id < nb; ++id) {
                exact.emplace_back(is_ip ? q * xb[id] : xb[id] * xb[id], id);
            }
            auto better = [is_ip](const auto& a, const auto& b) {
                return a.first == b.first ? a.second < b.second : is_ip ? a.first > b.first : a.first < b.first;
            };
            std::sort(exact.begin(), exact.end(), better);
            for (int64_t nprobe = 1; nprobe <= nlist; ++nprobe) {
                CAPTURE(nprobe);
                knowhere::Json search = {{"metric_type", metric_name}, {"k", nb}, {"nprobe", nprobe}};
                auto candidates = idx.Search(query, search, nullptr);
                REQUIRE(candidates.has_value());
                std::set<int64_t> current;
                for (int64_t rank = 0; rank < nb; ++rank) {
                    const int64_t id = candidates.value()->GetIds()[rank];
                    if (rank < nprobe) {
                        REQUIRE(id >= 0);
                        REQUIRE(current.insert(id).second);
                    } else {
                        REQUIRE(id == -1);
                    }
                }
                REQUIRE(std::includes(current.begin(), current.end(), previous.begin(), previous.end()));
                previous = current;
                std::vector<std::pair<float, int64_t>> truth;
                for (const auto& candidate : exact) {
                    if (current.count(candidate.second))
                        truth.push_back(candidate);
                }
                search["k"] = 3;
                auto result = idx.Search(query, search, nullptr);
                REQUIRE(result.has_value());
                int64_t hits = 0;
                for (int64_t rank = 0; rank < 3; ++rank) {
                    const int64_t id = result.value()->GetIds()[rank];
                    if (rank < static_cast<int64_t>(truth.size())) {
                        REQUIRE(id == truth[rank].second);
                        REQUIRE(result.value()->GetDistance()[rank] == Catch::Approx(truth[rank].first));
                    } else {
                        REQUIRE(id == -1);
                    }
                    for (int64_t t = 0; t < 3; ++t) hits += id == exact[t].second;
                }
                REQUIRE(hits >= previous_hits);
                previous_hits = hits;
                if (nprobe == nlist)
                    REQUIRE(hits == 3);
            }
        }
    }
}

TEST_CASE("FL_IVF_FLAT serialize/deserialize round trip", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);

    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);

    auto idx2 = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    knowhere::Json load_cfg = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16})");
    REQUIRE(idx2.Deserialize(binset, load_cfg) == knowhere::Status::success);
    REQUIRE(idx2.Count() == kNb);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"nprobe":16})");
    auto res = idx2.Search(query, search, nullptr);
    REQUIRE(res.has_value());
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);
}

TEST_CASE("FL_IVF_FLAT device-assign and CPU routing agree", "[fl]") {
    // With the host adapter registered, kmeans assignment routes through the
    // emulated device (ops->kmeans_assign). use_device_assign=false must give
    // identical recall via pure CPU routing (design D5 fallback parity).
    knowhere::fl::GetDeviceOps();              // Complete native lazy registration before the test override.
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"nprobe":16})");
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);

    auto idx_dev = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    REQUIRE(idx_dev.Build(
                base, knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16,"use_device_assign":true})")) ==
            knowhere::Status::success);
    auto res_dev = idx_dev.Search(query, search, nullptr);
    REQUIRE(res_dev.has_value());

    auto idx_cpu = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    REQUIRE(idx_cpu.Build(
                base, knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16,"use_device_assign":false})")) ==
            knowhere::Status::success);
    auto res_cpu = idx_cpu.Search(query, search, nullptr);
    REQUIRE(res_cpu.has_value());

    // both paths are exact at nprobe==nlist regardless of routing mode
    REQUIRE(RecallAtK(res_dev.value(), truth, kNq, kK) == 1.0f);
    REQUIRE(RecallAtK(res_cpu.value(), truth, kNq, kK) == 1.0f);

    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
}

TEST_CASE("FL_IVF_PQ device ADC and CPU fallback agree", "[fl]") {
    // Host adapter registered -> index stages codebooks/codes on the emulated
    // device and scans via ops->ivf_pq_scan. A second index built with a
    // 1-byte pool cap fails the OOM pre-check (D5) and scans on CPU. Both
    // must return identical top-k ids (deterministic ADC semantics).
    knowhere::fl::GetDeviceOps();              // Complete native lazy registration before the test override.
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    REQUIRE(knowhere::fl::RegisterHostAdapter());

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    const std::string build_json = R"({"metric_type":"L2","dim":16,"nlist":16,"m":4,"nbits":8})";
    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"nprobe":16})");

    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);  // unlimited
    auto idx_dev = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFPQ);
    REQUIRE(idx_dev.Build(base, knowhere::Json::parse(build_json)) == knowhere::Status::success);
    auto res_dev = idx_dev.Search(query, search, nullptr);
    REQUIRE(res_dev.has_value());

    knowhere::fl::FlMemoryPool::Instance().Configure(0, 1);  // force OOM pre-check failure
    auto idx_cpu = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFPQ);
    REQUIRE(idx_cpu.Build(base, knowhere::Json::parse(build_json)) == knowhere::Status::success);
    auto res_cpu = idx_cpu.Search(query, search, nullptr);
    REQUIRE(res_cpu.has_value());
    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);

    const int64_t* ids_dev = res_dev.value()->GetIds();
    const int64_t* ids_cpu = res_cpu.value()->GetIds();
    for (int64_t i = 0; i < kNq * kK; ++i) {
        REQUIRE(ids_dev[i] == ids_cpu[i]);
    }

    // ADC is approximate: assert a conservative recall floor vs exact truth
    // (recalibrate after the remote PPU run if needed)
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    REQUIRE(RecallAtK(res_dev.value(), truth, kNq, kK) >= 0.4f);

    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
}

TEST_CASE("FL_IVF_PQ serialize/deserialize round trip", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);

    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_GPU_IVFPQ);  // user-facing name
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16,"m":4,"nbits":8})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);
    REQUIRE(idx.Count() == kNb);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);

    auto idx2 = CreateFlIndex(knowhere::IndexEnum::INDEX_GPU_IVFPQ);
    knowhere::Json load_cfg = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16,"m":4,"nbits":8})");
    REQUIRE(idx2.Deserialize(binset, load_cfg) == knowhere::Status::success);
    REQUIRE(idx2.Count() == kNb);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"nprobe":16})");
    auto res = idx.Search(query, search, nullptr);
    auto res2 = idx2.Search(query, search, nullptr);
    REQUIRE(res.has_value());
    REQUIRE(res2.has_value());
    // same data + deterministic train (fixed seed) -> identical results
    const int64_t* ids = res.value()->GetIds();
    const int64_t* ids2 = res2.value()->GetIds();
    for (int64_t i = 0; i < kNq * kK; ++i) {
        REQUIRE(ids[i] == ids2[i]);
    }
}

TEST_CASE("FL_IVF_PQ rejects invalid train params", "[fl]") {
    auto base = GenDataSet(kNb, kDim, 42);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFPQ);

    // nbits != 8 not supported in v1
    knowhere::Json bad_nbits = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16,"m":4,"nbits":4})");
    REQUIRE(idx.Build(base, bad_nbits) == knowhere::Status::invalid_args);

    // dim % m != 0
    knowhere::Json bad_m = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16,"m":3,"nbits":8})");
    REQUIRE(idx.Build(base, bad_m) == knowhere::Status::invalid_args);

    // COSINE rejected per design D4
    knowhere::Json cosine = knowhere::Json::parse(R"({"metric_type":"COSINE","dim":16,"nlist":16,"m":4,"nbits":8})");
    REQUIRE(idx.Build(base, cosine) == knowhere::Status::invalid_metric_type);

    // IP rejected: plain-dot PQ ADC cannot decompose IP distance (base norms
    // are lost to quantization); FL IVF_PQ is L2-only
    knowhere::Json ip = knowhere::Json::parse(R"({"metric_type":"IP","dim":16,"nlist":16,"m":4,"nbits":8})");
    REQUIRE(idx.Build(base, ip) == knowhere::Status::invalid_metric_type);
}

TEST_CASE("FL centroid updates preserve serial double sums", "[fl][centroid-update]") {
    const int saved_threads = omp_get_max_threads();
    struct RestoreThreads {
        int count;
        ~RestoreThreads() {
            omp_set_num_threads(count);
        }
    } restore{saved_threads};
    constexpr int64_t nb = 9000, dim = 129, nlist = 17;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    std::vector<float> x(nb * dim);
    for (auto& v : x) v = random(rng);
    for (bool skewed : {false, true}) {
        std::vector<int64_t> owner(nb);
        for (int64_t j = 0; j < nb; ++j) owner[j] = skewed ? 0 : rng() % (nlist - 1);
        std::vector<double> expected(nlist * dim, 0.0), sums;
        std::vector<int64_t> expected_counts(nlist, 0), counts;
        for (int64_t j = 0; j < nb; ++j) {
            ++expected_counts[owner[j]];
            for (int64_t d = 0; d < dim; ++d) expected[owner[j] * dim + d] += x[j * dim + d];
        }
        for (int threads : {1, 2, 8, 88}) {
            omp_set_num_threads(threads);
            knowhere::fl::AccumulateCentroids(x.data(), owner.data(), nb, dim, nlist, sums, counts);
            REQUIRE(counts == expected_counts);
            REQUIRE(std::memcmp(sums.data(), expected.data(), sums.size() * sizeof(double)) == 0);
        }
    }
}

TEST_CASE("FL_IVF_FLAT Add reuses resident vectors and preserves fallback", "[fl][add-resident]") {
    DeviceAllowlistFixture fixture;
    fixture.Configure("6");
    static int base_uploads, assign_calls;
    static bool fail_assign;
    constexpr int64_t nb = 256, dim = 16, nlist = 8;
    constexpr size_t base_bytes = nb * dim * sizeof(float);
    const auto saved_used = knowhere::fl::FlMemoryPool::Instance().Used();
    fixture.ops.memcpy_h2d = [](void* dst, const void* src, size_t bytes, void* stream) {
        if (bytes == base_bytes)
            ++base_uploads;
        return DeviceAllowlistFixture::host_ops.memcpy_h2d(dst, src, bytes, stream);
    };
    fixture.ops.kmeans_assign = [](const void* x, int64_t rows, int64_t width, const void* c, int64_t lists, int metric,
                                   int64_t* out, void* stream) {
        ++assign_calls;
        if (fail_assign)
            return -1;
        return DeviceAllowlistFixture::host_ops.kmeans_assign(x, rows, width, c, lists, metric, out, stream);
    };
    auto base = GenDataSet(nb, dim, 42);
    auto query = GenDataSet(2, dim, 43);
    knowhere::Json config = {{"metric_type", "L2"}, {"dim", dim}, {"nlist", nlist}, {"use_device_assign", false}};
    std::vector<int64_t> expected;
    for (bool failure : {false, true}) {
        // Train with an unavailable callback to keep centroids identical;
        // re-enable device assignment only for Add.
        config["use_device_assign"] = true;
        auto assign_fn = fixture.ops.kmeans_assign;
        fixture.ops.kmeans_assign = nullptr;
        {
            auto index = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
            REQUIRE(index.Train(base, config) == knowhere::Status::success);
            fixture.ops.kmeans_assign = assign_fn;
            base_uploads = assign_calls = 0;
            fail_assign = failure;
            REQUIRE(index.Add(base, config) == knowhere::Status::success);
            REQUIRE(base_uploads == 1);
            REQUIRE(assign_calls == 1);
            REQUIRE(knowhere::fl::FlMemoryPool::Instance().Used() == saved_used + base_bytes);
            auto result = index.Search(query, {{"metric_type", "L2"}, {"k", 10}, {"nprobe", nlist}}, nullptr);
            REQUIRE(result.has_value());
            std::vector<int64_t> ids(result.value()->GetIds(), result.value()->GetIds() + 20);
            if (expected.empty())
                expected = ids;
            REQUIRE(ids == expected);
        }
        REQUIRE(knowhere::fl::FlMemoryPool::Instance().Used() == saved_used);
    }
}

TEST_CASE("FL device allowlist rejects malformed physical IDs", "[fl][device-allowlist]") {
    using Router = knowhere::fl::FlDeviceRouter;
    REQUIRE(Router::ParseDeviceIds("6", 8) == std::vector<int>{6});
    REQUIRE(Router::ParseDeviceIds("7,0,6", 8) == std::vector<int>{7, 0, 6});
    for (const auto* raw : {"", ",", "0,", ",0", "0,,1", "0,0", "06,6", "-1", "+1", " 1", "1 ", "1x", "8", "2147483648",
                            "999999999999999999999999"}) {
        INFO(raw);
        REQUIRE_THROWS_AS(Router::ParseDeviceIds(raw, 8), std::invalid_argument);
    }
    REQUIRE_THROWS_AS(Router::ParseDeviceIds("0", 0), std::invalid_argument);
}

TEST_CASE("FL device allowlist freezes physical round-robin and errors", "[fl][device-allowlist]") {
    DeviceAllowlistFixture fixture;
    auto& router = knowhere::fl::FlDeviceRouter::Instance();
    SECTION("unset retains all physical devices") {
        fixture.Configure(nullptr);
        for (int i = 0; i < 24; ++i) REQUIRE(router.Acquire() == i % 8);
    }
    SECTION("single physical device is not logical zero") {
        fixture.Configure("6");
        router.Preflight();
        setenv("KNOWHERE_FL_DEVICE_IDS", "0", 1);
        router.Reset();
        for (int i = 0; i < 24; ++i) REQUIRE(router.Acquire() == 6);
        REQUIRE(router.DeviceCount() == 8);
    }
    SECTION("multi-device order and cross-thread dispatch") {
        fixture.Configure("7,0,6");
        for (int expected : {7, 0, 6, 7, 0, 6}) REQUIRE(router.Acquire() == expected);
        router.Reset();
        std::vector<int> ids(300);
        std::vector<std::thread> workers;
        for (int t = 0; t < 3; ++t) {
            workers.emplace_back([&, t] {
                for (int i = t; i < 300; i += 3) ids[i] = router.Acquire();
            });
        }
        for (auto& worker : workers) worker.join();
        for (int id : {7, 0, 6}) REQUIRE(std::count(ids.begin(), ids.end(), id) == 100);
    }
    SECTION("error is sticky after environment repair") {
        fixture.Configure("8");
        REQUIRE_THROWS_AS(router.Preflight(), std::invalid_argument);
        setenv("KNOWHERE_FL_DEVICE_IDS", "6", 1);
        router.Reset();
        REQUIRE_THROWS_AS(router.Acquire(), std::invalid_argument);
    }
    SECTION("missing adapter is not successful CPU fallback") {
        fixture.Configure("6");
        knowhere::fl::RegisterDeviceOps(nullptr);
        REQUIRE_THROWS_AS(router.Preflight(), std::invalid_argument);
    }
    SECTION("unusable device is rejected at initialization") {
        fixture.Configure("6");
        fixture.ops.set_device = [](int) { return -1; };
        REQUIRE_THROWS_AS(router.Preflight(), std::invalid_argument);
    }
}

TEST_CASE("FL device allowlist rejects index creation before training or search", "[fl][device-allowlist]") {
    DeviceAllowlistFixture fixture;
    const auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    for (const auto* raw : {"", "0,0", "8", "garbage"}) {
        fixture.Configure(raw);
        for (const auto& name : {knowhere::IndexEnum::INDEX_FL_BRUTEFORCE, knowhere::IndexEnum::INDEX_FL_IVFFLAT,
                                 knowhere::IndexEnum::INDEX_FL_IVFPQ, knowhere::IndexEnum::INDEX_FL_CAGRA,
                                 knowhere::IndexEnum::INDEX_GPU_IVFFLAT}) {
            INFO(name);
            const auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(name, version);
            REQUIRE_FALSE(index.has_value());
            REQUIRE(index.what().find("KNOWHERE_FL_DEVICE_IDS") != std::string::npos);
        }
        REQUIRE(fixture.observed.empty());
    }
}

TEST_CASE("FL allowlist covers training Add Deserialize and cross-thread context", "[fl][device-allowlist]") {
    DeviceAllowlistFixture fixture;
    fixture.Configure("6");
    auto base = GenDataSet(256, kDim, 42);
    auto queries = GenDataSet(2, kDim, 43);
    for (const auto& name : {knowhere::IndexEnum::INDEX_FL_BRUTEFORCE, knowhere::IndexEnum::INDEX_FL_IVFFLAT,
                             knowhere::IndexEnum::INDEX_FL_IVFPQ, knowhere::IndexEnum::INDEX_FL_CAGRA,
                             knowhere::IndexEnum::INDEX_GPU_IVFFLAT}) {
        INFO(name);
        auto index = CreateFlIndex(name);
        fixture.observed.clear();  // Exclude initialization from lifecycle witnesses.
        knowhere::Json config = {{"metric_type", "L2"}, {"dim", kDim}};
        if (name == knowhere::IndexEnum::INDEX_FL_IVFFLAT || name == knowhere::IndexEnum::INDEX_GPU_IVFFLAT) {
            config["nlist"] = 4;
            config["use_device_assign"] = true;
        } else if (name == knowhere::IndexEnum::INDEX_FL_IVFPQ) {
            config["nlist"] = 4;
            config["m"] = 4;
            config["nbits"] = 8;
        } else if (name == knowhere::IndexEnum::INDEX_FL_CAGRA) {
            config["graph_degree"] = 16;
        }
        REQUIRE(index.Train(base, config) == knowhere::Status::success);
        const auto after_train = fixture.observed.size();
        if (name == knowhere::IndexEnum::INDEX_FL_IVFFLAT || name == knowhere::IndexEnum::INDEX_GPU_IVFFLAT)
            REQUIRE(after_train > 0);
        REQUIRE(index.Add(base, config) == knowhere::Status::success);
        REQUIRE(fixture.observed.size() > after_train);
        knowhere::BinarySet binary;
        REQUIRE(index.Serialize(binary) == knowhere::Status::success);
        auto loaded = CreateFlIndex(name);
        const auto before_load = fixture.observed.size();
        REQUIRE(loaded.Deserialize(binary, config) == knowhere::Status::success);
        REQUIRE(fixture.observed.size() > before_load);
        // Search runs on the registered global pool, not on this caller thread.
        // Observe TLS inside the adapter callback and require an actual device search.
        for (int repeat = 0; repeat < 2; ++repeat) {
            fixture.search_devices.clear();
            bool success = false;
            std::thread worker([&] {
                knowhere::Json search = {{"metric_type", "L2"}, {"k", 10}, {"nprobe", 4}};
                success = loaded.Search(queries, search, nullptr).has_value();
            });
            worker.join();
            REQUIRE(success);
            REQUIRE_FALSE(fixture.search_devices.empty());
            REQUIRE(std::all_of(fixture.search_devices.begin(), fixture.search_devices.end(),
                                [](int id) { return id == 6; }));
        }
        REQUIRE(std::all_of(fixture.observed.begin(), fixture.observed.end(), [](int id) { return id == 6; }));
    }
}

TEST_CASE("FL device router round-robin", "[fl]") {
    auto& router = knowhere::fl::FlDeviceRouter::Instance();
    router.Reset();
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    // no adapter -> single-device semantics
    REQUIRE(router.Acquire() == 0);
    // host adapter reports exactly 1 emulated device -> always 0
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    REQUIRE(router.Acquire() == 0);
    REQUIRE(router.Acquire() == 0);
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
}

TEST_CASE("FL_BRUTE_FORCE GetVectorByIds returns raw vectors", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    REQUIRE(idx.HasRawData("L2"));

    std::vector<int64_t> want{0, 7, 123, 1999};
    auto ids_ds = knowhere::GenIdsDataSet(want.size(), want.data());
    auto res = idx.GetVectorByIds(ids_ds);
    REQUIRE(res.has_value());
    const float* got = static_cast<const float*>(res.value()->GetTensor());
    const float* src = static_cast<const float*>(base->GetTensor());
    for (size_t i = 0; i < want.size(); ++i) {
        for (int64_t d = 0; d < kDim; ++d) {
            REQUIRE(got[i * kDim + d] == src[want[i] * kDim + d]);
        }
    }
}

TEST_CASE("FL_IVF_PQ reports no raw data", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFPQ);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"nlist":16,"m":4,"nbits":8})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    REQUIRE_FALSE(idx.HasRawData("L2"));
    int64_t query_id = 5;
    auto ids_ds = knowhere::GenIdsDataSet(1, &query_id);
    auto r = idx.GetVectorByIds(ids_ds);
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error() == knowhere::Status::not_implemented);
}

TEST_CASE("FL deserialize rejects corrupted format header", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);
    auto binary = binset.GetByName(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
    REQUIRE(binary != nullptr);
    REQUIRE(binary->size >= 8);  // {magic, version} header

    knowhere::Json load_cfg = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");

    // corrupt the magic -> must fail loudly instead of misreading the payload
    {
        knowhere::BinarySet tampered;
        std::shared_ptr<uint8_t[]> copy(new uint8_t[binary->size]);
        std::memcpy(copy.get(), binary->data.get(), binary->size);
        copy[0] ^= 0xFF;
        tampered.Append(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE, copy, binary->size);
        auto idx2 = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
        REQUIRE(idx2.Deserialize(tampered, load_cfg) != knowhere::Status::success);
    }

    // bump the format version beyond what this build understands -> reject
    {
        knowhere::BinarySet tampered;
        std::shared_ptr<uint8_t[]> copy(new uint8_t[binary->size]);
        std::memcpy(copy.get(), binary->data.get(), binary->size);
        int32_t too_new = knowhere::fl::kFlFormatVersion + 1;
        std::memcpy(copy.get() + sizeof(uint32_t), &too_new, sizeof(too_new));
        tampered.Append(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE, copy, binary->size);
        auto idx2 = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
        REQUIRE(idx2.Deserialize(tampered, load_cfg) != knowhere::Status::success);
    }
}

TEST_CASE("FL nodes expose static load-resource estimates", "[fl]") {
    using knowhere::IndexVersion;
    const uint64_t kFileSize = 4096;
    const int64_t kRows = 1000, kDimV = 16;

    auto bf_res = knowhere::FlBruteForceIndexNode<knowhere::fp32>::StaticEstimateLoadResource(
        kFileSize, kRows, kDimV, *knowhere::FlBruteForceIndexNode<knowhere::fp32>::StaticCreateConfig(),
        IndexVersion(0));
    REQUIRE(bf_res.has_value());
    REQUIRE(bf_res.value().memoryCost == kFileSize);  // raw vectors stage to device 1:1

    auto ivf_res = knowhere::FlIvfFlatIndexNode<knowhere::fp32>::StaticEstimateLoadResource(
        kFileSize, kRows, kDimV, *knowhere::FlIvfFlatIndexNode<knowhere::fp32>::StaticCreateConfig(), IndexVersion(0));
    REQUIRE(ivf_res.has_value());
    REQUIRE(ivf_res.value().memoryCost == kRows * kDimV * sizeof(float));

    auto pq_res = knowhere::FlIvfPqIndexNode<knowhere::fp32>::StaticEstimateLoadResource(
        kFileSize, kRows, kDimV, *knowhere::FlIvfPqIndexNode<knowhere::fp32>::StaticCreateConfig(), IndexVersion(0));
    REQUIRE(pq_res.has_value());
    // default m = 8 (FlIvfPqConfig) -> nb*m uint8 codes
    REQUIRE(pq_res.value().memoryCost == kRows * 8);

    // CAGRA: vectors + graph CSR (default degree 32)
    auto cagra_res = knowhere::FlCagraIndexNode<knowhere::fp32>::StaticEstimateLoadResource(
        kFileSize, kRows, kDimV, *knowhere::FlCagraIndexNode<knowhere::fp32>::StaticCreateConfig(), IndexVersion(0));
    REQUIRE(cagra_res.has_value());
    REQUIRE(cagra_res.value().memoryCost ==
            kRows * kDimV * sizeof(float) + kRows * 32 * static_cast<int64_t>(sizeof(int64_t)));
}

TEST_CASE("FL_CAGRA graph search reaches high recall with wide frontier", "[fl]") {
    knowhere::fl::RegisterDeviceOps(
        nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.  // CPU traversal path
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);

    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_CAGRA);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"graph_degree":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);
    REQUIRE(idx.Count() == kNb);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"itopk_size":256})");
    auto res = idx.Search(query, search, nullptr);
    REQUIRE(res.has_value());
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    // greedy traversal on a degree-16 kNN graph with a wide frontier is
    // near-exact; keep a conservative bound to stay robust across RNG seeds
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) >= 0.9f);
}

TEST_CASE("FL_CAGRA serialize/deserialize round trip preserves results", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);

    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_CAGRA);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"graph_degree":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(idx.Serialize(binset) == knowhere::Status::success);

    auto idx2 = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_CAGRA);
    knowhere::Json load_cfg = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx2.Deserialize(binset, load_cfg) == knowhere::Status::success);
    REQUIRE(idx2.Count() == kNb);
    REQUIRE(idx2.HasRawData("L2"));

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"itopk_size":256})");
    auto res1 = idx.Search(query, search, nullptr);
    auto res2 = idx2.Search(query, search, nullptr);
    REQUIRE(res1.has_value());
    REQUIRE(res2.has_value());
    // graph build is deterministic (full brute-force kNN) -> identical results
    const auto* id1 = res1.value()->GetIds();
    const auto* id2 = res2.value()->GetIds();
    for (int64_t i = 0; i < kNq * kK; ++i) {
        REQUIRE(id1[i] == id2[i]);
    }
}

TEST_CASE("FL_CAGRA refuses oversized CPU graph build", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    // v1 CPU builder is quadratic and refused beyond kMaxCpuGraphRows; the
    // rejection happens before any O(nb^2) work, so a tiny dim keeps this fast
    auto huge = GenDataSet(knowhere::FlCagraIndexNode<knowhere::fp32>::kMaxCpuGraphRows + 1, 2, 1);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_CAGRA);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":2,"graph_degree":8})");
    REQUIRE(idx.Build(huge, build) == knowhere::Status::invalid_args);
}

TEST_CASE("FL_CAGRA post-filters bitset from traversal results", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(1, kDim, 7);

    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_CAGRA);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"graph_degree":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"itopk_size":256})");
    auto unfiltered = idx.Search(query, search, nullptr);
    REQUIRE(unfiltered.has_value());
    const int64_t banned = unfiltered.value()->GetIds()[0];

    std::vector<uint8_t> bits((kNb + 7) / 8, 0);
    bits[static_cast<size_t>(banned) / 8] |= static_cast<uint8_t>(1u << (banned % 8));
    auto filtered = idx.Search(query, search, knowhere::BitsetView(bits.data(), kNb));
    REQUIRE(filtered.has_value());
    const auto* fids = filtered.value()->GetIds();
    for (int64_t t = 0; t < kK; ++t) {
        REQUIRE(fids[t] != banned);
    }
}

TEST_CASE("FL_CAGRA GetVectorByIds returns raw vectors", "[fl]") {
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
    auto base = GenDataSet(kNb, kDim, 42);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_CAGRA);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"graph_degree":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    REQUIRE(idx.HasRawData("L2"));
    std::vector<int64_t> want{0, 7, 123, 1999};
    auto ids_ds = knowhere::GenIdsDataSet(want.size(), want.data());
    auto res = idx.GetVectorByIds(ids_ds);
    REQUIRE(res.has_value());
    const float* got = static_cast<const float*>(res.value()->GetTensor());
    const float* src = static_cast<const float*>(base->GetTensor());
    for (size_t i = 0; i < want.size(); ++i) {
        for (int64_t d = 0; d < kDim; ++d) {
            REQUIRE(got[i * kDim + d] == src[want[i] * kDim + d]);
        }
    }
}

TEST_CASE("FL_BRUTE_FORCE bitset over-fetch keeps exact top-k", "[fl]") {
    knowhere::fl::RegisterHostAdapter();  // device path: kernel over-fetches, host compacts to exact top-k
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(1, kDim, 7);

    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10})");
    auto unfiltered = idx.Search(query, search, nullptr);
    REQUIRE(unfiltered.has_value());
    const int64_t banned = unfiltered.value()->GetIds()[0];

    std::vector<uint8_t> bits((kNb + 7) / 8, 0);
    bits[static_cast<size_t>(banned) / 8] |= static_cast<uint8_t>(1u << (banned % 8));
    auto filtered = idx.Search(query, search, knowhere::BitsetView(bits.data(), kNb));
    REQUIRE(filtered.has_value());
    const auto* fids = filtered.value()->GetIds();
    // exact semantics: banning the rank-0 hit must shift every later rank up
    // by one and still deliver a full top-k (no padding holes)
    for (int64_t t = 0; t < kK; ++t) {
        REQUIRE(fids[t] >= 0);
        REQUIRE(fids[t] != banned);
        if (t < kK - 1) {
            REQUIRE(fids[t] == unfiltered.value()->GetIds()[t + 1]);
        }
    }
    knowhere::fl::RegisterDeviceOps(nullptr);  // Explicitly isolate host/CPU tests from native PPU registration.
}

#endif  // KNOWHERE_WITH_FL
