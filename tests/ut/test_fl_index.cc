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

// FL (FlagOS domestic-GPU) backend tests. The whole file is compiled only in
// WITH_FL builds; device behavior is emulated through the host adapter, so no
// GPU is required.
#ifdef KNOWHERE_WITH_FL

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "common/fl/fl_device.h"
#include "common/fl/fl_host_adapter.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {

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
    knowhere::fl::UnregisterHostAdapter();
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    REQUIRE(knowhere::fl::DeviceAvailable());
    REQUIRE(knowhere::fl::DeviceCount() == 1);
    knowhere::fl::UnregisterHostAdapter();
    REQUIRE_FALSE(knowhere::fl::DeviceAvailable());
}

TEST_CASE("FL_BRUTE_FORCE build/search matches exact KNN (CPU fallback)", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();  // no adapter -> CPU fallback path
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);

    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);
    REQUIRE(idx.Count() == kNb);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10})");
    auto res = idx.Search(query, search);
    REQUIRE(res.has_value());

    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    // brute force must be exact regardless of path
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);
}

TEST_CASE("FL_BRUTE_FORCE device path via host adapter matches exact KNN", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);  // unlimited

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_GPU_BRUTEFORCE);  // user-facing name

    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"IP","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"IP","k":10})");
    auto res = idx.Search(query, search);
    REQUIRE(res.has_value());

    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/true);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);
    knowhere::fl::UnregisterHostAdapter();
}

TEST_CASE("FL_BRUTE_FORCE OOM pre-check falls back to CPU with correct results", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    knowhere::fl::FlMemoryPool::Instance().Configure(0, 1);  // 1-byte ceiling -> staging rejected (D5)

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_BRUTEFORCE);

    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10})");
    auto res = idx.Search(query, search);
    REQUIRE(res.has_value());

    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);

    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);
    knowhere::fl::UnregisterHostAdapter();
}

TEST_CASE("FL_BRUTE_FORCE serialize/deserialize round trip", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();
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
    auto res = idx2.Search(query, search);
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
    knowhere::fl::UnregisterHostAdapter();
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
    auto res = idx.Search(query, search);
    REQUIRE(res.has_value());
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);

    // nprobe < nlist -> approximate; on uniform random data recall should stay high
    knowhere::Json search2 = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"nprobe":8})");
    auto res2 = idx.Search(query, search2);
    REQUIRE(res2.has_value());
    REQUIRE(RecallAtK(res2.value(), truth, kNq, kK) >= 0.8f);

    knowhere::fl::UnregisterHostAdapter();
}

TEST_CASE("FL_IVF_FLAT serialize/deserialize round trip", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();
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
    auto res = idx2.Search(query, search);
    REQUIRE(res.has_value());
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);
    REQUIRE(RecallAtK(res.value(), truth, kNq, kK) == 1.0f);
}

TEST_CASE("FL_IVF_FLAT device-assign and CPU routing agree", "[fl]") {
    // With the host adapter registered, kmeans assignment routes through the
    // emulated device (ops->kmeans_assign). use_device_assign=false must give
    // identical recall via pure CPU routing (design D5 fallback parity).
    knowhere::fl::UnregisterHostAdapter();
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);

    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(kNq, kDim, 7);
    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"nprobe":16})");
    auto truth = ExactKnn(static_cast<const float*>(base->GetTensor()), static_cast<const float*>(query->GetTensor()),
                          kNb, kNq, kDim, kK, /*is_ip=*/false);

    auto idx_dev = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    REQUIRE(idx_dev.Build(base, knowhere::Json::parse(
                                      R"({"metric_type":"L2","dim":16,"nlist":16,"use_device_assign":true})")) ==
            knowhere::Status::success);
    auto res_dev = idx_dev.Search(query, search);
    REQUIRE(res_dev.has_value());

    auto idx_cpu = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    REQUIRE(idx_cpu.Build(base, knowhere::Json::parse(
                                      R"({"metric_type":"L2","dim":16,"nlist":16,"use_device_assign":false})")) ==
            knowhere::Status::success);
    auto res_cpu = idx_cpu.Search(query, search);
    REQUIRE(res_cpu.has_value());

    // both paths are exact at nprobe==nlist regardless of routing mode
    REQUIRE(RecallAtK(res_dev.value(), truth, kNq, kK) == 1.0f);
    REQUIRE(RecallAtK(res_cpu.value(), truth, kNq, kK) == 1.0f);

    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);
    knowhere::fl::UnregisterHostAdapter();
}

#endif  // KNOWHERE_WITH_FL
