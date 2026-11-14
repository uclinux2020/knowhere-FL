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
#include <limits>
#include <set>
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
    auto res = idx.Search(query, search, nullptr);
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
    auto res = idx.Search(query, search, nullptr);
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
    auto res = idx.Search(query, search, nullptr);
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
    auto res_dev = idx_dev.Search(query, search, nullptr);
    REQUIRE(res_dev.has_value());

    auto idx_cpu = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_IVFFLAT);
    REQUIRE(idx_cpu.Build(base, knowhere::Json::parse(
                                      R"({"metric_type":"L2","dim":16,"nlist":16,"use_device_assign":false})")) ==
            knowhere::Status::success);
    auto res_cpu = idx_cpu.Search(query, search, nullptr);
    REQUIRE(res_cpu.has_value());

    // both paths are exact at nprobe==nlist regardless of routing mode
    REQUIRE(RecallAtK(res_dev.value(), truth, kNq, kK) == 1.0f);
    REQUIRE(RecallAtK(res_cpu.value(), truth, kNq, kK) == 1.0f);

    knowhere::fl::FlMemoryPool::Instance().Configure(0, 0);
    knowhere::fl::UnregisterHostAdapter();
}

TEST_CASE("FL_IVF_PQ device ADC and CPU fallback agree", "[fl]") {
    // Host adapter registered -> index stages codebooks/codes on the emulated
    // device and scans via ops->ivf_pq_scan. A second index built with a
    // 1-byte pool cap fails the OOM pre-check (D5) and scans on CPU. Both
    // must return identical top-k ids (deterministic ADC semantics).
    knowhere::fl::UnregisterHostAdapter();
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

    knowhere::fl::UnregisterHostAdapter();
}

TEST_CASE("FL_IVF_PQ serialize/deserialize round trip", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();
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

TEST_CASE("FL device router round-robin", "[fl]") {
    auto& router = knowhere::fl::FlDeviceRouter::Instance();
    router.Reset();
    knowhere::fl::UnregisterHostAdapter();
    // no adapter -> single-device semantics
    REQUIRE(router.Acquire() == 0);
    // host adapter reports exactly 1 emulated device -> always 0
    REQUIRE(knowhere::fl::RegisterHostAdapter());
    REQUIRE(router.Acquire() == 0);
    REQUIRE(router.Acquire() == 0);
    knowhere::fl::UnregisterHostAdapter();
}

TEST_CASE("FL_BRUTE_FORCE GetVectorByIds returns raw vectors", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();
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
    knowhere::fl::UnregisterHostAdapter();
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
    knowhere::fl::UnregisterHostAdapter();
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
        kFileSize, kRows, kDimV, *knowhere::FlBruteForceIndexNode<knowhere::fp32>::StaticCreateConfig(), IndexVersion(0));
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
    knowhere::fl::UnregisterHostAdapter();  // CPU traversal path
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
    knowhere::fl::UnregisterHostAdapter();
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
    knowhere::fl::UnregisterHostAdapter();
    // v1 CPU builder is quadratic and refused beyond kMaxCpuGraphRows; the
    // rejection happens before any O(nb^2) work, so a tiny dim keeps this fast
    auto huge = GenDataSet(knowhere::FlCagraIndexNode<knowhere::fp32>::kMaxCpuGraphRows + 1, 2, 1);
    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_CAGRA);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":2,"graph_degree":8})");
    REQUIRE(idx.Build(huge, build) == knowhere::Status::invalid_args);
}

TEST_CASE("FL_CAGRA post-filters bitset from traversal results", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();
    auto base = GenDataSet(kNb, kDim, 42);
    auto query = GenDataSet(1, kDim, 7);

    auto idx = CreateFlIndex(knowhere::IndexEnum::INDEX_FL_CAGRA);
    knowhere::Json build = knowhere::Json::parse(R"({"metric_type":"L2","dim":16,"graph_degree":16})");
    REQUIRE(idx.Build(base, build) == knowhere::Status::success);

    knowhere::Json search = knowhere::Json::parse(R"({"metric_type":"L2","k":10,"itopk_size":256})");
    auto unfiltered = idx.Search(query, search, nullptr);
    REQUIRE(unfiltered.has_value());
    const int64_t banned = unfiltered.value()->GetIds()[0];

    std::vector<uint8_t> bits(kNb, 0);
    bits[static_cast<size_t>(banned)] = 1;
    auto filtered = idx.Search(query, search, knowhere::BitsetView(bits.data(), kNb));
    REQUIRE(filtered.has_value());
    const auto* fids = filtered.value()->GetIds();
    for (int64_t t = 0; t < kK; ++t) {
        REQUIRE(fids[t] != banned);
    }
}

TEST_CASE("FL_CAGRA GetVectorByIds returns raw vectors", "[fl]") {
    knowhere::fl::UnregisterHostAdapter();
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

    std::vector<uint8_t> bits(kNb, 0);
    bits[static_cast<size_t>(banned)] = 1;
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
    knowhere::fl::UnregisterHostAdapter();
}

#endif  // KNOWHERE_WITH_FL
