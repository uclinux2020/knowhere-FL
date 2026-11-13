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

#ifndef FL_INDEX_NODE_H
#define FL_INDEX_NODE_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <queue>
#include <string>
#include <vector>

#include "knowhere/index/index_static.h"
#include "common/fl/fl_device.h"
#include "common/fl/fl_device_router.h"
#include "index/fl/fl_brute_force_config.h"
#include "index/fl/fl_format.h"
#include "io/memory_io.h"
#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

// ---------------------------------------------------------------------------
// FlBruteForceIndexNode (GPU_BRUTE_FORCE / FL_BRUTE_FORCE on the FL path)
//
// Search strategy (design D5):
//   1. If an FL device adapter is registered and the memory pool admits the
//      staging bytes -> stage base/query vectors on device and run the
//      adapter's brute_force_topk kernel (FlagGems-composed distance +
//      top-k, AOT-compiled via FlagTree).
//   2. Otherwise (no adapter, no device, or OOM pre-check failure) -> CPU
//      fallback. The fallback is a straightforward OpenMP brute force, used
//      for correctness bring-up and graceful degradation only; it is NOT a
//      performance path (HNSW/IVF remain the CPU performance options).
// ---------------------------------------------------------------------------
template <typename DataType>
class FlBruteForceIndexNode : public IndexNode {
 public:
    FlBruteForceIndexNode(const int32_t version, const Object& object) : IndexNode(version) {
        static_assert(std::is_same_v<DataType, fp32>, "FlBruteForceIndexNode currently supports fp32 only");
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool /*use_knowhere_build_pool*/) override {
        const FlBruteForceConfig& f_cfg = static_cast<const FlBruteForceConfig&>(*cfg);
        std::string metric = f_cfg.metric_type.value();
        if (metric == "L2") {
            metric_code_ = 0;
        } else if (metric == "IP") {
            metric_code_ = 1;
        } else {
            LOG_KNOWHERE_ERROR_ << "FL brute force unsupported metric: " << metric;
            return Status::invalid_metric_type;
        }
        dim_ = dataset->GetDim();
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool /*use_knowhere_build_pool*/) override {
        auto rows = dataset->GetRows();
        const auto* x = static_cast<const float*>(dataset->GetTensor());
        if (x == nullptr || rows <= 0 || dim_ <= 0) {
            return Status::empty_index;
        }
        nb_ = rows;
        const size_t bytes = static_cast<size_t>(nb_) * dim_ * sizeof(float);
        host_xb_.resize(static_cast<size_t>(nb_) * dim_);
        std::memcpy(host_xb_.data(), x, bytes);

        // Best-effort device staging; failure only means CPU fallback later.
        TryStageToDevice();
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* /*op_context*/ = nullptr) const override {
        const FlBruteForceConfig& f_cfg = static_cast<const FlBruteForceConfig&>(*cfg);
        const int64_t nq = dataset->GetRows();
        const int64_t k = f_cfg.k.value();
        const auto* xq = static_cast<const float*>(dataset->GetTensor());

        if (nb_ == 0 || xq == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "FL brute force index is empty");
        }

        // Device path
        if (dev_ready_) {
            auto res = SearchOnDevice(xq, nq, k, bitset);
            if (res.has_value()) {
                return res;
            }
            LOG_KNOWHERE_WARNING_ << "FL device search failed, falling back to CPU: " << res.what();
        }

        // CPU fallback
        return SearchOnCpu(xq, nq, k, bitset);
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr /*dataset*/, std::unique_ptr<Config> /*cfg*/, const BitsetView& /*bitset*/,
                milvus::OpContext* /*op_context*/ = nullptr) const override {
        // Consistent with official GPU indexes: range search not supported.
        return expected<DataSetPtr>::Err(Status::not_implemented,
                                         "range search is not supported on FL GPU path (same as official GPU indexes)");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* /*op_context*/ = nullptr) const override {
        if (dataset == nullptr) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "GetVectorByIds dataset is null");
        }
        std::vector<int64_t> in_ids;
        const Status status = this->CompactOutToIn(dataset->GetIds(), dataset->GetRows(), in_ids);
        if (status != Status::success) {
            return expected<DataSetPtr>::Err(status, "GetVectorByIds failed to map ids");
        }
        const int64_t rows = static_cast<int64_t>(in_ids.size());
        if (rows == 0) {
            return GenResultDataSet(0, dim_, nullptr);
        }
        auto* data = new (std::nothrow) float[static_cast<size_t>(rows) * dim_];
        if (data == nullptr) {
            return expected<DataSetPtr>::Err(Status::malloc_error, "GetVectorByIds alloc failed");
        }
        for (int64_t i = 0; i < rows; ++i) {
            std::memcpy(data + i * dim_, host_xb_.data() + in_ids[i] * dim_, dim_ * sizeof(float));
        }
        return GenResultDataSet(rows, dim_, data);
    }

    bool
    HasRawData(const std::string& /*metric_type*/) const override {
        return true;  // FL brute force always retains raw fp32 vectors (host_xb_)
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> /*cfg*/) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (nb_ == 0) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty FL index.";
            return Status::empty_index;
        }
        try {
            MemoryIOWriter writer;
            fl::WriteFormatHeader(writer);
            writer(&dim_, sizeof(dim_), 1);
            writer(&nb_, sizeof(nb_), 1);
            writer(&metric_code_, sizeof(metric_code_), 1);
            writer(host_xb_.data(), sizeof(float), host_xb_.size());
            std::shared_ptr<uint8_t[]> data(writer.data());
            binset.Append(Type(), data, writer.tellg());
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "FL index serialize error: " << e.what();
            return Status::knowhere_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> /*cfg*/) override {
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set for FL index.";
            return Status::invalid_binary_set;
        }
        MemoryIOReader reader(binary->data.get(), binary->size);
        if (!fl::ReadFormatHeader(reader, "BRUTE_FORCE")) {
            return Status::invalid_binary_set;
        }
        reader(&dim_, sizeof(dim_), 1);
        reader(&nb_, sizeof(nb_), 1);
        reader(&metric_code_, sizeof(metric_code_), 1);
        host_xb_.resize(static_cast<size_t>(nb_) * dim_);
        reader(host_xb_.data(), sizeof(float), host_xb_.size());
        TryStageToDevice();
        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& /*filename*/, std::shared_ptr<Config> /*cfg*/) override {
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FlBruteForceConfig>();
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& /*config*/, const IndexVersion& /*version*/) {
        return true;  // FL brute force always retains raw fp32 vectors
    }

    // Milvus calls this before loading a segment to budget memory. FL brute
    // force stages the whole (already raw fp32) binary onto the device, so
    // the cost is ~1x the file size on device + a transient host copy.
    static expected<Resource>
    StaticEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t /*num_rows*/,
                               const int64_t /*dim*/, const knowhere::BaseConfig& /*config*/,
                               const IndexVersion& /*version*/) {
        Resource res;
        res.memoryCost = file_size_in_bytes;  // device-resident raw vectors
        res.diskCost = 0;                     // mmap/source managed by Milvus
        return res;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    int64_t
    Dim() const override {
        return dim_;
    }

    int64_t
    Size() const override {
        return nb_ * dim_ * sizeof(float);
    }

    int64_t
    Count() const override {
        return nb_;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FL_BRUTEFORCE;
    }

 private:
    void
    TryStageToDevice() {
        dev_ready_ = false;
        if (!fl::DeviceAvailable()) {
            return;
        }
        const fl::FlDeviceOps* ops = fl::GetDeviceOps();
        const size_t bytes = host_xb_.size() * sizeof(float);
        if (!fl::FlMemoryPool::Instance().Ensure(bytes)) {
            LOG_KNOWHERE_WARNING_ << "FL memory pool rejects staging " << bytes
                                  << " bytes; index will run on CPU fallback (design D5)";
            return;
        }
        if (ops->set_device == nullptr || ops->malloc == nullptr || ops->memcpy_h2d == nullptr) {
            return;
        }
        // multi-device: this index is bound to one card for its lifetime
        device_id_ = fl::FlDeviceRouter::Instance().Acquire();
        if (ops->set_device(device_id_) != 0) {
            return;
        }
        void* d = ops->malloc(bytes);
        if (d == nullptr) {
            return;
        }
        if (ops->memcpy_h2d(d, host_xb_.data(), bytes, nullptr) != 0) {
            ops->free(d);
            return;
        }
        dev_xb_ = fl::FlDeviceBuffer(d, bytes);
        fl::FlMemoryPool::Instance().TrackAlloc(bytes);
        dev_ready_ = true;
        LOG_KNOWHERE_INFO_ << "FL brute force staged " << nb_ << " x " << dim_ << " vectors on device";
    }

    expected<DataSetPtr>
    SearchOnDevice(const float* xq, int64_t nq, int64_t k, const BitsetView& bitset) const {
        const fl::FlDeviceOps* ops = fl::GetDeviceOps();
        if (ops->brute_force_topk == nullptr || ops->memcpy_h2d == nullptr) {
            return expected<DataSetPtr>::Err(Status::not_implemented, "FL adapter lacks brute_force_topk");
        }
        if (ops->set_device != nullptr && ops->set_device(device_id_) != 0) {
            return expected<DataSetPtr>::Err(Status::fl_device_error, "FL set_device failed");
        }
        const size_t q_bytes = static_cast<size_t>(nq) * dim_ * sizeof(float);
        void* d_q = ops->malloc(q_bytes);
        if (d_q == nullptr) {
            return expected<DataSetPtr>::Err(Status::fl_device_error, "FL device malloc for query failed");
        }
        fl::FlDeviceBuffer q_guard(d_q, q_bytes);  // not pool-tracked: transient
        if (ops->memcpy_h2d(d_q, xq, q_bytes, nullptr) != 0) {
            return expected<DataSetPtr>::Err(Status::fl_device_error, "FL H2D copy for query failed");
        }

        auto* distances = new (std::nothrow) float[nq * k];
        auto* ids = new (std::nothrow) int64_t[nq * k];
        if (distances == nullptr || ids == nullptr) {
            delete[] distances;
            delete[] ids;
            return expected<DataSetPtr>::Err(Status::malloc_error, "host result alloc failed");
        }
        int rc = ops->brute_force_topk(dev_xb_.get(), nb_, d_q, nq, dim_, k, metric_code_, distances, ids, nullptr);
        if (rc != 0) {
            delete[] distances;
            delete[] ids;
            return expected<DataSetPtr>::Err(Status::fl_device_error, "FL brute_force_topk kernel failed");
        }
        if (!bitset.empty()) {
            ApplyBitsetPostFilter(distances, ids, nq, k, bitset, metric_code_);
        }
        return GenResultDataSet(nq, k, ids, distances);
    }

    expected<DataSetPtr>
    SearchOnCpu(const float* xq, int64_t nq, int64_t k, const BitsetView& bitset) const {
        auto* distances = new (std::nothrow) float[nq * k];
        auto* ids = new (std::nothrow) int64_t[nq * k];
        if (distances == nullptr || ids == nullptr) {
            delete[] distances;
            delete[] ids;
            return expected<DataSetPtr>::Err(Status::malloc_error, "host result alloc failed");
        }

        const bool is_ip = (metric_code_ == 1);
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < nq; ++i) {
            // max-heap of (dist, id); for L2 keep smallest, for IP keep largest.
            using Cand = std::pair<float, int64_t>;
            auto cmp = [is_ip](const Cand& a, const Cand& b) {
                return is_ip ? (a.first > b.first) : (a.first < b.first);
            };
            std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> heap(cmp);
            const float* q = xq + i * dim_;
            // query norm is loop-invariant; compute once per query
            float na = 0.f;
            if (!is_ip) {
                for (int64_t d = 0; d < dim_; ++d) {
                    na += q[d] * q[d];
                }
            }
            for (int64_t j = 0; j < nb_; ++j) {
                if (!bitset.empty() && bitset.test(j)) {
                    continue;
                }
                const float* b = host_xb_.data() + j * dim_;
                float dot = 0.f, nbsum = 0.f;
                for (int64_t d = 0; d < dim_; ++d) {
                    dot += q[d] * b[d];
                    if (!is_ip) {
                        nbsum += b[d] * b[d];
                    }
                }
                float dist = is_ip ? dot : (na + nbsum - 2.f * dot);
                if ((int64_t)heap.size() < k) {
                    heap.emplace(dist, j);
                } else {
                    const float worst = heap.top().first;
                    if ((is_ip && dist > worst) || (!is_ip && dist < worst)) {
                        heap.pop();
                        heap.emplace(dist, j);
                    }
                }
            }
            for (int64_t t = k - 1; t >= 0; --t) {
                if (heap.empty()) {
                    ids[i * k + t] = -1;
                    distances[i * k + t] = is_ip ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
                } else {
                    distances[i * k + t] = heap.top().first;
                    ids[i * k + t] = heap.top().second;
                    heap.pop();
                }
            }
        }
        return GenResultDataSet(nq, k, ids, distances);
    }

    // Post-filter for the device path (the device kernel scans the whole
    // base set, so bitset must be applied on the host afterwards). Results
    // are compacted towards the front (same behavior as FL_CAGRA) so the
    // first `valid` entries are always usable; empty slots are padded with
    // -1 and the metric-appropriate worst distance.
    static void
    ApplyBitsetPostFilter(float* distances, int64_t* ids, int64_t nq, int64_t k, const BitsetView& bitset,
                          int metric_code) {
        const float kWorst =
            (metric_code == 1) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (int64_t i = 0; i < nq; ++i) {
            int64_t w = 0;
            for (int64_t t = 0; t < k; ++t) {
                const int64_t id = ids[i * k + t];
                if (id >= 0 && !bitset.test(id)) {
                    ids[i * k + w] = id;
                    distances[i * k + w] = distances[i * k + t];
                    ++w;
                }
            }
            for (int64_t t = w; t < k; ++t) {
                ids[i * k + t] = -1;
                distances[i * k + t] = kWorst;
            }
        }
    }

    int64_t dim_{0};
    int64_t nb_{0};
    int metric_code_{0};  // 0 = L2, 1 = IP
    std::vector<float> host_xb_;
    fl::FlDeviceBuffer dev_xb_;
    bool dev_ready_{false};
    int device_id_{0};  // bound at staging time via FlDeviceRouter (multi-device)
};

}  // namespace knowhere

#endif  // FL_INDEX_NODE_H
