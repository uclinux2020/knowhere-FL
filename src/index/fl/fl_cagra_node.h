// Copyright (C) 2026 Knowhere-FL contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

#ifndef FL_CAGRA_NODE_H
#define FL_CAGRA_NODE_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "common/fl/fl_device.h"
#include "common/fl/fl_device_router.h"
#include "common/fl/fl_graph_search_cpu.h"
#include "index/fl/fl_cagra_config.h"
#include "index/fl/fl_format.h"
#include "io/memory_io.h"
#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_node.h"
#include "knowhere/index/index_static.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

// ---------------------------------------------------------------------------
// FlCagraIndexNode (FL_CAGRA) -- Phase 2.3 v1
//
// v1 scope (per plan, with the documented adapt_for_cpu-style fallback):
//   - Build: CPU brute-force kNN graph construction (O(nb^2 * dim)). For nb
//     above kMaxCpuGraphRows the build is refused with a clear message --
//     use IVF_PQ until the device graph builder (FlagGems-composed distance
//     + per-row top-k, Phase 2.3 device milestone) lands.
//   - Search: device graph_search kernel when staged; otherwise the shared
//     CPU greedy traversal (common/fl/fl_graph_search_cpu.h) -- this IS the
//     sanctioned adapt_for_cpu mode: graph resident on host, search on CPU.
//
// Unlike the official GPU_CAGRA this node keeps raw fp32 vectors (needed to
// score neighbors during traversal), so HasRawData() == true and the memory
// estimate includes both vectors and the graph CSR.
// ---------------------------------------------------------------------------
template <typename DataType>
class FlCagraIndexNode : public IndexNode {
 public:
    // CPU brute-force graph build is quadratic; refuse beyond this many rows
    // until the device builder lands (Phase 2.3 device milestone).
    static constexpr int64_t kMaxCpuGraphRows = 200000;

    FlCagraIndexNode(const int32_t version, const Object& object) : IndexNode(version) {
        static_assert(std::is_same_v<DataType, fp32>, "FlCagraIndexNode currently supports fp32 only");
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool /*use_knowhere_build_pool*/) override {
        const FlCagraConfig& f_cfg = static_cast<const FlCagraConfig&>(*cfg);
        const std::string metric = f_cfg.metric_type.value();
        if (metric == "L2") {
            metric_code_ = 0;
        } else if (metric == "IP") {
            metric_code_ = 1;
        } else {
            LOG_KNOWHERE_ERROR_ << "FL_CAGRA unsupported metric: " << metric;
            return Status::invalid_metric_type;
        }
        graph_degree_ = f_cfg.graph_degree.value();
        dim_ = dataset->GetDim();
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> /*cfg*/, bool /*use_knowhere_build_pool*/) override {
        const int64_t rows = dataset->GetRows();
        const auto* x = static_cast<const float*>(dataset->GetTensor());
        if (x == nullptr || rows <= 0 || dim_ <= 0) {
            return Status::empty_index;
        }
        if (rows > kMaxCpuGraphRows) {
            LOG_KNOWHERE_ERROR_ << "FL_CAGRA v1 CPU graph build limited to " << kMaxCpuGraphRows
                                << " rows, got " << rows
                                << "; use FL_IVF_PQ for larger sets until the device graph builder lands";
            return Status::invalid_args;
        }
        nb_ = rows;
        const size_t bytes = static_cast<size_t>(nb_) * dim_ * sizeof(float);
        host_xb_.resize(static_cast<size_t>(nb_) * dim_);
        std::memcpy(host_xb_.data(), x, bytes);

        BuildGraphCpu();
        TryStageToDevice();
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* /*op_context*/ = nullptr) const override {
        const FlCagraConfig& f_cfg = static_cast<const FlCagraConfig&>(*cfg);
        const int64_t nq = dataset->GetRows();
        const int64_t k = f_cfg.k.value();
        const int64_t itopk = std::max<int64_t>(f_cfg.itopk_size.value(), k);
        const auto* xq = static_cast<const float*>(dataset->GetTensor());
        if (nb_ == 0 || xq == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "FL_CAGRA index is empty");
        }

        std::vector<float> dist(static_cast<size_t>(nq) * k);
        std::vector<int64_t> ids(static_cast<size_t>(nq) * k);

        bool done = false;
        if (dev_ready_) {
            const fl::FlDeviceOps* ops = fl::GetDeviceOps();
            if (ops->set_device != nullptr && ops->set_device(device_id_) != 0) {
                LOG_KNOWHERE_WARNING_ << "FL set_device failed in CAGRA search";
            } else if (ops->graph_search != nullptr) {
                const size_t q_bytes = static_cast<size_t>(nq) * dim_ * sizeof(float);
                void* d_xq = ops->malloc(q_bytes);
                if (d_xq != nullptr && ops->memcpy_h2d(d_xq, xq, q_bytes, nullptr) == 0) {
                    done = ops->graph_search(dev_xb_.get(), nb_, dim_, graph_ids_.data(), graph_offsets_.data(),
                                             graph_degree_, d_xq, nq, k, itopk, metric_code_, dist.data(), ids.data(),
                                             nullptr) == 0;
                }
                if (d_xq != nullptr) {
                    ops->free(d_xq);
                }
            }
            if (!done) {
                LOG_KNOWHERE_WARNING_ << "FL_CAGRA device traversal failed, CPU fallback (adapt_for_cpu mode)";
            }
        }
        if (!done) {
            fl::FlGraphSearchCpu(host_xb_.data(), nb_, dim_, graph_ids_.data(), graph_offsets_.data(), graph_degree_,
                                 xq, nq, k, itopk, metric_code_, dist.data(), ids.data());
        }

        // post-filter bitset (official GPU_CAGRA rejects filtered search; the
        // FL path degrades gracefully by filtering the traversal results)
        if (!bitset.empty()) {
            for (int64_t i = 0; i < nq; ++i) {
                int64_t w = 0;
                for (int64_t t = 0; t < k; ++t) {
                    const int64_t id = ids[i * k + t];
                    if (id >= 0 && !bitset.test(id)) {
                        ids[i * k + w] = id;
                        dist[i * k + w] = dist[i * k + t];
                        ++w;
                    }
                }
                for (int64_t t = w; t < k; ++t) {
                    ids[i * k + t] = -1;
                    dist[i * k + t] = (metric_code_ == 1) ? -std::numeric_limits<float>::max()
                                                          : std::numeric_limits<float>::max();
                }
            }
        }

        return GenResultDataSet(nq, k, ids.data(), dist.data());
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr /*dataset*/, std::unique_ptr<Config> /*cfg*/, const BitsetView& /*bitset*/,
                milvus::OpContext* /*op_context*/ = nullptr) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented,
                                         "range search is not supported on FL_CAGRA (same as official GPU_CAGRA)");
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
        std::vector<float> out(in_ids.size() * static_cast<size_t>(dim_));
        for (size_t i = 0; i < in_ids.size(); ++i) {
            std::memcpy(out.data() + i * dim_, host_xb_.data() + in_ids[i] * dim_, dim_ * sizeof(float));
        }
        return GenDataSet(static_cast<int64_t>(in_ids.size()), dim_, out.data());
    }

    bool
    HasRawData(const std::string& /*metric_type*/) const override {
        return true;  // traversal scores against raw fp32 vectors
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> /*cfg*/) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "FL_CAGRA does not expose index meta");
    }

    Status
    Serialize(BinarySet& binset) const override {
        try {
            MemoryIOWriter writer;
            fl::WriteFormatHeader(writer);
            writer(&dim_, sizeof(dim_), 1);
            writer(&nb_, sizeof(nb_), 1);
            writer(&graph_degree_, sizeof(graph_degree_), 1);
            writer(&metric_code_, sizeof(metric_code_), 1);
            writer(host_xb_.data(), sizeof(float), host_xb_.size());
            writer(graph_ids_.data(), sizeof(int64_t), graph_ids_.size());
            writer(graph_offsets_.data(), sizeof(int64_t), graph_offsets_.size());
            std::shared_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[writer.tellg()]);
            if (data == nullptr) {
                return Status::malloc_error;
            }
            std::memcpy(data.get(), writer.data(), writer.tellg());
            binset.Append(Type(), data, writer.tellg());
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "FL_CAGRA serialize failed: " << e.what();
            return Status::inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> /*cfg*/) override {
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            return Status::invalid_binary_set;
        }
        MemoryIOReader reader(binary->data.get(), binary->size);
        if (!fl::ReadFormatHeader(reader, "FL_CAGRA")) {
            return Status::format_error;
        }
        reader(&dim_, sizeof(dim_), 1);
        reader(&nb_, sizeof(nb_), 1);
        reader(&graph_degree_, sizeof(graph_degree_), 1);
        reader(&metric_code_, sizeof(metric_code_), 1);
        host_xb_.resize(static_cast<size_t>(nb_) * dim_);
        reader(host_xb_.data(), sizeof(float), host_xb_.size());
        graph_ids_.resize(static_cast<size_t>(nb_) * graph_degree_);
        reader(graph_ids_.data(), sizeof(int64_t), graph_ids_.size());
        graph_offsets_.resize(static_cast<size_t>(nb_) + 1);
        reader(graph_offsets_.data(), sizeof(int64_t), graph_offsets_.size());
        TryStageToDevice();
        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& /*filename*/, std::shared_ptr<Config> /*cfg*/) override {
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FlCagraConfig>();
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& /*config*/, const IndexVersion& /*version*/) {
        return true;
    }

    // Device/host cost = raw vectors + CSR graph (ids int64 per edge).
    static expected<Resource>
    StaticEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                               const knowhere::BaseConfig& config, const IndexVersion& /*version*/) {
        int64_t degree = 32;  // FlCagraConfig default
        if (const auto* cagra_cfg = dynamic_cast<const FlCagraConfig*>(&config)) {
            degree = cagra_cfg->graph_degree.value();
        }
        Resource res;
        res.memoryCost = static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(dim) * sizeof(float) +
                         static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(std::max<int64_t>(degree, 1)) *
                             sizeof(int64_t);
        (void)file_size_in_bytes;
        res.diskCost = 0;
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
        return static_cast<int64_t>(host_xb_.size() * sizeof(float) +
                                    (graph_ids_.size() + graph_offsets_.size()) * sizeof(int64_t));
    }

    int64_t
    Count() const override {
        return nb_;
    }

    std::string
    Type() const override {
        return IndexEnum::INDEX_FL_CAGRA;
    }

 private:
    // ---- CPU kNN graph build (v1; device builder is the Phase 2.3 milestone)
    void
    BuildGraphCpu() {
        const int64_t m = std::min<int64_t>(graph_degree_, nb_ - 1);
        graph_degree_ = m;  // effective degree (small sets)
        graph_offsets_.resize(static_cast<size_t>(nb_) + 1);
        for (int64_t v = 0; v <= nb_; ++v) {
            graph_offsets_[static_cast<size_t>(v)] = v * m;
        }
        graph_ids_.resize(static_cast<size_t>(nb_) * m, -1);

        const bool is_ip = (metric_code_ == 1);
#pragma omp parallel for schedule(dynamic)
        for (int64_t v = 0; v < nb_; ++v) {
            const float* vv = host_xb_.data() + v * dim_;
            std::vector<std::pair<float, int64_t>> all;
            all.reserve(static_cast<size_t>(nb_));
            for (int64_t u = 0; u < nb_; ++u) {
                if (u == v) {
                    continue;
                }
                all.emplace_back(fl::FlGraphDistance(vv, host_xb_.data() + u * dim_, dim_, metric_code_), u);
            }
            auto better = [is_ip](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
                return is_ip ? (a.first > b.first) : (a.first < b.first);
            };
            std::partial_sort(all.begin(), all.begin() + m, all.end(), better);
            for (int64_t e = 0; e < m; ++e) {
                graph_ids_[static_cast<size_t>(v) * m + e] = all[static_cast<size_t>(e)].second;
            }
        }
        LOG_KNOWHERE_INFO_ << "FL_CAGRA graph built: nb=" << nb_ << " degree=" << m << " dim=" << dim_;
    }

    void
    TryStageToDevice() {
        dev_ready_ = false;
        dev_xb_.reset();
        const fl::FlDeviceOps* ops = fl::GetDeviceOps();
        if (ops == nullptr || !fl::DeviceAvailable()) {
            return;
        }
        if (ops->set_device == nullptr || ops->malloc == nullptr || ops->memcpy_h2d == nullptr) {
            return;
        }
        device_id_ = fl::FlDeviceRouter::Instance().Acquire();
        if (device_id_ < 0 || ops->set_device(device_id_) != 0) {
            return;
        }
        const size_t bytes = static_cast<size_t>(nb_) * dim_ * sizeof(float);
        if (!fl::FlMemoryPool::Instance().Ensure(bytes)) {
            LOG_KNOWHERE_WARNING_ << "FL_CAGRA staging " << bytes << "B exceeds pool cap; staying on CPU (D5)";
            return;
        }
        void* d_xb = ops->malloc(bytes);
        if (d_xb == nullptr) {
            return;
        }
        if (ops->memcpy_h2d(d_xb, host_xb_.data(), bytes, nullptr) != 0) {
            ops->free(d_xb);
            return;
        }
        dev_xb_ = fl::FlDeviceBuffer(d_xb, bytes);
        fl::FlMemoryPool::Instance().TrackAlloc(bytes);
        dev_ready_ = true;
    }

    int64_t dim_{0};
    int64_t nb_{0};
    int64_t graph_degree_{0};  // effective (clamped) degree
    int metric_code_{0};       // 0 = L2, 1 = IP

    std::vector<float> host_xb_;        // nb * dim
    std::vector<int64_t> graph_ids_;    // nb * graph_degree_ CSR payload
    std::vector<int64_t> graph_offsets_;  // nb + 1 (uniform rows)

    bool dev_ready_{false};
    int device_id_{0};  // bound at staging time via FlDeviceRouter
    fl::FlDeviceBuffer dev_xb_;
};

}  // namespace knowhere

#endif  // FL_CAGRA_NODE_H
