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

#ifndef FL_IVF_FLAT_NODE_H
#define FL_IVF_FLAT_NODE_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "knowhere/index/index_static.h"
#include "common/fl/fl_device.h"
#include "common/fl/fl_device_router.h"
#include "index/fl/fl_format.h"
#include "index/fl/fl_ivf_flat_config.h"
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
// FlIvfFlatIndexNode (GPU_IVF_FLAT / FL_IVF_FLAT on the FL path)
//
// v1 design (per plan, Phase 1.2/1.3 with the documented fallback):
//   - Train: Lloyd kmeans. Assignment (mm+argmin, no atomics) runs on the
//     device via ops->kmeans_assign when available (xb staged once per Train);
//     centroid update stays on CPU until the PPU atomics micro-benchmark
//     (python/fl_kernels/kmeans_composition.py) clears index_add performance.
//   - Add: assign each vector to its nearest centroid, build inverted lists,
//     stage raw vectors to the FL device (with OOM pre-check, design D5).
//   - Search: centroid routing on host (nq x nlist, cheap), candidate member
//     ids gathered per query, scan via ops->ivf_flat_scan on device; CPU
//     fallback scans the same candidate lists.
// ---------------------------------------------------------------------------
template <typename DataType>
class FlIvfFlatIndexNode : public IndexNode {
 public:
    FlIvfFlatIndexNode(const int32_t version, const Object& object) : IndexNode(version) {
        static_assert(std::is_same_v<DataType, fp32>, "FlIvfFlatIndexNode currently supports fp32 only");
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool /*use_knowhere_build_pool*/) override {
        const FlIvfFlatConfig& f_cfg = static_cast<const FlIvfFlatConfig&>(*cfg);
        const std::string metric = f_cfg.metric_type.value();
        if (metric == "L2") {
            metric_code_ = 0;
        } else if (metric == "IP") {
            metric_code_ = 1;
        } else {
            return Status::invalid_metric_type;
        }
        dim_ = dataset->GetDim();
        nlist_ = std::min<int64_t>(f_cfg.nlist.value(), dataset->GetRows());
        if (nlist_ <= 0) {
            return Status::invalid_args;
        }
        use_device_assign_ = f_cfg.use_device_assign.value();

        const auto* x = static_cast<const float*>(dataset->GetTensor());
        const int64_t nb = dataset->GetRows();
        if (x == nullptr || nb < nlist_) {
            LOG_KNOWHERE_ERROR_ << "FL IVF_FLAT train: invalid dataset";
            return Status::invalid_args;
        }
        TrainKmeans(x, nb);
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> /*cfg*/, bool /*use_knowhere_build_pool*/) override {
        if (centroids_.empty()) {
            return Status::index_not_trained;
        }
        const auto* x = static_cast<const float*>(dataset->GetTensor());
        const int64_t rows = dataset->GetRows();
        if (x == nullptr || rows <= 0) {
            return Status::empty_index;
        }
        nb_ = rows;
        host_xb_.assign(x, x + static_cast<size_t>(nb_) * dim_);

        invlists_.assign(nlist_, {});
        assign_.resize(nb_);
        AssignAll(host_xb_.data(), nb_, assign_.data());
        for (int64_t j = 0; j < nb_; ++j) {
            invlists_[assign_[j]].push_back(j);
        }

        TryStageToDevice();
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* /*op_context*/ = nullptr) const override {
        const FlIvfFlatConfig& f_cfg = static_cast<const FlIvfFlatConfig&>(*cfg);
        const int64_t nq = dataset->GetRows();
        const int64_t k = f_cfg.k.value();
        const auto* xq = static_cast<const float*>(dataset->GetTensor());
        if (nb_ == 0 || xq == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "FL IVF_FLAT index is empty");
        }
        const int64_t nprobe = std::min<int64_t>(f_cfg.nprobe.value(), nlist_);

        // 1) centroid routing on host: per query pick nprobe nearest lists
        std::vector<int64_t> cand_ids;
        std::vector<int64_t> cand_offsets(static_cast<size_t>(nq) + 1, 0);
        BuildCandidateLists(xq, nq, nprobe, bitset, cand_ids, cand_offsets);

        // 2) scan candidates: device path first, CPU fallback (design D5)
        auto* distances = new (std::nothrow) float[nq * k];
        auto* ids = new (std::nothrow) int64_t[nq * k];
        if (distances == nullptr || ids == nullptr) {
            delete[] distances;
            delete[] ids;
            return expected<DataSetPtr>::Err(Status::malloc_error, "host result alloc failed");
        }

        bool done = false;
        if (dev_ready_) {
            const fl::FlDeviceOps* ops = fl::GetDeviceOps();
            if (ops->set_device != nullptr && ops->set_device(device_id_) != 0) {
                LOG_KNOWHERE_WARNING_ << "FL set_device failed in IVF_FLAT search";
            } else if (ops->ivf_flat_scan != nullptr) {
                const size_t q_bytes = static_cast<size_t>(nq) * dim_ * sizeof(float);
                void* d_q = ops->malloc(q_bytes);
                if (d_q != nullptr && ops->memcpy_h2d(d_q, xq, q_bytes, nullptr) == 0) {
                    done = ops->ivf_flat_scan(dev_xb_.get(), nb_, dim_, d_q, nq, cand_ids.data(), cand_offsets.data(),
                                              k, metric_code_, distances, ids, nullptr) == 0;
                }
                if (d_q != nullptr) {
                    ops->free(d_q);
                }
            }
            if (!done) {
                LOG_KNOWHERE_WARNING_ << "FL device ivf scan unavailable/failed, CPU fallback";
            }
        }
        if (!done) {
            CpuScanCandidates(xq, nq, k, cand_ids, cand_offsets, distances, ids);
        }
        if (!bitset.empty()) {
            ApplyBitsetPostFilter(distances, ids, nq, k, bitset);
        }
        return GenResultDataSet(nq, k, ids, distances);
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr, std::unique_ptr<Config>, const BitsetView&,
                milvus::OpContext* = nullptr) const override {
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
        return true;  // FL IVF_FLAT retains raw fp32 vectors (host_xb_)
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config>) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (nb_ == 0 || centroids_.empty()) {
            return Status::empty_index;
        }
        try {
            MemoryIOWriter writer;
            fl::WriteFormatHeader(writer);
            writer(&dim_, sizeof(dim_), 1);
            writer(&nb_, sizeof(nb_), 1);
            writer(&nlist_, sizeof(nlist_), 1);
            writer(&metric_code_, sizeof(metric_code_), 1);
            writer(centroids_.data(), sizeof(float), centroids_.size());
            writer(host_xb_.data(), sizeof(float), host_xb_.size());
            for (const auto& list : invlists_) {
                int64_t sz = static_cast<int64_t>(list.size());
                writer(&sz, sizeof(sz), 1);
                if (sz > 0) {
                    writer(list.data(), sizeof(int64_t), sz);
                }
            }
            std::shared_ptr<uint8_t[]> data(writer.data());
            binset.Append(Type(), data, writer.tellg());
            return Status::success;
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "FL IVF_FLAT serialize error: " << e.what();
            return Status::knowhere_inner_error;
        }
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> /*cfg*/) override {
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            return Status::invalid_binary_set;
        }
        MemoryIOReader reader(binary->data.get(), binary->size);
        if (!fl::ReadFormatHeader(reader, "IVF_FLAT")) {
            return Status::invalid_binary_set;
        }
        reader(&dim_, sizeof(dim_), 1);
        reader(&nb_, sizeof(nb_), 1);
        reader(&nlist_, sizeof(nlist_), 1);
        reader(&metric_code_, sizeof(metric_code_), 1);
        centroids_.resize(static_cast<size_t>(nlist_) * dim_);
        reader(centroids_.data(), sizeof(float), centroids_.size());
        host_xb_.resize(static_cast<size_t>(nb_) * dim_);
        reader(host_xb_.data(), sizeof(float), host_xb_.size());
        invlists_.assign(nlist_, {});
        assign_.resize(nb_);
        for (int64_t l = 0; l < nlist_; ++l) {
            int64_t sz = 0;
            reader(&sz, sizeof(sz), 1);
            invlists_[l].resize(sz);
            if (sz > 0) {
                reader(invlists_[l].data(), sizeof(int64_t), sz);
                for (auto id : invlists_[l]) {
                    assign_[id] = l;
                }
            }
        }
        TryStageToDevice();
        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& /*filename*/, std::shared_ptr<Config> /*cfg*/) override {
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FlIvfFlatConfig>();
    }

    static bool
    StaticHasRawData(const knowhere::BaseConfig& /*config*/, const IndexVersion& /*version*/) {
        return true;  // FL IVF_FLAT retains raw fp32 vectors
    }

    // Device-resident cost ~= raw vector payload; centroids + invlists are
    // host-side and negligible relative to nb*dim*4.
    static expected<Resource>
    StaticEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                               const knowhere::BaseConfig& /*config*/, const IndexVersion& /*version*/) {
        Resource res;
        res.memoryCost = static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(dim) * sizeof(float);
        (void)file_size_in_bytes;  // layout-dependent; row/col estimate is stabler
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
        return nb_ * dim_ * sizeof(float) + nlist_ * dim_ * sizeof(float);
    }

    int64_t
    Count() const override {
        return nb_;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FL_IVFFLAT;
    }

 private:
    // ---- Lloyd kmeans: device assignment (mm+argmin, no atomics) + CPU update ----
    // Per plan Phase 1.2 fallback: centroid *update* (atomic index_add) stays
    // on CPU until kmeans_composition.py clears PPU atomics performance; the
    // assignment step (the dominant O(nb*nlist*dim) cost) runs on the device
    // whenever the adapter provides kmeans_assign and use_device_assign=true.
    void
    TrainKmeans(const float* x, int64_t nb) {
        constexpr int kMaxIters = 25;
        std::mt19937_64 rng(42);
        centroids_.resize(static_cast<size_t>(nlist_) * dim_);

        // init: random distinct samples
        std::vector<int64_t> perm(nb);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t l = 0; l < nlist_; ++l) {
            std::memcpy(centroids_.data() + l * dim_, x + perm[l] * dim_, dim_ * sizeof(float));
        }

        // Stage xb on the device once for the whole training loop; centroids
        // are re-uploaded every iteration (they are tiny: nlist * dim).
        const fl::FlDeviceOps* ops = nullptr;
        void* d_x = nullptr;
        void* d_c = nullptr;
        size_t staged_bytes = 0;
        const size_t x_bytes = static_cast<size_t>(nb) * dim_ * sizeof(float);
        const size_t c_bytes = static_cast<size_t>(nlist_) * dim_ * sizeof(float);
        if (use_device_assign_ && fl::DeviceAvailable()) {
            ops = fl::GetDeviceOps();
            if (ops->kmeans_assign != nullptr && ops->malloc != nullptr && ops->free != nullptr &&
                ops->memcpy_h2d != nullptr && ops->set_device != nullptr &&
                ops->set_device(fl::FlDeviceRouter::Instance().Acquire()) == 0 &&
                fl::FlMemoryPool::Instance().Ensure(x_bytes + c_bytes)) {
                d_x = ops->malloc(x_bytes);
                d_c = ops->malloc(c_bytes);
                if (d_x != nullptr && d_c != nullptr && ops->memcpy_h2d(d_x, x, x_bytes, nullptr) == 0) {
                    staged_bytes = x_bytes + c_bytes;
                    fl::FlMemoryPool::Instance().TrackAlloc(staged_bytes);
                    LOG_KNOWHERE_INFO_ << "FL IVF_FLAT kmeans: device assignment enabled";
                } else {
                    if (d_x != nullptr) {
                        ops->free(d_x);
                    }
                    if (d_c != nullptr) {
                        ops->free(d_c);
                    }
                    d_x = d_c = nullptr;
                }
            }
        }

        std::vector<int64_t> owner(nb, -1);
        std::vector<int64_t> cur(nb, -1);
        for (int iter = 0; iter < kMaxIters; ++iter) {
            bool assigned_on_device = false;
            if (d_x != nullptr) {
                assigned_on_device =
                    ops->memcpy_h2d(d_c, centroids_.data(), c_bytes, nullptr) == 0 &&
                    ops->kmeans_assign(d_x, nb, dim_, d_c, nlist_, metric_code_, cur.data(), nullptr) == 0;
                if (!assigned_on_device) {
                    LOG_KNOWHERE_WARNING_ << "FL device kmeans_assign failed mid-train; CPU routing from now on";
                    ops->free(d_x);
                    ops->free(d_c);
                    fl::FlMemoryPool::Instance().TrackFree(staged_bytes);
                    d_x = d_c = nullptr;
                }
            }
            if (!assigned_on_device) {
#pragma omp parallel for schedule(dynamic)
                for (int64_t j = 0; j < nb; ++j) {
                    cur[j] = NearestCentroid(x + j * dim_);
                }
            }
            bool changed = false;
#pragma omp parallel for schedule(static) reduction(| : changed)
            for (int64_t j = 0; j < nb; ++j) {
                if (cur[j] != owner[j]) {
                    owner[j] = cur[j];
                    changed = true;
                }
            }
            // recompute centroids (CPU; device update needs atomics, see header note)
            std::vector<double> sums(static_cast<size_t>(nlist_) * dim_, 0.0);
            std::vector<int64_t> counts(nlist_, 0);
            for (int64_t j = 0; j < nb; ++j) {
                const int64_t c = owner[j];
                ++counts[c];
                for (int64_t d = 0; d < dim_; ++d) {
                    sums[static_cast<size_t>(c) * dim_ + d] += x[j * dim_ + d];
                }
            }
            for (int64_t l = 0; l < nlist_; ++l) {
                if (counts[l] == 0) {  // empty cluster: re-seed randomly
                    std::memcpy(centroids_.data() + l * dim_, x + perm[(l + iter + 1) % nb] * dim_,
                                dim_ * sizeof(float));
                } else {
                    for (int64_t d = 0; d < dim_; ++d) {
                        centroids_[static_cast<size_t>(l) * dim_ + d] =
                            static_cast<float>(sums[static_cast<size_t>(l) * dim_ + d] / counts[l]);
                    }
                }
            }
            if (!changed) {
                break;
            }
        }
        if (d_x != nullptr) {
            ops->free(d_x);
            ops->free(d_c);
            fl::FlMemoryPool::Instance().TrackFree(staged_bytes);
        }
        LOG_KNOWHERE_INFO_ << "FL IVF_FLAT kmeans trained: nlist=" << nlist_ << " dim=" << dim_
                           << (staged_bytes != 0 ? " (device assignment)" : " (CPU assignment)");
    }

    // One-shot assignment of vectors to centroids: device kmeans_assign when
    // available (single upload), CPU parallel routing otherwise.
    void
    AssignAll(const float* x, int64_t nb, int64_t* out) const {
        if (use_device_assign_ && fl::DeviceAvailable()) {
            const fl::FlDeviceOps* ops = fl::GetDeviceOps();
            const size_t x_bytes = static_cast<size_t>(nb) * dim_ * sizeof(float);
            const size_t c_bytes = static_cast<size_t>(nlist_) * dim_ * sizeof(float);
            if (ops->kmeans_assign != nullptr && ops->malloc != nullptr && ops->free != nullptr &&
                ops->memcpy_h2d != nullptr && ops->set_device != nullptr &&
                ops->set_device(fl::FlDeviceRouter::Instance().Acquire()) == 0 &&
                fl::FlMemoryPool::Instance().Ensure(x_bytes + c_bytes)) {
                void* d_x = ops->malloc(x_bytes);
                void* d_c = ops->malloc(c_bytes);
                const bool alloc_ok = d_x != nullptr && d_c != nullptr;
                if (alloc_ok) {
                    fl::FlMemoryPool::Instance().TrackAlloc(x_bytes + c_bytes);
                }
                bool ok = alloc_ok && ops->memcpy_h2d(d_x, x, x_bytes, nullptr) == 0 &&
                          ops->memcpy_h2d(d_c, centroids_.data(), c_bytes, nullptr) == 0 &&
                          ops->kmeans_assign(d_x, nb, dim_, d_c, nlist_, metric_code_, out, nullptr) == 0;
                if (d_x != nullptr) {
                    ops->free(d_x);
                }
                if (d_c != nullptr) {
                    ops->free(d_c);
                }
                if (alloc_ok) {
                    fl::FlMemoryPool::Instance().TrackFree(x_bytes + c_bytes);
                }
                if (ok) {
                    return;
                }
                LOG_KNOWHERE_WARNING_ << "FL device assignment failed in Add; CPU routing fallback";
            }
        }
#pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < nb; ++j) {
            out[j] = NearestCentroid(x + j * dim_);
        }
    }

    int64_t
    NearestCentroid(const float* v) const {
        int64_t best = 0;
        float best_score = metric_code_ == 1 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (int64_t l = 0; l < nlist_; ++l) {
            const float* c = centroids_.data() + l * dim_;
            float dot = 0.f, na = 0.f, nc = 0.f;
            for (int64_t d = 0; d < dim_; ++d) {
                dot += v[d] * c[d];
                if (metric_code_ == 0) {
                    na += v[d] * v[d];
                    nc += c[d] * c[d];
                }
            }
            const float score = metric_code_ == 1 ? dot : (na + nc - 2.f * dot);
            if ((metric_code_ == 1 && score > best_score) || (metric_code_ == 0 && score < best_score)) {
                best_score = score;
                best = l;
            }
        }
        return best;
    }

    void
    BuildCandidateLists(const float* xq, int64_t nq, int64_t nprobe, const BitsetView& bitset,
                        std::vector<int64_t>& cand_ids, std::vector<int64_t>& cand_offsets) const {
        // approximate candidate count to reserve once
        size_t approx = 0;
        std::vector<std::vector<int64_t>> probes(nq);
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < nq; ++i) {
            std::vector<std::pair<float, int64_t>> scores(nlist_);
            const float* q = xq + i * dim_;
            for (int64_t l = 0; l < nlist_; ++l) {
                const float* c = centroids_.data() + l * dim_;
                float dot = 0.f, na = 0.f, nc = 0.f;
                for (int64_t d = 0; d < dim_; ++d) {
                    dot += q[d] * c[d];
                    if (metric_code_ == 0) {
                        na += q[d] * q[d];
                        nc += c[d] * c[d];
                    }
                }
                scores[l] = {metric_code_ == 1 ? dot : (na + nc - 2.f * dot), l};
            }
            const bool is_ip = metric_code_ == 1;
            std::partial_sort(scores.begin(), scores.begin() + nprobe, scores.end(),
                              [is_ip](const auto& a, const auto& b) {
                                  return is_ip ? (a.first > b.first) : (a.first < b.first);
                              });
            auto& pl = probes[i];
            for (int64_t p = 0; p < nprobe; ++p) {
                pl.push_back(scores[p].second);
            }
        }
        for (int64_t i = 0; i < nq; ++i) {
            cand_offsets[i] = static_cast<int64_t>(cand_ids.size());
            for (const int64_t l : probes[i]) {
                const auto& list = invlists_[l];
                if (!bitset.empty()) {
                    for (const int64_t id : list) {
                        if (!bitset.test(id)) {
                            cand_ids.push_back(id);
                        }
                    }
                } else {
                    cand_ids.insert(cand_ids.end(), list.begin(), list.end());
                }
            }
        }
        cand_offsets[nq] = static_cast<int64_t>(cand_ids.size());
    }

    void
    CpuScanCandidates(const float* xq, int64_t nq, int64_t k, const std::vector<int64_t>& cand_ids,
                      const std::vector<int64_t>& cand_offsets, float* distances, int64_t* ids) const {
        const bool is_ip = metric_code_ == 1;
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < nq; ++i) {
            using Cand = std::pair<float, int64_t>;
            std::vector<Cand> best;
            best.reserve(static_cast<size_t>(k) + 1);
            const float* q = xq + i * dim_;
            for (int64_t c = cand_offsets[i]; c < cand_offsets[i + 1]; ++c) {
                const int64_t id = cand_ids[c];
                const float* b = host_xb_.data() + id * dim_;
                float dot = 0.f, na = 0.f, nbsum = 0.f;
                for (int64_t d = 0; d < dim_; ++d) {
                    dot += q[d] * b[d];
                    if (!is_ip) {
                        na += q[d] * q[d];
                        nbsum += b[d] * b[d];
                    }
                }
                const float dist = is_ip ? dot : (na + nbsum - 2.f * dot);
                best.emplace_back(dist, id);
            }
            const int64_t nk = std::min<int64_t>(k, best.size());
            std::partial_sort(best.begin(), best.begin() + nk, best.end(), [is_ip](const Cand& a, const Cand& b) {
                return is_ip ? (a.first > b.first) : (a.first < b.first);
            });
            for (int64_t t = 0; t < k; ++t) {
                if (t < nk) {
                    distances[i * k + t] = best[t].first;
                    ids[i * k + t] = best[t].second;
                } else {
                    distances[i * k + t] =
                        is_ip ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
                    ids[i * k + t] = -1;
                }
            }
        }
    }

    void
    TryStageToDevice() {
        dev_ready_ = false;
        if (!fl::DeviceAvailable()) {
            return;
        }
        const fl::FlDeviceOps* ops = fl::GetDeviceOps();
        const size_t bytes = host_xb_.size() * sizeof(float);
        if (!fl::FlMemoryPool::Instance().Ensure(bytes)) {
            LOG_KNOWHERE_WARNING_ << "FL memory pool rejects staging " << bytes << " bytes; IVF_FLAT on CPU fallback";
            return;
        }
        if (ops->set_device == nullptr || ops->malloc == nullptr || ops->memcpy_h2d == nullptr) {
            return;
        }
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
        LOG_KNOWHERE_INFO_ << "FL IVF_FLAT staged " << nb_ << " x " << dim_ << " vectors on device";
    }

    static void
    ApplyBitsetPostFilter(float* distances, int64_t* ids, int64_t nq, int64_t k, const BitsetView& bitset) {
        for (int64_t i = 0; i < nq; ++i) {
            for (int64_t t = 0; t < k; ++t) {
                const int64_t id = ids[i * k + t];
                if (id >= 0 && bitset.test(id)) {
                    ids[i * k + t] = -1;
                    distances[i * k + t] = std::numeric_limits<float>::max();
                }
            }
        }
    }

    int64_t dim_{0};
    int64_t nb_{0};
    int64_t nlist_{0};
    int metric_code_{0};             // 0 = L2, 1 = IP
    bool use_device_assign_{true};   // device centroid routing (train/add)
    std::vector<float> centroids_;           // nlist * dim
    std::vector<float> host_xb_;             // nb * dim
    std::vector<int64_t> assign_;            // nb -> list id
    std::vector<std::vector<int64_t>> invlists_;
    fl::FlDeviceBuffer dev_xb_;
    bool dev_ready_{false};
    int device_id_{0};  // bound at staging time via FlDeviceRouter (multi-device)
};

}  // namespace knowhere

#endif  // FL_IVF_FLAT_NODE_H
