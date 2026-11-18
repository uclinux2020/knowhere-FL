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

#ifndef KNOWHERE_FL_IVF_FLAT_NODE_H
#define KNOWHERE_FL_IVF_FLAT_NODE_H

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
#include "knowhere/index/index_static.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

// ---------------------------------------------------------------------------
// FlIvfFlatIndexNode (GPU_IVF_FLAT / FL_IVF_FLAT on the FL path).
// One of knowhere-FL's FL_* index nodes, built on the backend-neutral device
// abstraction layer that accelerates vector search on diverse AI chips.
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

        const bool profile = std::getenv("KNOWHERE_FL_PPU_PROFILE") != nullptr;
        using Clock = std::chrono::steady_clock;
        const auto started = profile ? Clock::now() : Clock::time_point{};
        const auto epoch_us = profile ? std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::system_clock::now().time_since_epoch())
                                            .count()
                                      : 0;
        double route_ms = 0, materialize_ms = 0, query_stage_ms = 0, scan_ms = 0;
        // 1) centroid routing on host: per query pick nprobe nearest lists
        std::vector<int64_t> cand_ids;
        std::unique_ptr<int64_t[]> unfiltered_ids;
        std::vector<int64_t> cand_offsets(static_cast<size_t>(nq) + 1, 0);
        // Full unfiltered probing is exact brute-force. Reuse the resident
        // base and implicit IDs instead of building nq copies of all row IDs.
        const auto* device_ops = dev_ready_ ? fl::GetDeviceOps() : nullptr;
        const bool full_scan =
            bitset.empty() && nprobe == nlist_ && device_ops != nullptr && device_ops->brute_force_topk != nullptr;
        if (!full_scan) {
            BuildCandidateLists(xq, nq, nprobe, bitset, cand_ids, cand_offsets, profile ? &route_ms : nullptr,
                                profile ? &materialize_ms : nullptr, unfiltered_ids);
        }

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
            } else if (ops->ivf_flat_scan != nullptr || full_scan) {
                const auto stage_started = profile ? Clock::now() : Clock::time_point{};
                const size_t q_bytes = static_cast<size_t>(nq) * dim_ * sizeof(float);
                void* d_q = ops->malloc(q_bytes);
                const bool staged = d_q != nullptr && ops->memcpy_h2d(d_q, xq, q_bytes, nullptr) == 0;
                if (profile)
                    query_stage_ms = std::chrono::duration<double, std::milli>(Clock::now() - stage_started).count();
                if (staged) {
                    const auto scan_started = profile ? Clock::now() : Clock::time_point{};
                    if (full_scan) {
                        done = ops->brute_force_topk(dev_xb_.get(), nb_, d_q, nq, dim_, k, metric_code_, distances, ids,
                                                     nullptr) == 0;
                    } else {
                        const auto* candidate_data = unfiltered_ids ? unfiltered_ids.get() : cand_ids.data();
                        done = ops->ivf_flat_scan(dev_xb_.get(), nb_, dim_, d_q, nq, candidate_data,
                                                  cand_offsets.data(), k, metric_code_, distances, ids, nullptr) == 0;
                    }
                    if (profile)
                        scan_ms = std::chrono::duration<double, std::milli>(Clock::now() - scan_started).count();
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
            // Missing specializations or device failures preserve the original
            // exact CPU fallback, including padding and filter semantics.
            if (full_scan) {
                BuildCandidateLists(xq, nq, nprobe, bitset, cand_ids, cand_offsets, profile ? &route_ms : nullptr,
                                    profile ? &materialize_ms : nullptr, unfiltered_ids);
            }
            CpuScanCandidates(xq, nq, k, unfiltered_ids ? unfiltered_ids.get() : cand_ids.data(), cand_offsets,
                              distances, ids);
        }
        if (profile) {
            int64_t minimum = nb_, maximum = 0;
            for (int64_t q = 0; q < nq; ++q) {
                const int64_t count = cand_offsets[q + 1] - cand_offsets[q];
                minimum = std::min(minimum, count);
                maximum = std::max(maximum, count);
            }
            LOG_KNOWHERE_INFO_ << "FL_IVF_SCAN nb=" << nb_ << " nq=" << nq << " nlist=" << nlist_
                               << " nprobe=" << nprobe << " k=" << k << " candidates=" << cand_offsets[nq]
                               << " candidate_min=" << minimum << " candidate_max=" << maximum
                               << " device=" << device_id_ << " device_done=" << done << " full_scan=" << full_scan
                               << " node=" << reinterpret_cast<uintptr_t>(this)
                               << " base_ptr=" << reinterpret_cast<uintptr_t>(dev_xb_.get()) << " start_us=" << epoch_us
                               << " route_ms=" << route_ms << " materialize_ms=" << materialize_ms
                               << " query_stage_ms=" << query_stage_ms << " scan_ms=" << scan_ms
                               << " host_capacity_bytes="
                               << (unfiltered_ids ? static_cast<size_t>(cand_offsets[nq]) : cand_ids.capacity()) *
                                      sizeof(int64_t)
                               << " elapsed_ms="
                               << std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        }
        // bitset already applied up-front in BuildCandidateLists (both device
        // and CPU scans receive pre-filtered candidates); no post-filter here.
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
        // MapOutToIn passes caller ids through unchanged when the node has no
        // id map, so a caller-supplied id may exceed nb_ (no payload). Keep
        // the result shape aligned with the request (upstream IVF returns one
        // row per requested id and relies on faiss reconstruct's assert for
        // the bounds); emit a zero vector for out-of-range ids instead of
        // reading out of bounds.
        const int64_t rows = static_cast<int64_t>(in_ids.size());
        if (rows == 0) {
            return GenResultDataSet(0, dim_, nullptr);
        }
        auto* data = new (std::nothrow) float[static_cast<size_t>(rows) * dim_];
        if (data == nullptr) {
            return expected<DataSetPtr>::Err(Status::malloc_error, "GetVectorByIds alloc failed");
        }
        for (int64_t i = 0; i < rows; ++i) {
            const int64_t id = in_ids[i];
            if (id < 0 || id >= nb_) {
                std::memset(data + i * dim_, 0, static_cast<size_t>(dim_) * sizeof(float));
            } else {
                std::memcpy(data + i * dim_, host_xb_.data() + id * dim_, dim_ * sizeof(float));
            }
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
        if (!fl::ReadChecked(reader, &dim_, sizeof(dim_), 1) || !fl::ReadChecked(reader, &nb_, sizeof(nb_), 1) ||
            !fl::ReadChecked(reader, &nlist_, sizeof(nlist_), 1) ||
            !fl::ReadChecked(reader, &metric_code_, sizeof(metric_code_), 1) || !fl::FlScalarDimsSane(dim_, nb_) ||
            nlist_ <= 0 || nlist_ > nb_ || (metric_code_ != 0 && metric_code_ != 1)) {
            LOG_KNOWHERE_WARNING_ << "FL IVF_FLAT deserialize: corrupt header";
            return Status::invalid_binary_set;
        }
        centroids_.resize(static_cast<size_t>(nlist_) * dim_);
        host_xb_.resize(static_cast<size_t>(nb_) * dim_);
        if (!fl::ReadChecked(reader, centroids_.data(), sizeof(float), centroids_.size()) ||
            !fl::ReadChecked(reader, host_xb_.data(), sizeof(float), host_xb_.size())) {
            LOG_KNOWHERE_WARNING_ << "FL IVF_FLAT deserialize: truncated payload";
            return Status::invalid_binary_set;
        }
        invlists_.assign(nlist_, {});
        assign_.resize(nb_);
        for (int64_t l = 0; l < nlist_; ++l) {
            int64_t sz = 0;
            if (!fl::ReadChecked(reader, &sz, sizeof(sz), 1) || sz < 0 || sz > nb_) {
                LOG_KNOWHERE_WARNING_ << "FL IVF_FLAT deserialize: corrupt inverted list header";
                return Status::invalid_binary_set;
            }
            invlists_[l].resize(static_cast<size_t>(sz));
            if (sz > 0) {
                if (!fl::ReadChecked(reader, invlists_[l].data(), sizeof(int64_t), static_cast<size_t>(sz))) {
                    LOG_KNOWHERE_WARNING_ << "FL IVF_FLAT deserialize: truncated inverted list";
                    return Status::invalid_binary_set;
                }
                for (auto id : invlists_[l]) {
                    if (id < 0 || id >= nb_) {
                        LOG_KNOWHERE_WARNING_ << "FL IVF_FLAT deserialize: member id out of range";
                        return Status::invalid_binary_set;
                    }
                    assign_[id] = l;
                }
            }
        }
        RefreshCentroidNorms();
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
                fl::FlMemoryPool::Instance().TryReserve(x_bytes + c_bytes)) {
                d_x = ops->malloc(x_bytes);
                d_c = ops->malloc(c_bytes);
                if (d_x != nullptr && d_c != nullptr && ops->memcpy_h2d(d_x, x, x_bytes, nullptr) == 0) {
                    staged_bytes = x_bytes + c_bytes;  // reservation held; released by TrackFree below
                    LOG_KNOWHERE_INFO_ << "FL IVF_FLAT kmeans: device assignment enabled";
                } else {
                    if (d_x != nullptr) {
                        ops->free(d_x);
                    }
                    if (d_c != nullptr) {
                        ops->free(d_c);
                    }
                    fl::FlMemoryPool::Instance().TrackFree(x_bytes + c_bytes);  // release reservation
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
                RefreshCentroidNorms();
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
        RefreshCentroidNorms();  // final centroids -> search-time norm cache
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
                fl::FlMemoryPool::Instance().TryReserve(x_bytes + c_bytes)) {
                void* d_x = ops->malloc(x_bytes);
                void* d_c = ops->malloc(c_bytes);
                const bool ok = d_x != nullptr && d_c != nullptr && ops->memcpy_h2d(d_x, x, x_bytes, nullptr) == 0 &&
                                ops->memcpy_h2d(d_c, centroids_.data(), c_bytes, nullptr) == 0 &&
                                ops->kmeans_assign(d_x, nb, dim_, d_c, nlist_, metric_code_, out, nullptr) == 0;
                if (d_x != nullptr) {
                    ops->free(d_x);
                }
                if (d_c != nullptr) {
                    ops->free(d_c);
                }
                fl::FlMemoryPool::Instance().TrackFree(x_bytes + c_bytes);  // release reservation
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
        // L2 routing uses ||c||^2 - 2*v.c: the query norm is a constant
        // offset across centroids and never changes the argmin. Centroid
        // norms come from the cent_norms_ cache (refreshed whenever the
        // centroids change) instead of being recomputed per call.
        int64_t best = 0;
        float best_score = metric_code_ == 1 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (int64_t l = 0; l < nlist_; ++l) {
            const float* c = centroids_.data() + l * dim_;
            float dot = 0.f;
            for (int64_t d = 0; d < dim_; ++d) {
                dot += v[d] * c[d];
            }
            const float score = metric_code_ == 1 ? dot : (cent_norms_[static_cast<size_t>(l)] - 2.f * dot);
            if ((metric_code_ == 1 && score > best_score) || (metric_code_ == 0 && score < best_score)) {
                best_score = score;
                best = l;
            }
        }
        return best;
    }

    // Recompute the cached centroid norms; must be called after any centroid
    // mutation (each kmeans iteration, Deserialize).
    void
    RefreshCentroidNorms() {
        if (metric_code_ != 0) {  // IP routing needs no norms
            cent_norms_.clear();
            return;
        }
        cent_norms_.resize(static_cast<size_t>(nlist_), 0.f);
        for (int64_t l = 0; l < nlist_; ++l) {
            const float* c = centroids_.data() + l * dim_;
            float s = 0.f;
            for (int64_t d = 0; d < dim_; ++d) {
                s += c[d] * c[d];
            }
            cent_norms_[static_cast<size_t>(l)] = s;
        }
    }

    void
    BuildCandidateLists(const float* xq, int64_t nq, int64_t nprobe, const BitsetView& bitset,
                        std::vector<int64_t>& cand_ids, std::vector<int64_t>& cand_offsets, double* route_ms,
                        double* materialize_ms, std::unique_ptr<int64_t[]>& unfiltered_ids) const {
        using Clock = std::chrono::steady_clock;
        const auto started = route_ms ? Clock::now() : Clock::time_point{};
        std::vector<int64_t> probes(static_cast<size_t>(nq) * nprobe);
        // Bound the routing team by both available queries and useful work.
        // Reuse one score buffer per worker instead of allocating per query.
        const double routing_work = static_cast<double>(nlist_) * dim_ * nq;
        const int route_threads = static_cast<int>(
            std::max<double>(1, std::min<double>({static_cast<double>(nq), static_cast<double>(omp_get_max_threads()),
                                                  32, routing_work / 262144})));
#pragma omp parallel num_threads(route_threads) if (route_threads > 1)
        {
            std::vector<std::pair<float, int64_t>> scores(nlist_);
#pragma omp for schedule(static)
            for (int64_t i = 0; i < nq; ++i) {
                const float* q = xq + i * dim_;
                // Route score: IP uses the raw dot; L2 uses ||c||^2 - 2*q.c from
                // the cached centroid norms (the query norm is a constant offset
                // across centroids and does not affect probe selection).
                for (int64_t l = 0; l < nlist_; ++l) {
                    const float* c = centroids_.data() + l * dim_;
                    float dot = 0.f;
                    for (int64_t d = 0; d < dim_; ++d) {
                        dot += q[d] * c[d];
                    }
                    scores[l] = {metric_code_ == 1 ? dot : (cent_norms_[static_cast<size_t>(l)] - 2.f * dot), l};
                }
                const bool is_ip = metric_code_ == 1;
                std::partial_sort(
                    scores.begin(), scores.begin() + nprobe, scores.end(), [is_ip](const auto& a, const auto& b) {
                        // A total order keeps probe prefixes nested even when centroids tie.
                        return a.first == b.first ? a.second < b.second : is_ip ? a.first > b.first : a.first < b.first;
                    });
                auto* pl = probes.data() + i * nprobe;
                for (int64_t p = 0; p < nprobe; ++p) {
                    pl[p] = scores[p].second;
                }
            }
        }
        const auto routed = route_ms ? Clock::now() : Clock::time_point{};
        if (route_ms)
            *route_ms = std::chrono::duration<double, std::milli>(routed - started).count();
        if (bitset.empty()) {
            // Count before allocation, then fill disjoint query ranges. Do not
            // evaluate filters twice: evaluator-backed bitsets stay below.
            cand_offsets[0] = 0;
            for (int64_t i = 0; i < nq; ++i) {
                int64_t count = 0;
                for (int64_t p = 0; p < nprobe; ++p)
                    count += static_cast<int64_t>(invlists_[probes[i * nprobe + p]].size());
                if (count > std::numeric_limits<int64_t>::max() - cand_offsets[i]) {
                    throw std::overflow_error("FL IVF candidate size overflow");
                }
                cand_offsets[i + 1] = cand_offsets[i] + count;
            }
            // Every element is overwritten below before either scanner reads it.
            // Default-initialized scalar storage avoids an extra serial zero-fill.
            unfiltered_ids.reset(new int64_t[static_cast<size_t>(cand_offsets[nq])]);
            auto* output = unfiltered_ids.get();
            // CSR copying is bandwidth-bound. A full machine-wide OpenMP team
            // adds scheduling cost and contention without more useful bandwidth.
            const int copy_threads = static_cast<int>(std::min<int64_t>({nq, omp_get_max_threads(), 8}));
#pragma omp parallel for schedule(static) num_threads(copy_threads) if (cand_offsets[nq] >= 1048576 && copy_threads > 1)
            for (int64_t i = 0; i < nq; ++i) {
                auto position = cand_offsets[i];
                for (int64_t p = 0; p < nprobe; ++p) {
                    const auto& list = invlists_[probes[i * nprobe + p]];
                    if (!list.empty())
                        std::copy(list.begin(), list.end(), output + position);
                    position += static_cast<int64_t>(list.size());
                }
            }
            if (materialize_ms)
                *materialize_ms = std::chrono::duration<double, std::milli>(Clock::now() - routed).count();
            return;
        }
        for (int64_t i = 0; i < nq; ++i) {
            cand_offsets[i] = static_cast<int64_t>(cand_ids.size());
            for (int64_t p = 0; p < nprobe; ++p) {
                const auto& list = invlists_[probes[i * nprobe + p]];
                for (const int64_t id : list) {
                    if (!bitset.test(id))
                        cand_ids.push_back(id);
                }
            }
        }
        cand_offsets[nq] = static_cast<int64_t>(cand_ids.size());
        if (materialize_ms)
            *materialize_ms = std::chrono::duration<double, std::milli>(Clock::now() - routed).count();
    }

    void
    CpuScanCandidates(const float* xq, int64_t nq, int64_t k, const int64_t* cand_ids,
                      const std::vector<int64_t>& cand_offsets, float* distances, int64_t* ids) const {
        const bool is_ip = metric_code_ == 1;
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < nq; ++i) {
            // bounded max-heap of (dist, id); for L2 keep smallest, for IP keep
            // largest (same selection pattern as the host adapter). O(k) memory
            // instead of collecting every candidate then partial_sort.
            using Cand = std::pair<float, int64_t>;
            auto cmp = [is_ip](const Cand& a, const Cand& b) {
                // Match device selection: lower global ID wins equal scores.
                return a.first == b.first ? a.second < b.second : is_ip ? a.first > b.first : a.first < b.first;
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
            for (int64_t c = cand_offsets[i]; c < cand_offsets[i + 1]; ++c) {
                const int64_t id = cand_ids[c];
                const float* b = host_xb_.data() + id * dim_;
                float dot = 0.f, nbsum = 0.f;
                for (int64_t d = 0; d < dim_; ++d) {
                    dot += q[d] * b[d];
                    if (!is_ip) {
                        nbsum += b[d] * b[d];
                    }
                }
                const float dist = is_ip ? dot : (na + nbsum - 2.f * dot);
                if (static_cast<int64_t>(heap.size()) < k) {
                    heap.emplace(dist, id);
                } else {
                    if (cmp(Cand{dist, id}, heap.top())) {
                        heap.pop();
                        heap.emplace(dist, id);
                    }
                }
            }
            const int64_t count = static_cast<int64_t>(heap.size());
            // Write missing neighbors after the valid sorted prefix, not before it.
            for (int64_t t = count; t < k; ++t) {
                distances[i * k + t] = is_ip ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
                ids[i * k + t] = -1;
            }
            for (int64_t t = count - 1; t >= 0; --t) {
                distances[i * k + t] = heap.top().first;
                ids[i * k + t] = heap.top().second;
                heap.pop();
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
        if (!fl::FlMemoryPool::Instance().TryReserve(bytes)) {
            LOG_KNOWHERE_WARNING_ << "FL memory pool rejects staging " << bytes << " bytes; IVF_FLAT on CPU fallback";
            return;
        }
        if (ops->set_device == nullptr || ops->malloc == nullptr || ops->memcpy_h2d == nullptr) {
            fl::FlMemoryPool::Instance().TrackFree(bytes);
            return;
        }
        device_id_ = fl::FlDeviceRouter::Instance().Acquire();
        if (ops->set_device(device_id_) != 0) {
            fl::FlMemoryPool::Instance().TrackFree(bytes);
            return;
        }
        void* d = ops->malloc(bytes);
        if (d == nullptr) {
            fl::FlMemoryPool::Instance().TrackFree(bytes);
            return;
        }
        if (ops->memcpy_h2d(d, host_xb_.data(), bytes, nullptr) != 0) {
            ops->free(d);
            fl::FlMemoryPool::Instance().TrackFree(bytes);
            return;
        }
        dev_xb_ = fl::FlDeviceBuffer(d, bytes);
        dev_ready_ = true;
        LOG_KNOWHERE_INFO_ << "FL IVF_FLAT staged " << nb_ << " x " << dim_ << " vectors on device";
    }

    int64_t dim_{0};
    int64_t nb_{0};
    int64_t nlist_{0};
    int metric_code_{0};             // 0 = L2, 1 = IP
    bool use_device_assign_{true};   // device centroid routing (train/add)
    std::vector<float> centroids_;   // nlist * dim
    std::vector<float> cent_norms_;  // cached ||c_l||^2 (L2 routing; empty for IP)
    std::vector<float> host_xb_;     // nb * dim
    std::vector<int64_t> assign_;    // nb -> list id
    std::vector<std::vector<int64_t>> invlists_;
    fl::FlDeviceBuffer dev_xb_;
    bool dev_ready_{false};
    int device_id_{0};  // bound at staging time via FlDeviceRouter (multi-device)
};

}  // namespace knowhere

#endif  // KNOWHERE_FL_IVF_FLAT_NODE_H
