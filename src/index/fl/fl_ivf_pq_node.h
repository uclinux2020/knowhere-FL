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

#ifndef FL_IVF_PQ_NODE_H
#define FL_IVF_PQ_NODE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "common/fl/fl_device.h"
#include "index/fl/fl_ivf_pq_config.h"
#include "index/fl/fl_kmeans_cpu.h"
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
// FlIvfPqIndexNode (GPU_IVF_PQ / FL_IVF_PQ on the FL path)
//
// v1 design (plan Phase 2.1):
//   - Train: CPU Lloyd kmeans for coarse centroids + per-subspace PQ
//     codebooks (fl::LloydKmeansCpu). Non-residual PQ (no centroid residual
//     encoding) in v1 — documented simplification vs faiss IndexIVFPQ.
//   - Add: coarse assignment + per-subspace encoding to nb x m uint8 codes,
//     inverted lists, stage codebooks+codes to the FL device (OOM pre-check
//     D5, CPU fallback).
//   - Search: host centroid routing -> candidate lists -> device
//     ops->ivf_pq_scan (per-query LUT via mm composition + ADC scan) with
//     CPU ADC fallback. bitset pre-filter + post-filter double insurance.
// ---------------------------------------------------------------------------
template <typename DataType>
class FlIvfPqIndexNode : public IndexNode {
 public:
    FlIvfPqIndexNode(const int32_t version, const Object& object) : IndexNode(version) {
        static_assert(std::is_same_v<DataType, fp32>, "FlIvfPqIndexNode currently supports fp32 only");
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool /*use_knowhere_build_pool*/) override {
        const FlIvfPqConfig& f_cfg = static_cast<const FlIvfPqConfig&>(*cfg);
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
        m_ = f_cfg.m.value();
        dsub_ = dim_ / m_;
        ksub_ = 1 << f_cfg.nbits.value();  // config guarantees nbits == 8

        const auto* x = static_cast<const float*>(dataset->GetTensor());
        const int64_t nb = dataset->GetRows();
        if (x == nullptr || nb < std::max<int64_t>(nlist_, ksub_)) {
            LOG_KNOWHERE_ERROR_ << "FL IVF_PQ train: need at least max(nlist, 2^nbits) training vectors";
            return Status::invalid_args;
        }

        // 1) coarse centroids
        centroids_.resize(static_cast<size_t>(nlist_) * dim_);
        fl::LloydKmeansCpu(x, nb, dim_, nlist_, metric_code_, centroids_.data());

        // 2) per-subspace PQ codebooks (each subspace kmeans on nb x dsub)
        codebooks_.resize(static_cast<size_t>(m_) * ksub_ * dsub_);
        std::vector<float> sub(static_cast<size_t>(nb) * dsub_);
        for (int64_t s = 0; s < m_; ++s) {
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < nb; ++j) {
                std::memcpy(sub.data() + j * dsub_, x + j * dim_ + s * dsub_, dsub_ * sizeof(float));
            }
            fl::LloydKmeansCpu(sub.data(), nb, dsub_, ksub_, metric_code_, codebooks_.data() + s * ksub_ * dsub_,
                               nullptr, 25, 42 + s);
        }
        LOG_KNOWHERE_INFO_ << "FL IVF_PQ trained: nlist=" << nlist_ << " m=" << m_ << " ksub=" << ksub_
                           << " dsub=" << dsub_;
        return Status::success;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> /*cfg*/, bool /*use_knowhere_build_pool*/) override {
        if (centroids_.empty() || codebooks_.empty()) {
            return Status::index_not_trained;
        }
        const auto* x = static_cast<const float*>(dataset->GetTensor());
        const int64_t rows = dataset->GetRows();
        if (x == nullptr || rows <= 0) {
            return Status::empty_index;
        }
        nb_ = rows;

        invlists_.assign(nlist_, {});
        codes_.resize(static_cast<size_t>(nb_) * m_);
        std::vector<int64_t> assign(nb_);
#pragma omp parallel for schedule(dynamic)
        for (int64_t j = 0; j < nb_; ++j) {
            const float* v = x + j * dim_;
            assign[j] = NearestIn(v, centroids_.data(), nlist_, dim_);
            // per-subspace encoding
            for (int64_t s = 0; s < m_; ++s) {
                codes_[static_cast<size_t>(j) * m_ + s] = static_cast<uint8_t>(
                    NearestIn(v + s * dsub_, codebooks_.data() + s * ksub_ * dsub_, ksub_, dsub_));
            }
        }
        for (int64_t j = 0; j < nb_; ++j) {
            invlists_[assign[j]].push_back(j);
        }

        TryStageToDevice();
        return Status::success;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* /*op_context*/ = nullptr) const override {
        const FlIvfPqConfig& f_cfg = static_cast<const FlIvfPqConfig&>(*cfg);
        const int64_t nq = dataset->GetRows();
        const int64_t k = f_cfg.k.value();
        const auto* xq = static_cast<const float*>(dataset->GetTensor());
        if (nb_ == 0 || xq == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "FL IVF_PQ index is empty");
        }
        const int64_t nprobe = std::min<int64_t>(f_cfg.nprobe.value(), nlist_);

        // 1) host centroid routing -> candidate lists
        std::vector<int64_t> cand_ids;
        std::vector<int64_t> cand_offsets(static_cast<size_t>(nq) + 1, 0);
        BuildCandidateLists(xq, nq, nprobe, bitset, cand_ids, cand_offsets);

        // 2) ADC scan: device path first, CPU fallback (design D5)
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
            if (ops->ivf_pq_scan != nullptr) {
                const size_t q_bytes = static_cast<size_t>(nq) * dim_ * sizeof(float);
                void* d_q = ops->malloc(q_bytes);
                if (d_q != nullptr && ops->memcpy_h2d(d_q, xq, q_bytes, nullptr) == 0) {
                    done = ops->ivf_pq_scan(dev_codebooks_.get(), m_, ksub_, dsub_, dev_codes_.get(), nb_, d_q, nq,
                                            dim_, cand_ids.data(), cand_offsets.data(), k, metric_code_, distances,
                                            ids, nullptr) == 0;
                }
                if (d_q != nullptr) {
                    ops->free(d_q);
                }
            }
            if (!done) {
                LOG_KNOWHERE_WARNING_ << "FL device ivf_pq_scan unavailable/failed, CPU ADC fallback";
            }
        }
        if (!done) {
            CpuAdcScan(xq, nq, k, cand_ids, cand_offsets, distances, ids);
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
    GetIndexMeta(std::unique_ptr<Config>) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (nb_ == 0 || codebooks_.empty()) {
            return Status::empty_index;
        }
        try {
            MemoryIOWriter writer;
            writer(&dim_, sizeof(dim_), 1);
            writer(&nb_, sizeof(nb_), 1);
            writer(&nlist_, sizeof(nlist_), 1);
            writer(&m_, sizeof(m_), 1);
            writer(&ksub_, sizeof(ksub_), 1);
            writer(&dsub_, sizeof(dsub_), 1);
            writer(&metric_code_, sizeof(metric_code_), 1);
            writer(centroids_.data(), sizeof(float), centroids_.size());
            writer(codebooks_.data(), sizeof(float), codebooks_.size());
            writer(codes_.data(), sizeof(uint8_t), codes_.size());
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
            LOG_KNOWHERE_WARNING_ << "FL IVF_PQ serialize error: " << e.what();
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
        reader(&dim_, sizeof(dim_), 1);
        reader(&nb_, sizeof(nb_), 1);
        reader(&nlist_, sizeof(nlist_), 1);
        reader(&m_, sizeof(m_), 1);
        reader(&ksub_, sizeof(ksub_), 1);
        reader(&dsub_, sizeof(dsub_), 1);
        reader(&metric_code_, sizeof(metric_code_), 1);
        centroids_.resize(static_cast<size_t>(nlist_) * dim_);
        reader(centroids_.data(), sizeof(float), centroids_.size());
        codebooks_.resize(static_cast<size_t>(m_) * ksub_ * dsub_);
        reader(codebooks_.data(), sizeof(float), codebooks_.size());
        codes_.resize(static_cast<size_t>(nb_) * m_);
        reader(codes_.data(), sizeof(uint8_t), codes_.size());
        invlists_.assign(nlist_, {});
        for (int64_t l = 0; l < nlist_; ++l) {
            int64_t sz = 0;
            reader(&sz, sizeof(sz), 1);
            invlists_[l].resize(sz);
            if (sz > 0) {
                reader(invlists_[l].data(), sizeof(int64_t), sz);
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
        return std::make_unique<FlIvfPqConfig>();
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
        return nb_ * m_ * sizeof(uint8_t) + m_ * ksub_ * dsub_ * sizeof(float) + nlist_ * dim_ * sizeof(float);
    }

    int64_t
    Count() const override {
        return nb_;
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FL_IVFPQ;
    }

 private:
    // nearest entry in `table` (n rows of width `width`) for subvector v
    int64_t
    NearestIn(const float* v, const float* table, int64_t n, int64_t width) const {
        int64_t best = 0;
        float best_score = metric_code_ == 1 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (int64_t l = 0; l < n; ++l) {
            const float* c = table + l * width;
            float dot = 0.f, na = 0.f, nc = 0.f;
            for (int64_t d = 0; d < width; ++d) {
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

    // CPU ADC fallback: per-query LUT + code lookup scan (same semantics as
    // the host adapter's host_ivf_pq_scan reference).
    void
    CpuAdcScan(const float* xq, int64_t nq, int64_t k, const std::vector<int64_t>& cand_ids,
               const std::vector<int64_t>& cand_offsets, float* distances, int64_t* ids) const {
        const bool is_ip = metric_code_ == 1;
#pragma omp parallel for schedule(dynamic)
        for (int64_t i = 0; i < nq; ++i) {
            const float* q = xq + i * dim_;
            std::vector<float> lut(static_cast<size_t>(m_) * ksub_);
            for (int64_t s = 0; s < m_; ++s) {
                const float* q_sub = q + s * dsub_;
                const float* cb_s = codebooks_.data() + s * ksub_ * dsub_;
                for (int64_t ks = 0; ks < ksub_; ++ks) {
                    const float* c = cb_s + ks * dsub_;
                    float dot = 0.f, na = 0.f, nc = 0.f;
                    for (int64_t d = 0; d < dsub_; ++d) {
                        dot += q_sub[d] * c[d];
                        if (!is_ip) {
                            na += q_sub[d] * q_sub[d];
                            nc += c[d] * c[d];
                        }
                    }
                    lut[static_cast<size_t>(s) * ksub_ + ks] = is_ip ? dot : (na + nc - 2.f * dot);
                }
            }
            using Cand = std::pair<float, int64_t>;
            std::vector<Cand> best;
            best.reserve(static_cast<size_t>(k) + 1);
            for (int64_t c = cand_offsets[i]; c < cand_offsets[i + 1]; ++c) {
                const int64_t id = cand_ids[c];
                const uint8_t* vc = codes_.data() + static_cast<size_t>(id) * m_;
                float dist = 0.f;
                for (int64_t s = 0; s < m_; ++s) {
                    dist += lut[static_cast<size_t>(s) * ksub_ + vc[s]];
                }
                best.emplace_back(dist, id);
            }
            const int64_t nk = std::min<int64_t>(k, static_cast<int64_t>(best.size()));
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
        const size_t cb_bytes = codebooks_.size() * sizeof(float);
        const size_t code_bytes = codes_.size() * sizeof(uint8_t);
        if (!fl::FlMemoryPool::Instance().Ensure(cb_bytes + code_bytes)) {
            LOG_KNOWHERE_WARNING_ << "FL memory pool rejects staging " << (cb_bytes + code_bytes)
                                  << " bytes; IVF_PQ on CPU fallback";
            return;
        }
        if (ops->set_device == nullptr || ops->malloc == nullptr || ops->memcpy_h2d == nullptr || ops->set_device(0) != 0) {
            return;
        }
        void* d_cb = ops->malloc(cb_bytes);
        void* d_codes = ops->malloc(code_bytes);
        bool ok = d_cb != nullptr && d_codes != nullptr &&
                  ops->memcpy_h2d(d_cb, codebooks_.data(), cb_bytes, nullptr) == 0 &&
                  ops->memcpy_h2d(d_codes, codes_.data(), code_bytes, nullptr) == 0;
        if (!ok) {
            if (d_cb != nullptr) {
                ops->free(d_cb);
            }
            if (d_codes != nullptr) {
                ops->free(d_codes);
            }
            return;
        }
        dev_codebooks_ = fl::FlDeviceBuffer(d_cb, cb_bytes);
        dev_codes_ = fl::FlDeviceBuffer(d_codes, code_bytes);
        fl::FlMemoryPool::Instance().TrackAlloc(cb_bytes + code_bytes);
        dev_ready_ = true;
        LOG_KNOWHERE_INFO_ << "FL IVF_PQ staged codebooks+codes on device (" << (cb_bytes + code_bytes) << " bytes)";
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
    int64_t m_{0};           // subquantizers
    int64_t ksub_{0};        // centroids per subquantizer (2^nbits, v1: 256)
    int64_t dsub_{0};        // dim / m
    int metric_code_{0};     // 0 = L2, 1 = IP
    std::vector<float> centroids_;    // nlist * dim (coarse)
    std::vector<float> codebooks_;    // m * ksub * dsub
    std::vector<uint8_t> codes_;      // nb * m
    std::vector<std::vector<int64_t>> invlists_;
    fl::FlDeviceBuffer dev_codebooks_;
    fl::FlDeviceBuffer dev_codes_;
    bool dev_ready_{false};
};

}  // namespace knowhere

#endif  // FL_IVF_PQ_NODE_H
