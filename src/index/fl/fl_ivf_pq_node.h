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

#ifndef KNOWHERE_FL_IVF_PQ_NODE_H
#define KNOWHERE_FL_IVF_PQ_NODE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "knowhere/index/index_static.h"
#include "common/fl/fl_device.h"
#include "common/fl/fl_device_router.h"
#include "index/fl/fl_format.h"
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
// FlIvfPqIndexNode (GPU_IVF_PQ / FL_IVF_PQ on the FL path).
// One of Knowhere-FL's FL_* index nodes, built on the backend-neutral device
// abstraction layer that accelerates vector search on diverse AI chips.
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
//     CPU ADC fallback. bitset is pre-filtered in BuildCandidateLists (both
//     device and CPU scans receive pre-filtered candidates).
//   - Metric: L2 only. IP is rejected outright (Train and Deserialize):
//     a plain-dot PQ LUT cannot decompose IP distance because the
//     base-vector norm ||x|| is lost to quantization, so ADC accumulations
//     mis-rank. COSINE stays excluded per design D4. Callers needing IP
//     must use FL IVF_FLAT or BRUTE_FORCE instead.
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
        } else {
            // IP rejected: plain-dot ADC cannot decompose IP distance (base
            // norms are lost to quantization); see the class header note.
            LOG_KNOWHERE_ERROR_ << "FL IVF_PQ supports L2 only, got metric: " << metric;
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
        RefreshCentroidNorms();
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
        // Norm caches make the L2 nearest-selection a pure dot-product scan:
        // argmin ||v-c||^2 == argmin ||c||^2 - 2*v.c (query norm is a constant
        // offset). Coarse norms come from the Train-time cache; codebook
        // norms are computed once per Add instead of nb*m times inside
        // NearestIn.
        // Codebook sub-vector norms are cached as a member: recomputed on Add /
        // Deserialize and reused by the per-query LUT build in CpuAdcScan (the
        // node is L2-only, so cb_norms_ is always populated).
        cb_norms_ = ComputeNorms(codebooks_.data(), m_ * ksub_, dsub_);
        const float* cent_norms = cent_norms_.data();  // empty for IP (unused)
#pragma omp parallel for schedule(dynamic)
        for (int64_t j = 0; j < nb_; ++j) {
            const float* v = x + j * dim_;
            assign[j] = NearestIn(v, centroids_.data(), cent_norms, nlist_, dim_);
            // per-subspace encoding
            for (int64_t s = 0; s < m_; ++s) {
                codes_[static_cast<size_t>(j) * m_ + s] = static_cast<uint8_t>(NearestIn(
                    v + s * dsub_, codebooks_.data() + s * ksub_ * dsub_, cb_norms_.data() + s * ksub_, ksub_, dsub_));
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
            if (ops->set_device != nullptr && ops->set_device(device_id_) != 0) {
                LOG_KNOWHERE_WARNING_ << "FL set_device failed in IVF_PQ search";
            } else if (ops->ivf_pq_scan != nullptr) {
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
    GetVectorByIds(const DataSetPtr, milvus::OpContext* = nullptr) const override {
        return expected<DataSetPtr>::Err(
            Status::not_implemented,
            "FL IVF_PQ stores quantized codes only (no raw vectors); use GPU_IVF_FLAT/GPU_BRUTE_FORCE or a "
            "refine-capable index when raw retrieval is required");
    }

    bool
    HasRawData(const std::string& /*metric_type*/) const override {
        return false;  // PQ keeps codes only
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
            fl::WriteFormatHeader(writer);
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
        if (!fl::ReadFormatHeader(reader, "IVF_PQ")) {
            return Status::invalid_binary_set;
        }
        if (!fl::ReadChecked(reader, &dim_, sizeof(dim_), 1) || !fl::ReadChecked(reader, &nb_, sizeof(nb_), 1) ||
            !fl::ReadChecked(reader, &nlist_, sizeof(nlist_), 1) || !fl::ReadChecked(reader, &m_, sizeof(m_), 1) ||
            !fl::ReadChecked(reader, &ksub_, sizeof(ksub_), 1) || !fl::ReadChecked(reader, &dsub_, sizeof(dsub_), 1) ||
            !fl::ReadChecked(reader, &metric_code_, sizeof(metric_code_), 1) || !fl::FlScalarDimsSane(dim_, nb_) ||
            nlist_ <= 0 || nlist_ > nb_ || m_ <= 0 || ksub_ != 256 || dsub_ <= 0 || m_ * dsub_ != dim_) {
            LOG_KNOWHERE_WARNING_ << "FL IVF_PQ deserialize: corrupt header";
            return Status::invalid_binary_set;
        }
        if (metric_code_ != 0) {
            // Refuse to load legacy IP-encoded FL IVF_PQ payloads: their ADC
            // results mis-rank by design (see class header note).
            LOG_KNOWHERE_ERROR_ << "FL IVF_PQ deserialize: payload metric is IP; L2-only since v1";
            return Status::invalid_binary_set;
        }
        centroids_.resize(static_cast<size_t>(nlist_) * dim_);
        codebooks_.resize(static_cast<size_t>(m_) * ksub_ * dsub_);
        codes_.resize(static_cast<size_t>(nb_) * m_);
        if (!fl::ReadChecked(reader, centroids_.data(), sizeof(float), centroids_.size()) ||
            !fl::ReadChecked(reader, codebooks_.data(), sizeof(float), codebooks_.size()) ||
            !fl::ReadChecked(reader, codes_.data(), sizeof(uint8_t), codes_.size())) {
            LOG_KNOWHERE_WARNING_ << "FL IVF_PQ deserialize: truncated payload";
            return Status::invalid_binary_set;
        }
        invlists_.assign(nlist_, {});
        for (int64_t l = 0; l < nlist_; ++l) {
            int64_t sz = 0;
            if (!fl::ReadChecked(reader, &sz, sizeof(sz), 1) || sz < 0 || sz > nb_) {
                LOG_KNOWHERE_WARNING_ << "FL IVF_PQ deserialize: corrupt inverted list header";
                return Status::invalid_binary_set;
            }
            invlists_[l].resize(static_cast<size_t>(sz));
            if (sz > 0) {
                if (!fl::ReadChecked(reader, invlists_[l].data(), sizeof(int64_t), static_cast<size_t>(sz))) {
                    LOG_KNOWHERE_WARNING_ << "FL IVF_PQ deserialize: truncated inverted list";
                    return Status::invalid_binary_set;
                }
                for (auto id : invlists_[l]) {
                    if (id < 0 || id >= nb_) {
                        LOG_KNOWHERE_WARNING_ << "FL IVF_PQ deserialize: member id out of range";
                        return Status::invalid_binary_set;
                    }
                }
            }
        }
        RefreshCentroidNorms();
        cb_norms_ = ComputeNorms(codebooks_.data(), m_ * ksub_, dsub_);
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

    static bool
    StaticHasRawData(const knowhere::BaseConfig& /*config*/, const IndexVersion& /*version*/) {
        return false;  // PQ keeps quantized codes only
    }

    // Device-resident cost ~= the nb*m uint8 code payload; codebooks and
    // coarse centroids stay on host and are small (m*ksub*dsub*4).
    static expected<Resource>
    StaticEstimateLoadResource(const uint64_t file_size_in_bytes, const int64_t num_rows, const int64_t dim,
                               const knowhere::BaseConfig& config, const IndexVersion& /*version*/) {
        Resource res;
        int64_t m = 8;  // FlIvfPqConfig default m (unreachable fallback: the
                        // framework always passes a FlIvfPqConfig here)
        if (const auto* pq_cfg = dynamic_cast<const FlIvfPqConfig*>(&config)) {
            m = pq_cfg->m.value();
        }
        res.memoryCost = static_cast<uint64_t>(num_rows) * static_cast<uint64_t>(std::max<int64_t>(m, 1));
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
    // Squared norms of `n` row vectors of width `width` (L2 only; callers
    // skip this entirely for IP).
    static std::vector<float>
    ComputeNorms(const float* table, int64_t n, int64_t width) {
        std::vector<float> norms(static_cast<size_t>(n), 0.f);
        for (int64_t l = 0; l < n; ++l) {
            const float* c = table + l * width;
            float s = 0.f;
            for (int64_t d = 0; d < width; ++d) {
                s += c[d] * c[d];
            }
            norms[static_cast<size_t>(l)] = s;
        }
        return norms;
    }

    // Recompute the cached coarse-centroid norms; called after Train and
    // Deserialize (centroids never change afterwards in v1).
    void
    RefreshCentroidNorms() {
        if (metric_code_ != 0) {  // IP needs no norms
            cent_norms_.clear();
            return;
        }
        cent_norms_ = ComputeNorms(centroids_.data(), nlist_, dim_);
    }

    // Nearest row in `table` under the node metric. For L2 the score is
    // ||c||^2 - 2*v.c using the precomputed `table_norms` (the query norm is
    // a constant offset and is eliminated); `table_norms` may be nullptr only
    // when the metric is IP.
    int64_t
    NearestIn(const float* v, const float* table, const float* table_norms, int64_t n, int64_t width) const {
        int64_t best = 0;
        float best_score = metric_code_ == 1 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
        for (int64_t l = 0; l < n; ++l) {
            const float* c = table + l * width;
            float dot = 0.f;
            for (int64_t d = 0; d < width; ++d) {
                dot += v[d] * c[d];
            }
            const float score =
                metric_code_ == 1 ? dot : (table_norms[static_cast<size_t>(l)] - 2.f * dot);
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
            // Route score: IP uses the raw dot; L2 uses ||c||^2 - 2*q.c from
            // the cached centroid norms (query norm is a constant offset).
            for (int64_t l = 0; l < nlist_; ++l) {
                const float* c = centroids_.data() + l * dim_;
                float dot = 0.f;
                for (int64_t d = 0; d < dim_; ++d) {
                    dot += q[d] * c[d];
                }
                scores[l] = {metric_code_ == 1 ? dot : (cent_norms_[static_cast<size_t>(l)] - 2.f * dot), l};
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
                // query sub-vector norm is loop-invariant per subspace
                float na = 0.f;
                if (!is_ip) {
                    for (int64_t d = 0; d < dsub_; ++d) {
                        na += q_sub[d] * q_sub[d];
                    }
                }
                for (int64_t ks = 0; ks < ksub_; ++ks) {
                    const float* c = cb_s + ks * dsub_;
                    float dot = 0.f;
                    for (int64_t d = 0; d < dsub_; ++d) {
                        dot += q_sub[d] * c[d];
                    }
                    // codebook sub-vector norm comes from the cached cb_norms_
                    // (recomputed only on Add/Deserialize); query sub-vector
                    // norm (na) is loop-invariant per subspace.
                    lut[static_cast<size_t>(s) * ksub_ + ks] =
                        is_ip ? dot : (na + cb_norms_[static_cast<size_t>(s) * ksub_ + ks] - 2.f * dot);
                }
            }
            // bounded max-heap over the ADC distances (O(k) memory; same
            // selection pattern as the host adapter's host_ivf_pq_scan).
            using Cand = std::pair<float, int64_t>;
            auto cmp = [is_ip](const Cand& a, const Cand& b) {
                return is_ip ? (a.first > b.first) : (a.first < b.first);
            };
            std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> heap(cmp);
            for (int64_t c = cand_offsets[i]; c < cand_offsets[i + 1]; ++c) {
                const int64_t id = cand_ids[c];
                const uint8_t* vc = codes_.data() + static_cast<size_t>(id) * m_;
                float dist = 0.f;
                for (int64_t s = 0; s < m_; ++s) {
                    dist += lut[static_cast<size_t>(s) * ksub_ + vc[s]];
                }
                if (static_cast<int64_t>(heap.size()) < k) {
                    heap.emplace(dist, id);
                } else {
                    const float worst = heap.top().first;
                    if ((is_ip && dist > worst) || (!is_ip && dist < worst)) {
                        heap.pop();
                        heap.emplace(dist, id);
                    }
                }
            }
            for (int64_t t = k - 1; t >= 0; --t) {
                if (heap.empty()) {
                    distances[i * k + t] =
                        is_ip ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
                    ids[i * k + t] = -1;
                } else {
                    distances[i * k + t] = heap.top().first;
                    ids[i * k + t] = heap.top().second;
                    heap.pop();
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
        if (!fl::FlMemoryPool::Instance().TryReserve(cb_bytes + code_bytes)) {
            LOG_KNOWHERE_WARNING_ << "FL memory pool rejects staging " << (cb_bytes + code_bytes)
                                  << " bytes; IVF_PQ on CPU fallback";
            return;
        }
        if (ops->set_device == nullptr || ops->malloc == nullptr || ops->memcpy_h2d == nullptr) {
            fl::FlMemoryPool::Instance().TrackFree(cb_bytes + code_bytes);
            return;
        }
        device_id_ = fl::FlDeviceRouter::Instance().Acquire();
        if (ops->set_device(device_id_) != 0) {
            fl::FlMemoryPool::Instance().TrackFree(cb_bytes + code_bytes);
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
            fl::FlMemoryPool::Instance().TrackFree(cb_bytes + code_bytes);
            return;
        }
        dev_codebooks_ = fl::FlDeviceBuffer(d_cb, cb_bytes);
        dev_codes_ = fl::FlDeviceBuffer(d_codes, code_bytes);
        dev_ready_ = true;
        LOG_KNOWHERE_INFO_ << "FL IVF_PQ staged codebooks+codes on device (" << (cb_bytes + code_bytes) << " bytes)";
    }

    int64_t dim_{0};
    int64_t nb_{0};
    int64_t nlist_{0};
    int64_t m_{0};           // subquantizers
    int64_t ksub_{0};        // centroids per subquantizer (2^nbits, v1: 256)
    int64_t dsub_{0};        // dim / m
    int metric_code_{0};     // 0 = L2, 1 = IP
    std::vector<float> centroids_;    // nlist * dim (coarse)
    std::vector<float> cent_norms_;   // cached ||c_l||^2 of coarse centroids (L2 only)
    std::vector<float> codebooks_;    // m * ksub * dsub
    std::vector<float> cb_norms_;     // cached ||c||^2 of codebook sub-vectors (m * ksub, L2 only)
    std::vector<uint8_t> codes_;      // nb * m
    std::vector<std::vector<int64_t>> invlists_;
    fl::FlDeviceBuffer dev_codebooks_;
    fl::FlDeviceBuffer dev_codes_;
    bool dev_ready_{false};
    int device_id_{0};  // bound at staging time via FlDeviceRouter (multi-device)
};

}  // namespace knowhere

#endif  // KNOWHERE_FL_IVF_PQ_NODE_H
