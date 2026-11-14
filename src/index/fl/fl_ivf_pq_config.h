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

#ifndef KNOWHERE_FL_IVF_PQ_CONFIG_H
#define KNOWHERE_FL_IVF_PQ_CONFIG_H

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "knowhere/config.h"

namespace knowhere {

// Config for the FL (FlagOS) IVF_PQ index. Parameter names/ranges mirror the
// official GPU_IVF_PQ semantics (design D2). v1 constraints:
//   - metric: L2 only (plain-dot PQ ADC cannot decompose IP distance; see
//     the node header note)
//   - nbits must be 8 (one uint8 code per subquantizer)
//   - dim must be divisible by m
// bitset semantics: exact pre-filter inside the probed lists (same as
// IVF_FLAT).
struct FlIvfPqConfig : public BaseConfig {
    CFG_INT nlist;
    CFG_INT nprobe;
    CFG_INT m;
    CFG_INT nbits;
    KNOWHERE_DECLARE_CONFIG(FlIvfPqConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(nlist)
            .description("number of inverted lists")
            .set_default(128)
            .for_train()
            .set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(nprobe)
            .description("number of probes at query time")
            .set_default(8)
            .for_search()
            .set_range(1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(m).description("number of subquantizers").set_default(8).for_train().set_range(
            1, 65536);
        KNOWHERE_CONFIG_DECLARE_FIELD(nbits)
            .description("bits per subquantizer code (FL v1 supports 8 only)")
            .set_default(8)
            .for_train()
            .set_range(1, 24);
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        if (param_type == PARAM_TYPE::TRAIN) {
            constexpr std::array<std::string_view, 1> legal_metric_list{"L2"};
            std::string metric = metric_type.value();
            if (std::find(legal_metric_list.begin(), legal_metric_list.end(), metric) == legal_metric_list.end()) {
                std::string msg = "metric type " + metric +
                                  " not supported on FL IVF_PQ (supported: [L2]); plain-dot PQ LUT "
                                  "decompositions mis-rank under IP (base-vector norms are lost to "
                                  "quantization), and COSINE is excluded per design D4 - normalize "
                                  "vectors and use IVF_FLAT/BRUTE_FORCE with IP instead";
                return HandleError(err_msg, msg, Status::invalid_metric_type);
            }
            if (nbits.value() != 8) {
                std::string msg = "FL IVF_PQ v1 supports nbits=8 only (one uint8 code per subquantizer)";
                return HandleError(err_msg, msg, Status::invalid_args);
            }
            if (m.value() <= 0 || dim.value() % m.value() != 0) {
                std::string msg = "dim must be divisible by m (dim=" + std::to_string(dim.value()) +
                                  ", m=" + std::to_string(m.value()) + ")";
                return HandleError(err_msg, msg, Status::invalid_args);
            }
        }
        if (param_type == PARAM_TYPE::SEARCH && nprobe.value() > nlist.value()) {
            nprobe = nlist.value();
        }
        return Status::success;
    }
};

}  // namespace knowhere

#endif  // KNOWHERE_FL_IVF_PQ_CONFIG_H
