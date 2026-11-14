// Copyright (C) 2026 knowhere-FL contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

#ifndef KNOWHERE_FL_CAGRA_CONFIG_H
#define KNOWHERE_FL_CAGRA_CONFIG_H

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "knowhere/config.h"

namespace knowhere {

// Config for the FL (FlagOS) CAGRA-like graph index. Parameter names/ranges
// mirror the official GPU_CAGRA semantics where applicable (design D2).
// v1 builds a kNN graph (CPU brute force composition) and traverses it
// greedily; device traversal lands via FlDeviceOps::graph_search.
struct FlCagraConfig : public BaseConfig {
    CFG_INT graph_degree;   // build-time: neighbors per node (CAGRA graph_degree)
    CFG_INT itopk_size;     // search-time: candidate frontier cap (CAGRA itopk_size)
    KNOWHERE_DECLARE_CONFIG(FlCagraConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(graph_degree)
            .description("out-degree of each graph node")
            .set_default(32)
            .for_train()
            .set_range(1, 1024);
        KNOWHERE_CONFIG_DECLARE_FIELD(itopk_size)
            .description("candidate frontier size during greedy traversal; must be >= k")
            .set_default(64)
            .for_search()
            .set_range(1, 100000);
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        if (param_type == PARAM_TYPE::TRAIN) {
            constexpr std::array<std::string_view, 2> legal_metric_list{"L2", "IP"};
            std::string metric = metric_type.value();
            if (std::find(legal_metric_list.begin(), legal_metric_list.end(), metric) == legal_metric_list.end()) {
                std::string msg = "metric type " + metric +
                                  " not supported on FL path, supported: [L2 IP]; "
                                  "for COSINE, L2-normalize vectors and use IP (design D4)";
                return HandleError(err_msg, msg, Status::invalid_metric_type);
            }
        }
        if (param_type == PARAM_TYPE::SEARCH && itopk_size.value() < k.value()) {
            // official CAGRA also clamps itopk to >= k
            itopk_size = k.value();
        }
        return Status::success;
    }
};

}  // namespace knowhere

#endif  // KNOWHERE_FL_CAGRA_CONFIG_H
