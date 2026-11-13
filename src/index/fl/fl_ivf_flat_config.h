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

#ifndef FL_IVF_FLAT_CONFIG_H
#define FL_IVF_FLAT_CONFIG_H

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "knowhere/config.h"

namespace knowhere {

// Config for the FL (FlagOS) IVF_FLAT index. Parameter names/ranges mirror
// the official GPU_IVF_FLAT semantics (design D2).
struct FlIvfFlatConfig : public BaseConfig {
    CFG_INT nlist;
    CFG_INT nprobe;
    CFG_BOOL use_device_assign;
    KNOWHERE_DECLARE_CONFIG(FlIvfFlatConfig) {
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
        KNOWHERE_CONFIG_DECLARE_FIELD(use_device_assign)
            .description("route vectors to centroids on the FL device during kmeans/add "
                         "(mm+argmin composition, no atomics); set false to force CPU routing")
            .set_default(true)
            .for_train();
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
        if (param_type == PARAM_TYPE::SEARCH && nprobe.value() > nlist.value()) {
            nprobe = nlist.value();
        }
        return Status::success;
    }
};

}  // namespace knowhere

#endif  // FL_IVF_FLAT_CONFIG_H
