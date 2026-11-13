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

#ifndef FL_BRUTE_FORCE_CONFIG_H
#define FL_BRUTE_FORCE_CONFIG_H

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "knowhere/config.h"

namespace knowhere {

// Config for the FL (FlagOS) brute-force index.
// Design decision D4: COSINE is not a kernel metric on the FL path (same
// constraint as the official GPU indexes). Callers must L2-normalize vectors
// and use IP instead.
struct FlBruteForceConfig : public BaseConfig {
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
        return Status::success;
    }
};

}  // namespace knowhere

#endif  // FL_BRUTE_FORCE_CONFIG_H
