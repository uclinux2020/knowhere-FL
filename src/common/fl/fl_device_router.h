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

#ifndef KNOWHERE_FL_DEVICE_ROUTER_H
#define KNOWHERE_FL_DEVICE_ROUTER_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/fl/fl_device.h"
#include "knowhere/log.h"

namespace knowhere {
namespace fl {

// Round-robin physical adapter IDs. An explicit allowlist is frozen on first
// preflight/Acquire, including errors; changing the environment cannot repair
// a running process. DeviceCount and FlDeviceOps retain their physical-ID ABI.
class FlDeviceRouter {
 public:
    static FlDeviceRouter&
    Instance() {
        static FlDeviceRouter router;
        return router;
    }

    int
    DeviceCount() const {
        return fl::DeviceCount();
    }

    // Call during index initialization, before CPU/device fallback decisions.
    // Adapter absence or unusable IDs are fatal only for explicit configuration.
    void
    Preflight() {
        const auto* ops = fl::GetDeviceOps();
        std::call_once(initialized_, [this, ops] {
            const char* raw = std::getenv("KNOWHERE_FL_DEVICE_IDS");
            configured_ = raw != nullptr;
            if (!configured_) {
                return;
            }
            try {
                ids_ = ParseDeviceIds(raw, ops && ops->device_count ? ops->device_count() : 0);
                if (ops == nullptr || ops->set_device == nullptr) {
                    throw std::invalid_argument("adapter has no set_device");
                }
                for (int id : ids_) {
                    if (ops->set_device(id) != 0) {
                        throw std::invalid_argument("physical device " + std::to_string(id) + " is unavailable");
                    }
                }
                LOG_KNOWHERE_INFO_ << "FL KNOWHERE_FL_DEVICE_IDS frozen (physical adapter/HG IDs): " << raw;
            } catch (const std::exception& e) {
                error_ = std::string("KNOWHERE_FL_DEVICE_IDS: ") + e.what();
                LOG_KNOWHERE_ERROR_ << error_;
            }
        });
        if (!error_.empty()) {
            throw std::invalid_argument(error_);
        }
    }

    // Returns the physical ID itself, never its position in the allowlist.
    // Unset configuration preserves the previous all-device/CPU policy.
    int
    Acquire() {
        Preflight();
        if (configured_) {
            return ids_[next_.fetch_add(1, std::memory_order_relaxed) % ids_.size()];
        }
        const int n = DeviceCount();
        if (n <= 1) {
            return 0;
        }
        return static_cast<int>(next_.fetch_add(1, std::memory_order_relaxed) % static_cast<uint64_t>(n));
    }

    // Strict decimal CSV: whitespace, signs, empty entries and duplicates fail.
    // Pure parsing also permits device-free regression coverage.
    static std::vector<int>
    ParseDeviceIds(const std::string& raw, int physical_count) {
        if (raw.empty()) {
            throw std::invalid_argument("allowlist is empty");
        }
        std::vector<int> ids;
        size_t start = 0;
        while (start < raw.size()) {
            const size_t end = raw.find(',', start);
            const size_t stop = end == std::string::npos ? raw.size() : end;
            if (stop == start) {
                throw std::invalid_argument("empty device ID");
            }
            int id = 0;
            for (size_t pos = start; pos < stop; ++pos) {
                const char digit = raw[pos];
                if (digit < '0' || digit > '9' || id > (std::numeric_limits<int>::max() - (digit - '0')) / 10) {
                    throw std::invalid_argument("device IDs must be non-negative decimal integers");
                }
                id = id * 10 + (digit - '0');
            }
            if (id >= physical_count) {
                throw std::invalid_argument("physical device " + std::to_string(id) + " out of range (count=" +
                                            std::to_string(physical_count) + ")");
            }
            if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
                throw std::invalid_argument("duplicate physical device " + std::to_string(id));
            }
            ids.push_back(id);
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
            if (start == raw.size()) {
                throw std::invalid_argument("empty device ID");
            }
        }
        return ids;
    }

    // Reset the dispatch counter only; never re-read the frozen environment.
    void
    Reset() {
        next_.store(0, std::memory_order_relaxed);
    }

 private:
    friend struct FlDeviceRouterTestAccess;
    FlDeviceRouter() = default;
    std::once_flag initialized_;
    std::vector<int> ids_;
    std::string error_;
    bool configured_{false};
    std::atomic<uint64_t> next_{0};
};

}  // namespace fl
}  // namespace knowhere

#endif  // KNOWHERE_FL_DEVICE_ROUTER_H
