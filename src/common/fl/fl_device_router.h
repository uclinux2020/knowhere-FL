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

#ifndef FL_DEVICE_ROUTER_H
#define FL_DEVICE_ROUTER_H

#include <atomic>
#include <cstdint>

#include "common/fl/fl_device.h"

namespace knowhere {
namespace fl {

// ---------------------------------------------------------------------------
// Multi-device dispatch for FL indexes (Phase 2 skeleton).
//
// v1 policy: round-robin device selection. Every FL index node must route
// set_device() through Acquire() instead of hardcoding device 0, so that
// multiple indexes (or multiple Milvus segments) spread across cards.
//
// Later phases (not implemented here):
//   - shard-per-device replication with per-shard FlDeviceBuffer sets
//   - FlagCX-based cross-device scatter/gather for brute-force merge
//   - load-aware dispatch using per-device inflight counters
// ---------------------------------------------------------------------------
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

    // Returns a device id in [0, DeviceCount()), round-robin. Returns 0 when
    // no device (or a single device) is available.
    int
    Acquire() {
        const int n = DeviceCount();
        if (n <= 1) {
            return 0;
        }
        return static_cast<int>(next_.fetch_add(1, std::memory_order_relaxed) % static_cast<uint64_t>(n));
    }

    // Test hook: reset round-robin state.
    void
    Reset() {
        next_.store(0, std::memory_order_relaxed);
    }

 private:
    FlDeviceRouter() = default;
    std::atomic<uint64_t> next_{0};
};

}  // namespace fl
}  // namespace knowhere

#endif  // FL_DEVICE_ROUTER_H
