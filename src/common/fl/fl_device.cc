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

#include "common/fl/fl_device.h"

#include "knowhere/log.h"

namespace knowhere {
namespace fl {

namespace {
const FlDeviceOps* g_ops = nullptr;
}  // namespace

const FlDeviceOps*
GetDeviceOps() {
    return g_ops;
}

void
RegisterDeviceOps(const FlDeviceOps* ops) {
    g_ops = ops;
    LOG_KNOWHERE_INFO_ << "FL device ops registered, device_count="
                       << ((ops != nullptr && ops->device_count != nullptr) ? ops->device_count() : 0);
}

bool
DeviceAvailable() {
    if (g_ops == nullptr || g_ops->device_count == nullptr) {
        return false;
    }
    const int count = g_ops->device_count();
    if (count < 1) {
        LOG_KNOWHERE_INFO_ << "FL device not available (adapter registered but device_count=" << count << ")";
        return false;
    }
    return true;
}

int
DeviceCount() {
    if (g_ops == nullptr || g_ops->device_count == nullptr) {
        return 0;
    }
    return g_ops->device_count();
}

void
FlDeviceBuffer::reset() {
    if (ptr_ != nullptr && g_ops != nullptr && g_ops->free != nullptr) {
        g_ops->free(ptr_);
        FlMemoryPool::Instance().TrackFree(bytes_);
    }
    ptr_ = nullptr;
    bytes_ = 0;
}

}  // namespace fl
}  // namespace knowhere
