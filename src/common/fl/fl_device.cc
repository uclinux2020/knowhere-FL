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

#include <cstdlib>

#include "knowhere/log.h"

namespace knowhere {
namespace fl {

namespace {
const FlDeviceOps* g_ops = nullptr;

// Parses a non-negative MiB value from an env var; returns -1 when unset or
// invalid. Defined for ConfigureFromEnv only.
long long
ParseEnvMiB(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return -1;
    }
    char* end = nullptr;
    const long long v = std::strtoll(raw, &end, 10);
    if (end == raw || *end != '\0' || v < 0) {
        LOG_KNOWHERE_WARNING_ << "FL memory pool: ignoring invalid " << name << "=\"" << raw << "\"";
        return -1;
    }
    return v;
}
}  // namespace

const FlDeviceOps*
GetDeviceOps() {
    return g_ops;
}

void
RegisterDeviceOps(const FlDeviceOps* ops) {
    g_ops = ops;
    FlMemoryPool::Instance().ConfigureFromEnv();
    LOG_KNOWHERE_INFO_ << "FL device ops registered, device_count="
                       << ((ops != nullptr && ops->device_count != nullptr) ? ops->device_count() : 0);
}

void
FlMemoryPool::ConfigureFromEnv() {
    const long long init_mb = ParseEnvMiB("KNOWHERE_FL_POOL_INIT_MB");
    const long long max_mb = ParseEnvMiB("KNOWHERE_FL_POOL_MAX_MB");
    if (init_mb < 0 && max_mb < 0) {
        return;  // defaults preserved
    }
    size_t init_bytes = 0;
    size_t max_bytes = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        init_bytes = init_bytes_;
        max_bytes = max_bytes_;
    }
    constexpr size_t kMiB = 1024ull * 1024ull;
    if (init_mb >= 0) {
        init_bytes = static_cast<size_t>(init_mb) * kMiB;
    }
    if (max_mb >= 0) {
        max_bytes = static_cast<size_t>(max_mb) * kMiB;
    }
    if (max_bytes != 0 && max_bytes < init_bytes) {
        LOG_KNOWHERE_WARNING_ << "FL memory pool: max < init, clamping max to init";
        max_bytes = init_bytes;
    }
    Configure(init_bytes, max_bytes);
    LOG_KNOWHERE_INFO_ << "FL memory pool configured from env: init=" << init_bytes << "B max=" << max_bytes << "B";
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
