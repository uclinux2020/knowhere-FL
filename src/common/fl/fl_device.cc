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

#include "common/fl/fl_device.h"

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdlib>

#include "knowhere/log.h"

namespace knowhere {
namespace fl {

namespace {
// Published with release semantics so late-joining search threads never
// observe a half-written ops table (registration itself is still
// single-threaded by contract, per fl_device.h).
std::atomic<const FlDeviceOps*> g_ops{nullptr};

// Parses a non-negative MiB value from an env var; returns -1 when unset or
// invalid. Out-of-range values are clamped (they would otherwise wrap when
// scaled to bytes). Defined for ConfigureFromEnv only.
long long
ParseEnvMiB(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return -1;
    }
    errno = 0;
    char* end = nullptr;
    long long v = std::strtoll(raw, &end, 10);
    if (errno == ERANGE) {
        LOG_KNOWHERE_WARNING_ << "FL memory pool: clamping out-of-range " << name << "=\"" << raw << "\"";
        v = (v < 0) ? 0 : LLONG_MAX / (1024 * 1024);  // avoid overflow at MiB->bytes
    }
    if (end == raw || *end != '\0' || v < 0) {
        LOG_KNOWHERE_WARNING_ << "FL memory pool: ignoring invalid " << name << "=\"" << raw << "\"";
        return -1;
    }
    // clamp so that v * 1 MiB can never overflow size_t
    constexpr long long kMaxMiB = static_cast<long long>(static_cast<unsigned long long>(-1) >> 20);
    if (v > kMaxMiB) {
        LOG_KNOWHERE_WARNING_ << "FL memory pool: clamping oversized " << name << "=\"" << raw << "\"";
        v = kMaxMiB;
    }
    return v;
}
}  // namespace

const FlDeviceOps*
GetDeviceOps() {
#ifdef KNOWHERE_FL_WITH_PPU
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        if (g_ops.load(std::memory_order_acquire) == nullptr) {
            RegisterPpuAdapter();
        }
    });
#endif
    return g_ops.load(std::memory_order_acquire);
}

void
RegisterDeviceOps(const FlDeviceOps* ops) {
    const bool had_ops = (g_ops.load(std::memory_order_acquire) != nullptr);
    g_ops.store(ops, std::memory_order_release);
    // Re-reading the env on adapter registration makes sense; re-reading it
    // on UNregistration (ops == nullptr, a test-only path) would silently
    // mutate pool limits without any device present, so skip that case.
    if (ops != nullptr || !had_ops) {
        FlMemoryPool::Instance().ConfigureFromEnv();
    }
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
    const FlDeviceOps* ops = GetDeviceOps();
    if (ops == nullptr || ops->device_count == nullptr) {
        return false;
    }
    const int count = ops->device_count();
    if (count < 1) {
        // TryStageToDevice() calls this on every Add/Deserialize; log once.
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_KNOWHERE_INFO_ << "FL device not available (adapter registered but device_count=" << count << ")";
        }
        return false;
    }
    return true;
}

int
DeviceCount() {
    const FlDeviceOps* ops = GetDeviceOps();
    if (ops == nullptr || ops->device_count == nullptr) {
        return 0;
    }
    return ops->device_count();
}

void
FlDeviceBuffer::reset() {
    if (ptr_ != nullptr) {
        // Free through the function captured at construction so the buffer is
        // always released by the same adapter that allocated it, even if the
        // global ops table was re-registered (or cleared) in between.
        if (free_fn_ != nullptr) {
            free_fn_(ptr_);
        } else {
            // The adapter vanished without providing free(): releasing here
            // could crash, so we intentionally leak and log instead.
            LOG_KNOWHERE_ERROR_ << "FL device buffer leak: no free function captured (" << bytes_ << " bytes)";
        }
        // Only pool-tracked buffers are TrackFree'd; transient buffers were
        // never TrackAlloc'd and must not touch the counter.
        if (tracked_) {
            FlMemoryPool::Instance().TrackFree(bytes_);
        }
    }
    ptr_ = nullptr;
    bytes_ = 0;
    free_fn_ = nullptr;
    tracked_ = false;
}

}  // namespace fl
}  // namespace knowhere
