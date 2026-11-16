// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
#include "common/fl/fl_device.h"

#include <iostream>

int main() {
    // The first query must register the native PPU adapter without external
    // Milvus initialization glue or a caller manually invoking registration.
    const auto* ops = knowhere::fl::GetDeviceOps();
    if (!ops || !knowhere::fl::DeviceAvailable() || knowhere::fl::DeviceCount() <= 0 ||
        !ops->brute_force_topk || !ops->ivf_flat_scan || !ops->kmeans_assign || !ops->ivf_pq_scan ||
        ops->graph_search != nullptr || ops->set_device(0) != 0) {
        std::cerr << "FAIL: PPU lazy registration or capability table\n";
        return 1;
    }
    const auto before = knowhere::fl::FlMemoryPool::Instance().Used();
    if (!knowhere::fl::FlMemoryPool::Instance().TryReserve(64)) return 1;
    void* ptr = ops->malloc(64);
    if (!ptr) return 1;
    {
        knowhere::fl::FlDeviceBuffer buffer(ptr, 64);
    }
    if (knowhere::fl::FlMemoryPool::Instance().Used() != before) return 1;
    ops->free(nullptr);
    // Explicit unregister is preserved for tests; lazy registration only runs
    // once, not on every access that observes an empty table.
    knowhere::fl::RegisterDeviceOps(nullptr);
    if (knowhere::fl::GetDeviceOps() != nullptr || knowhere::fl::DeviceAvailable()) return 1;
    std::cout << "PASS: native adapter lazy registration, capabilities, RAII/pool accounting, unregister\n";
    return 0;
}
