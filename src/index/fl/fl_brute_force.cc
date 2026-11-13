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

#include "index/fl/fl_index_node.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"

namespace knowhere {

// Registration for the FL (FlagOS domestic-GPU) brute-force index.
//
// Two names are registered:
//   - GPU_BRUTE_FORCE: user-facing name in FL builds (the CUDA path is
//     excluded by CMake when WITH_FL=ON, so no clash), keeping Milvus-side
//     index type semantics unchanged (design D2).
//   - FL_BRUTE_FORCE: explicit alias for debugging and A/B testing.
//
// Concurrent-search capacity scales with the visible FL device count, mirroring
// the CUDA path's cuda_concurrent_size_per_device policy.
namespace {
constexpr int64_t fl_concurrent_size_per_device = 2;
int64_t
FlThreadSize() {
    const int count = fl::DeviceCount();
    return count > 0 ? count * fl_concurrent_size_per_device : 1;
}
}  // namespace

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_BRUTE_FORCE, FlBruteForceIndexNode, fp32,
                                          knowhere::feature::GPU_KNN_FLOAT_INDEX, FlThreadSize());

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(FL_BRUTE_FORCE, FlBruteForceIndexNode, fp32,
                                          knowhere::feature::GPU_KNN_FLOAT_INDEX, FlThreadSize());

}  // namespace knowhere
