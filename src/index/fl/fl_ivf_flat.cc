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

#include "index/fl/fl_ivf_flat_node.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"

namespace knowhere {

// Registration for the FL (FlagOS domestic-GPU) IVF_FLAT index.
// GPU_IVF_FLAT is the user-facing name in FL builds (CUDA path excluded at
// build time); FL_IVF_FLAT is the explicit alias for debugging/A-B testing.
namespace {
constexpr int64_t fl_concurrent_size_per_device = 2;
int64_t
FlIvfThreadSize() {
    const int count = fl::DeviceCount();
    return count > 0 ? count * fl_concurrent_size_per_device : 1;
}
}  // namespace

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_IVF_FLAT, FlIvfFlatIndexNode, fp32,
                                          knowhere::feature::GPU_ANN_FLOAT_INDEX, FlIvfThreadSize());

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(FL_IVF_FLAT, FlIvfFlatIndexNode, fp32,
                                          knowhere::feature::GPU_ANN_FLOAT_INDEX, FlIvfThreadSize());

}  // namespace knowhere
