// Copyright (C) 2026 knowhere-FL contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and limitations under the License.

#include "index/fl/fl_cagra_node.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"

namespace knowhere {

// Registration for the FL (FlagOS) CAGRA-like graph index.
//
// FL_CAGRA is registered under its own name only. It deliberately does NOT
// claim "GPU_CAGRA": v1 builds the graph on CPU and traverses on CPU/device,
// which does not yet match official GPU_CAGRA semantics/performance claims.
// Per plan Phase 2.3, the full device path (device build + device traversal)
// is the milestone that would justify adding the GPU_ alias.
namespace {
constexpr int64_t fl_concurrent_size_per_device = 2;
int64_t
FlCagraThreadSize() {
    const int count = fl::DeviceCount();
    return count > 0 ? count * fl_concurrent_size_per_device : 1;
}
}  // namespace

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(FL_CAGRA, FlCagraIndexNode, fp32, knowhere::feature::GPU_ANN_FLOAT_INDEX,
                                          FlCagraThreadSize());

}  // namespace knowhere
