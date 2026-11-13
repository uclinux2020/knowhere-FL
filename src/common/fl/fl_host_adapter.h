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

#ifndef KNOWHERE_FL_HOST_ADAPTER_H
#define KNOWHERE_FL_HOST_ADAPTER_H

namespace knowhere {
namespace fl {

// Registers a host-memory implementation of FlDeviceOps. "Device" buffers are
// plain host allocations and kernels run on CPU (OpenMP). Purposes:
//   1. Unit tests exercise the full FL device path without a GPU;
//   2. CPU-only CI validates adapter/node contract end to end;
//   3. Reference for chip adapter authors (PPU/Ascend/...) - each function
//      here is the semantic spec its device counterpart must match.
// Returns false if another adapter is already registered.
bool
RegisterHostAdapter();

// Unregisters the host adapter (mainly for test isolation).
void
UnregisterHostAdapter();

}  // namespace fl
}  // namespace knowhere

#endif  // KNOWHERE_FL_HOST_ADAPTER_H
