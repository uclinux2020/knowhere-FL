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

#ifndef KNOWHERE_KNOWHERE_FL_FORMAT_H
#define KNOWHERE_KNOWHERE_FL_FORMAT_H

#include <cstdint>
#include <iomanip>
#include <sstream>

#include "io/memory_io.h"
#include "knowhere/log.h"

namespace knowhere {
namespace fl {

// ---------------------------------------------------------------------------
// FL index serialization format header.
//
// Every FL index binary starts with {magic, format_version}. Deserialize
// verifies both before interpreting the payload, so a future format change
// (e.g. residual PQ, fp16 storage, multi-shard layout) fails loudly on old
// readers instead of silently producing garbage. Bump kFlFormatVersion on
// any payload layout change; keep a reader branch for older versions.
// ---------------------------------------------------------------------------

constexpr uint32_t kFlIndexMagic = 0x464C4944;  // "FLID" (little-endian)
constexpr int32_t kFlFormatVersion = 1;

inline void
WriteFormatHeader(MemoryIOWriter& writer) {
    uint32_t magic = kFlIndexMagic;
    int32_t version = kFlFormatVersion;
    writer(&magic, sizeof(magic), 1);
    writer(&version, sizeof(version), 1);
}

// Returns false (and logs) when the stream does not carry a valid FL header.
inline bool
ReadFormatHeader(MemoryIOReader& reader, const char* index_kind) {
    uint32_t magic = 0;
    int32_t version = 0;
    reader(&magic, sizeof(magic), 1);
    reader(&version, sizeof(version), 1);
    if (magic != kFlIndexMagic) {
        LOG_KNOWHERE_WARNING_ << "FL " << index_kind << " deserialize: bad magic 0x" << std::hex << magic
                              << " (not an FL index binary?)";
        return false;
    }
    if (version > kFlFormatVersion) {
        LOG_KNOWHERE_WARNING_ << "FL " << index_kind << " deserialize: format version " << version
                              << " newer than supported " << kFlFormatVersion;
        return false;
    }
    return true;
}

// Bounds-checked read: MemoryIOReader silently short-reads on truncated
// payloads (the unread tail stays uninitialized and every later field lands
// on misaligned bytes). FL deserializers must treat a short stream as
// corruption instead of consuming garbage.
inline bool
ReadChecked(MemoryIOReader& reader, void* dst, size_t elem_size, size_t count) {
    if (elem_size != 0 && count > reader.remaining() / elem_size) {
        return false;
    }
    return reader(dst, elem_size, count) == count;
}

// Sanity gate for deserialized scalar dims, applied BEFORE any resize():
// rejects absurd values from corrupt payloads that would otherwise trigger
// giant allocations (std::bad_alloc escaping Deserialize).
inline bool
FlScalarDimsSane(int64_t dim, int64_t nb) {
    // dim <= 4M and nb <= 2^40 are far beyond any real workload and only
    // exist to stop corrupt-payload arithmetic explosions.
    return dim > 0 && nb > 0 && dim <= (int64_t{1} << 22) && nb <= (int64_t{1} << 40);
}

}  // namespace fl
}  // namespace knowhere

#endif  // KNOWHERE_FL_FORMAT_H
