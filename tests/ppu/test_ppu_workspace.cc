// Copyright (C) 2026 knowhere-FL contributors.
// Licensed under the Apache License, Version 2.0.
// Include the implementation to inspect private lease/accounting invariants.
#include <barrier>
#include <set>
#include <thread>

#include "common/fl/adapter/ppu/fl_ppu_runtime.cc"

namespace ppu = knowhere::fl::ppu;
namespace {
void
Require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
}  // namespace

int
main() {
    try {
        setenv("KNOWHERE_FL_PPU_WORKSPACE_CACHE_MB", "1", 1);
        setenv("KNOWHERE_FL_PPU_PROFILE", "1", 1);
        Require(ppu::SetDevice(0) == 0, "device binding failed");
        auto& pool = ppu::ArenaPool();
        constexpr size_t bytes = 256 * 1024;
        {
            ppu::Workspace failed(bytes, nullptr);
            Require(failed.pool != nullptr, "pool reservation missing");
            // Simulate exceptional unwinding before launch-chain completion.
        }
        Require(pool.bytes == 0 && pool.idle.empty(), "failed lease was retained");
        {
            ppu::Workspace first(bytes, nullptr);
            const auto ptr = first.buffer->ptr;
            first.completed = true;
            first.Release();
            ppu::Workspace reused(bytes / 2, nullptr);
            Require(reused.cache_hit && reused.buffer->ptr == ptr, "best-fit lease was not reused");
            Require(reused.bytes == bytes, "capacity accounting shrank on reuse");
            reused.completed = true;
        }
        // Five simultaneous leases exceed the four-entry global budget. Each
        // worker binds its own thread-local context and owns a distinct arena.
        std::barrier ready(5);
        std::vector<HGdeviceptr> pointers(5);
        std::vector<std::exception_ptr> failures(5);
        std::vector<std::thread> workers;
        for (int i = 0; i < 5; ++i) {
            workers.emplace_back([&, i] {
                std::unique_ptr<ppu::Workspace> lease;
                try {
                    Require(ppu::SetDevice(0) == 0, "worker context failed");
                    lease = std::make_unique<ppu::Workspace>(bytes, nullptr);
                    pointers[i] = lease->buffer->ptr;
                } catch (...) {
                    failures[i] = std::current_exception();
                }
                ready.arrive_and_wait();
                if (lease)
                    lease->completed = true;
            });
        }
        for (auto& worker : workers) worker.join();
        for (const auto& failure : failures)
            if (failure)
                std::rethrow_exception(failure);
        Require(std::set<HGdeviceptr>(pointers.begin(), pointers.end()).size() == 5, "in-flight leases alias");
        Require(pool.bytes == (1ULL << 20) && pool.idle.size() == 4, "pool exceeded budget");
        if (ppu::DeviceCount() > 1) {
            Require(ppu::SetDevice(1) == 0, "second context failed");
            ppu::Workspace other(bytes, nullptr);
            Require(!other.cache_hit && other.buffer->ctx != pool.idle.front().buffer->ctx, "lease crossed contexts");
            other.completed = true;
            Require(ppu::SetDevice(0) == 0, "restore context failed");
        }
        // Free the accounting fixture in its owning context before exercising
        // real kernel chains with smaller shapes and changing queries/streams.
        ppu::TrimWorkspaceCache();
        Require(ppu::GetWorkspaceCacheStats().retained_bytes == 0, "trim left idle allocations");
        {
            ppu::Workspace active(bytes, nullptr);
            ppu::Workspace small(bytes, nullptr);
            small.completed = true;
            small.Release();
            ppu::TrimWorkspaceCache();
            auto stats = ppu::GetWorkspaceCacheStats();
            Require(stats.idle_bytes == 0 && stats.retained_bytes == bytes, "trim touched active lease");
            active.completed = true;
        }
        // Larger shapes must reclaim idle capacity instead of permanently
        // falling back after smaller requests fill the cache budget.
        {
            const auto before = ppu::GetWorkspaceCacheStats().evictions;
            ppu::Workspace larger(1ULL << 20, nullptr);
            Require(larger.pool != nullptr && !larger.cache_hit, "larger shape could not replace idle entries");
            Require(ppu::GetWorkspaceCacheStats().evictions > before, "idle replacement was not recorded");
            larger.completed = true;
        }
        ppu::TrimWorkspaceCache();
        constexpr int64_t nb = 128, nq = 4, dim = 5, k = 3;
        std::vector<float> xb(nb * dim, 0.f), xq(nq * dim, 1.f), scores(nq * k);
        std::vector<int64_t> candidates(nq * nb), offsets(nq + 1), ids(nq * k);
        for (int64_t q = 0; q < nq; ++q) {
            offsets[q] = q * nb;
            for (int64_t j = 0; j < nb; ++j) candidates[q * nb + j] = nb - 1 - j;
        }
        offsets[nq] = nq * nb;
        void* base = ppu::Alloc(xb.size() * sizeof(float));
        void* query = ppu::Alloc(xq.size() * sizeof(float));
        Require(base && query, "kernel inputs allocation failed");
        Require(ppu::CopyH2D(base, xb.data(), xb.size() * sizeof(float), nullptr) == 0, "base upload failed");
        for (int repeat = 0; repeat < 6; ++repeat) {
            std::fill(xq.begin(), xq.end(), static_cast<float>(repeat + 1));
            void* stream = ppu::StreamCreate();
            Require(stream != nullptr, "stream creation failed");
            Require(ppu::CopyH2D(query, xq.data(), xq.size() * sizeof(float), stream) == 0, "query upload failed");
            Require(ppu::Search(base, nb, query, nq, dim, k, repeat % 2, candidates.data(), offsets.data(),
                                scores.data(), ids.data(), stream) == 0,
                    "cached kernel search failed");
            for (int64_t q = 0; q < nq; ++q)
                for (int64_t rank = 0; rank < k; ++rank) {
                    Require(ids[q * k + rank] == rank, "cached workspace returned stale IDs");
                    const float expected = repeat % 2 ? 0.f : dim * (repeat + 1.f) * (repeat + 1.f);
                    Require(std::abs(scores[q * k + rank] - expected) < 1e-4f,
                            "cached workspace returned stale scores");
                }
            ppu::StreamDestroy(stream);
        }
        workers.clear();
        failures.assign(3, nullptr);
        for (int worker = 0; worker < 3; ++worker) {
            workers.emplace_back([&, worker] {
                try {
                    Require(ppu::SetDevice(0) == 0, "kernel worker context failed");
                    void* private_query = ppu::Alloc(xq.size() * sizeof(float));
                    void* stream = ppu::StreamCreate();
                    Require(private_query && stream, "kernel worker resources failed");
                    std::vector<float> queries(xq.size(), worker + 1.f), distances(nq * k);
                    std::vector<int64_t> results(nq * k);
                    Require(ppu::CopyH2D(private_query, queries.data(), queries.size() * sizeof(float), stream) == 0,
                            "kernel worker upload failed");
                    for (int repeat = 0; repeat < 5; ++repeat) {
                        Require(ppu::Search(base, nb, private_query, nq, dim, k, 0, candidates.data(), offsets.data(),
                                            distances.data(), results.data(), stream) == 0,
                                "concurrent cached kernel failed");
                        for (int64_t q = 0; q < nq; ++q)
                            for (int64_t rank = 0; rank < k; ++rank) {
                                Require(results[q * k + rank] == rank, "concurrent cached ID corrupted");
                                Require(
                                    std::abs(distances[q * k + rank] - dim * (worker + 1.f) * (worker + 1.f)) < 1e-4f,
                                    "concurrent cached score corrupted");
                            }
                    }
                    ppu::StreamDestroy(stream);
                    ppu::Free(private_query);
                } catch (...) {
                    failures[worker] = std::current_exception();
                }
            });
        }
        for (auto& worker : workers) worker.join();
        for (const auto& failure : failures)
            if (failure)
                std::rethrow_exception(failure);
        ppu::Free(query);
        ppu::Free(base);
        Require(pool.idle.size() >= 1 && pool.bytes <= pool.limit, "concurrent cache accounting failed");
        ppu::TrimWorkspaceCache();
        Require(ppu::GetWorkspaceCacheStats().retained_bytes == 0, "final trim did not release cache");
        std::cout << "PASS: bounded workspace leases, failure discard, context isolation, streams, changing queries\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
