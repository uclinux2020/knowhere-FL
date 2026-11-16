// fl_bench.cc — knowhere-FL 深度功能与性能验证（精确 GT 召回 / 规模扫描 / nprobe 曲线 / 并发 QPS）
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>
#include "knowhere/index/index_factory.h"
#include "knowhere/dataset.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/version.h"

using namespace knowhere;
using Clock = std::chrono::steady_clock;

static std::vector<float> gen(int64_t n, int64_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = d(rng);
    return v;
}

// 精确 L2 top-k 真值（OpenMP 并行）
static void ground_truth(const float* base, int64_t nb, const float* q, int64_t nq,
                         int64_t dim, int64_t k, std::vector<int64_t>& gt) {
    gt.assign(nq * k, -1);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < nq; i++) {
        std::vector<std::pair<float, int64_t>> best(k, {1e30f, -1});
        const float* xq = q + i * dim;
        for (int64_t j = 0; j < nb; j++) {
            const float* xb = base + j * dim;
            float s = 0;
            for (int64_t d = 0; d < dim; d++) { float df = xb[d] - xq[d]; s += df * df; }
            if (s < best[k - 1].first) {
                best[k - 1] = {s, j};
                for (int64_t t = k - 1; t > 0 && best[t].first < best[t - 1].first; t--)
                    std::swap(best[t], best[t - 1]);
            }
        }
        for (int64_t t = 0; t < k; t++) gt[i * k + t] = best[t].second;
    }
}

static double recall_at_k(const int64_t* ids, const std::vector<int64_t>& gt, int64_t nq, int64_t k, int64_t gk) {
    double hit = 0;
    for (int64_t i = 0; i < nq; i++)
        for (int64_t a = 0; a < k; a++)
            for (int64_t b = 0; b < k; b++)
                if (gt[i * gk + b] == ids[i * k + a]) { hit += 1.0; break; }
    return hit / (double)(nq * k);
}

static Json mkcfg(int64_t dim, int64_t topk, int nlist, int nprobe) {
    Json cfg;
    cfg[meta::DIM] = dim;
    cfg[meta::METRIC_TYPE] = metric::L2;
    cfg[meta::TOPK] = topk;
    cfg["nlist"] = nlist;
    cfg["nprobe"] = nprobe;
    return cfg;
}

int main() {
    const int64_t dim = 128;
    auto version = Version::GetCurrentVersion().VersionNumber();
    printf("[bench] knowhere api version=%d dim=%ld\n", version, dim);

    // ============ A. 规模扫描：nb = 100k / 500k / 1M（recall 对精确 GT） ============
    for (int64_t nb : {100000L, 500000L, 1000000L}) {
        const int64_t nq = 200, topk = 10;
        auto base = gen(nb, dim, 42);
        auto query = gen(nq, dim, 7);
        std::vector<int64_t> gt;
        auto g0 = Clock::now();
        ground_truth(base.data(), nb, query.data(), nq, dim, topk, gt);
        printf("[A] nb=%-8ld GT %.1fs\n", nb, std::chrono::duration<double>(Clock::now() - g0).count());

        int iters = nb <= 100000 ? 20 : (nb <= 500000 ? 5 : 3);
        for (auto name : {"FL_BRUTE_FORCE", "FL_IVF_FLAT"}) {
            auto idx_r = IndexFactory::Instance().Create<fp32>(name, version);
            if (!idx_r.has_value()) { printf("[A] %s create FAIL\n", name); continue; }
            auto idx = idx_r.value();
            Json cfg = mkcfg(dim, topk, 256, 16);
            auto train_ds = GenDataSet(nb, dim, base.data());
            auto query_ds = GenDataSet(nq, dim, query.data());
            auto t0 = Clock::now();
            auto st = idx.Build(train_ds, cfg);
            double tb = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if (st != Status::success) { printf("[A] %s nb=%ld build FAIL\n", name, nb); continue; }
            auto res = idx.Search(query_ds, cfg, nullptr);
            if (!res.has_value()) { printf("[A] %s nb=%ld search FAIL\n", name, nb); continue; }
            double rec = recall_at_k(res.value()->GetIds(), gt, nq, topk, topk);
            auto t2 = Clock::now();
            for (int it = 0; it < iters; it++) { volatile auto r = idx.Search(query_ds, cfg, nullptr); }
            double ts = std::chrono::duration<double, std::milli>(Clock::now() - t2).count() / iters;
            printf("[A] nb=%-8ld %-14s build=%8.1fms  search=%7.2fms/batch (%5.2fms/q, %6.0f QPS)  recall@10=%.4f\n",
                   nb, name, tb, ts, ts / nq, nq * 1000.0 / ts, rec);
        }
    }

    // ============ B. nprobe 扫描（nb=1M IVF_FLAT，recall-QPS 权衡曲线） ============
    {
        const int64_t nb = 1000000, nq = 200, topk = 10;
        auto base = gen(nb, dim, 42);
        auto query = gen(nq, dim, 7);
        std::vector<int64_t> gt;
        ground_truth(base.data(), nb, query.data(), nq, dim, topk, gt);
        auto idx_r = IndexFactory::Instance().Create<fp32>("FL_IVF_FLAT", version);
        auto idx = idx_r.value();
        Json cfg = mkcfg(dim, topk, 256, 16);
        idx.Build(GenDataSet(nb, dim, base.data()), cfg);
        auto query_ds = GenDataSet(nq, dim, query.data());
        for (int nprobe : {1, 4, 16, 64, 128}) {
            cfg["nprobe"] = nprobe;
            auto res = idx.Search(query_ds, cfg, nullptr);
            if (!res.has_value()) { printf("[B] nprobe=%d search FAIL\n", nprobe); continue; }
            double rec = recall_at_k(res.value()->GetIds(), gt, nq, topk, topk);
            auto t2 = Clock::now();
            for (int it = 0; it < 5; it++) { volatile auto r = idx.Search(query_ds, cfg, nullptr); }
            double ts = std::chrono::duration<double, std::milli>(Clock::now() - t2).count() / 5;
            printf("[B] nprobe=%-4d search=%7.2fms/batch (%5.2fms/q, %6.0f QPS)  recall@10=%.4f\n",
                   nprobe, ts, ts / nq, nq * 1000.0 / ts, rec);
        }
    }

    // ============ C. topk 扫描（nb=500k IVF_FLAT nprobe=16，GT@100） ============
    {
        const int64_t nb = 500000, nq = 200;
        auto base = gen(nb, dim, 42);
        auto query = gen(nq, dim, 7);
        std::vector<int64_t> gt;
        ground_truth(base.data(), nb, query.data(), nq, dim, 100, gt);
        auto idx_r = IndexFactory::Instance().Create<fp32>("FL_IVF_FLAT", version);
        auto idx = idx_r.value();
        Json cfg = mkcfg(dim, 100, 256, 16);
        idx.Build(GenDataSet(nb, dim, base.data()), cfg);
        auto query_ds = GenDataSet(nq, dim, query.data());
        for (int64_t topk : {1L, 10L, 100L}) {
            cfg[meta::TOPK] = topk;
            auto res = idx.Search(query_ds, cfg, nullptr);
            if (!res.has_value()) { printf("[C] topk=%ld search FAIL\n", topk); continue; }
            double rec = recall_at_k(res.value()->GetIds(), gt, nq, topk, 100);
            auto t2 = Clock::now();
            for (int it = 0; it < 5; it++) { volatile auto r = idx.Search(query_ds, cfg, nullptr); }
            double ts = std::chrono::duration<double, std::milli>(Clock::now() - t2).count() / 5;
            printf("[C] topk=%-4ld search=%7.2fms/batch (%5.2fms/q, %6.0f QPS)  recall@%ld=%.4f\n",
                   topk, ts, ts / nq, nq * 1000.0 / ts, topk, rec);
        }
    }

    // ============ D. 并发吞吐（nb=1M IVF_FLAT nprobe=16，T 线程 × 3s） ============
    {
        const int64_t nb = 1000000, nq = 200, topk = 10;
        auto base = gen(nb, dim, 42);
        auto query = gen(nq, dim, 7);
        auto idx_r = IndexFactory::Instance().Create<fp32>("FL_IVF_FLAT", version);
        auto idx = idx_r.value();
        Json cfg = mkcfg(dim, topk, 256, 16);
        idx.Build(GenDataSet(nb, dim, base.data()), cfg);
        auto query_ds = GenDataSet(nq, dim, query.data());
        for (int T : {1, 8, 32}) {
            std::vector<std::thread> pool;
            std::vector<int64_t> cnt(T, 0);
            auto start = Clock::now();
            for (int t = 0; t < T; t++)
                pool.emplace_back([&, t] {
                    while (std::chrono::duration<double>(Clock::now() - start).count() < 3.0) {
                        volatile auto r = idx.Search(query_ds, cfg, nullptr);
                        cnt[t]++;
                    }
                });
            for (auto& th : pool) th.join();
            double el = std::chrono::duration<double>(Clock::now() - start).count();
            int64_t tot = 0;
            for (auto c : cnt) tot += c;
            printf("[D] threads=%-3d total_batches=%-6ld elapsed=%.1fs  throughput=%.0f QPS\n",
                   T, tot, el, tot * nq / el);
        }
    }

    printf("[bench] DONE\n");
    return 0;
}
