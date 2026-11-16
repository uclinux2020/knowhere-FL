// t7.c — PPU 融合 topk kernel 真机基准：shim hgAot* 路径 + CPU GT 校验 + QPS 对比
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <random>
#include <vector>

extern "C" {
int hgDeviceCount(void);
int hgSetDevice(int);
void* hgMemAlloc(size_t);
int hgMemFree(void*);
int hgMemcpyH2D(void*, const void*, size_t, void*);
int hgMemcpyD2H(void*, const void*, size_t, void*);
void* hgStreamCreate(void);
int hgStreamSynchronize(void*);
void* hgAotKernelLoad(const char*, const char*);
int hgAotLaunchTopK(const void*, const void*, long, const void*, long, long, long,
                    int, float*, long*, void*);
}

static std::vector<float> gen(long n, long dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = d(rng);
    return v;
}

int main(int argc, char** argv) {
    long nb = argc > 1 ? atol(argv[1]) : 1000000;
    long nq = argc > 2 ? atol(argv[2]) : 200;
    long dim = 128, k = 10;
    int iters = argc > 3 ? atoi(argv[3]) : 10;
    printf("[t7] nb=%ld nq=%ld dim=%ld k=%ld iters=%d\n", nb, nq, dim, k, iters);

    if (hgDeviceCount() < 1 || hgSetDevice(0) != 0) { printf("[t7] device FAIL\n"); return 2; }
    auto base = gen(nb, dim, 42);
    auto query = gen(nq, dim, 7);

    float* d_xb = (float*)hgMemAlloc(nb * dim * 4);
    float* d_xq = (float*)hgMemAlloc(nq * dim * 4);
    float* d_od = (float*)hgMemAlloc(nq * k * 4);
    long* d_oi = (long*)hgMemAlloc(nq * k * 8);
    if (!d_xb || !d_xq || !d_od || !d_oi) { printf("[t7] alloc FAIL\n"); return 3; }
    hgMemcpyH2D(d_xb, base.data(), nb * dim * 4, NULL);
    hgMemcpyH2D(d_xq, query.data(), nq * dim * 4, NULL);
    printf("[t7] H2D done (base %.0f MB)\n", nb * dim * 4 / 1e6);

    void* kern = hgAotKernelLoad("brute_force_topk", "/root/topk.cubin");
    if (!kern) { printf("[t7] kernel load FAIL\n"); return 4; }
    void* stream = hgStreamCreate();
    if (!stream) { printf("[t7] stream FAIL\n"); return 5; }

    // warmup + correctness
    int rc = hgAotLaunchTopK(kern, d_xb, nb, d_xq, nq, dim, k, 0, d_od, d_oi, stream);
    hgStreamSynchronize(stream);
    if (rc != 0) { printf("[t7] launch FAIL rc=%d\n", rc); return 6; }
    std::vector<float> h_od(nq * k);
    std::vector<long> h_oi(nq * k);
    hgMemcpyD2H(h_od.data(), d_od, nq * k * 4, NULL);
    hgMemcpyD2H(h_oi.data(), d_oi, nq * k * 8, NULL);

    // CPU GT 校验前 20 条查询
    int bad = 0;
    for (long i = 0; i < 20 && i < nq; i++) {
        std::vector<std::pair<float, long>> best(k, {1e30f, -1});
        const float* q = query.data() + i * dim;
        for (long j = 0; j < nb; j++) {
            const float* b = base.data() + j * dim;
            float s = 0;
            for (long d = 0; d < dim; d++) { float df = b[d] - q[d]; s += df * df; }
            if (s < best[k - 1].first) {
                best[k - 1] = {s, j};
                for (long t = k - 1; t > 0 && best[t].first < best[t - 1].first; t--)
                    std::swap(best[t], best[t - 1]);
            }
        }
        for (long t = 0; t < k; t++) {
            long got = h_oi[i * k + t];
            bool found = false;
            for (long u = 0; u < k; u++) if (best[u].second == got) found = true;
            if (!found) bad++;
        }
    }
    printf("[t7] correctness: %d/%ld wrong in first 20 queries (k=%ld)\n", bad, 20 * k, k);

    // 性能：N 次 launch + sync
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++)
        hgAotLaunchTopK(kern, d_xb, nb, d_xq, nq, dim, k, 0, d_od, d_oi, stream);
    hgStreamSynchronize(stream);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
    printf("[t7] PPU fused topk: %.3f ms/batch (%.3f ms/q, %.0f QPS)\n",
           ms, ms / nq, nq * 1000.0 / ms);

    hgMemFree(d_xb); hgMemFree(d_xq); hgMemFree(d_od); hgMemFree(d_oi);
    printf("[t7] DONE\n");
    return 0;
}
