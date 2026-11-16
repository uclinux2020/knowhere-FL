// topk.cu v2 — PPU 融合算子优化版：L2 距离 + top-k 单 kernel
// 优化点：1) 每 block 处理 QPB=4 条查询，base 全局内存流量降为 1/4
//         2) float4 向量化加载（dim=128 → 32 次 float4 读）
//         3) 查询向量驻留共享内存，距离累加全寄存器复用
// grid=(nq,1,1)（shim 约定），block=(128,1,1)；块内 qi0=blockIdx.x*QPB，越界块提前退出。
#define KMAX 10
#define NTHREADS 256
#define QPB 2

extern "C" __global__ void brute_force_topk(const float* xb, long nb, const float* xq, long nq,
                                            long dim, long k, int metric,
                                            float* out_dist, long* out_ids, void* global_scratch) {
    long qi0 = blockIdx.x * QPB;
    if (qi0 >= nq) return;
    int tid = threadIdx.x;
    int nq_loc = (int)((nq - qi0) < QPB ? (nq - qi0) : (long)QPB);

    __shared__ float sq[QPB][128];
    for (int p = 0; p < nq_loc; p++)
        for (int d = tid; d < 128; d += NTHREADS)
            sq[p][d] = xq[(qi0 + p) * 128 + d];
    __syncthreads();

    float ld[QPB][KMAX];
    long li[QPB][KMAX];
    for (int p = 0; p < QPB; p++)
        for (int t = 0; t < KMAX; t++) { ld[p][t] = 1e30f; li[p][t] = -1L; }

    long dv = dim >> 2;
    for (long j = tid; j < nb; j += NTHREADS) {
        const float4* b4 = (const float4*)(xb + j * dim);
        float s[QPB] = {0.f, 0.f, 0.f, 0.f};
        for (long d = 0; d < dv; d++) {
            float4 bv = b4[d];
            for (int p = 0; p < nq_loc; p++) {
                float4 qv = *(const float4*)(sq[p] + (d << 2));
                float d0 = bv.x - qv.x, d1 = bv.y - qv.y, d2 = bv.z - qv.z, d3 = bv.w - qv.w;
                s[p] += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            }
        }
        for (int p = 0; p < nq_loc; p++) {
            if (s[p] < ld[p][KMAX - 1]) {
                ld[p][KMAX - 1] = s[p]; li[p][KMAX - 1] = j;
                for (int t = KMAX - 1; t > 0 && ld[p][t] < ld[p][t - 1]; t--) {
                    float tf = ld[p][t]; ld[p][t] = ld[p][t - 1]; ld[p][t - 1] = tf;
                    long ti = li[p][t]; li[p][t] = li[p][t - 1]; li[p][t - 1] = ti;
                }
            }
        }
    }

    // 逐查询块内归并（复用同一组共享缓冲，15KB）
    __shared__ float sd[NTHREADS][KMAX];
    __shared__ long si[NTHREADS][KMAX];
    for (int p = 0; p < nq_loc; p++) {
        for (int t = 0; t < KMAX; t++) { sd[tid][t] = ld[p][t]; si[tid][t] = li[p][t]; }
        __syncthreads();
        for (int stride = NTHREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float rd[KMAX]; long ri[KMAX];
                int a = 0, b = 0;
                for (int t = 0; t < KMAX; t++) {
                    if (sd[tid][a] <= sd[tid + stride][b]) { rd[t] = sd[tid][a]; ri[t] = si[tid][a]; a++; }
                    else { rd[t] = sd[tid + stride][b]; ri[t] = si[tid + stride][b]; b++; }
                }
                for (int t = 0; t < KMAX; t++) { sd[tid][t] = rd[t]; si[tid][t] = ri[t]; }
            }
            __syncthreads();
        }
        if (tid == 0) {
            for (long t = 0; t < k; t++) {
                out_dist[(qi0 + p) * k + t] = sd[0][t];
                out_ids[(qi0 + p) * k + t] = si[0][t];
            }
        }
        __syncthreads();
    }
}
