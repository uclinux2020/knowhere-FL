// libppu_runtime.so — knowhere-FL PPU 适配层 launch-wrapper（FlagFit 忠实版）
//
// 参照 FlagFit crates/fitel-sys/src/ppu_sys.rs 的 PpuLib：
//   - dlopen 厂商库 libhggc.so（PPU_LIB_NAMES = libhggc.so[.1/.2]，
//     搜索 $PPU_SDK/$PPU_HOME/$PPU_ROOT + /usr/local/PPU_SDK + /opt/ppu + /usr/local/ppu）
//   - hgLaunchKernelEx + HGLaunchConfig（stream@offset32, attrs, num_attrs）
//   - kernel_params = 声明参数 + Triton global_scratch，3 个 HIDDEN_PARAM 槽由 SDK 填充
// AOT artifact 加载与 FlagFit 一致走 CUDA 兼容层 cuModuleLoadData（libcuda.so.1）。
//
// 导出的 13 个符号即 knowhere-FL fl_ppu_adapter.cc 的 extern "C" 契约。
// 所有失败路径返回 nullptr/-1，适配层按 Phase 1 设计回退 CPU。
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int HGresult;                 // HG_SUCCESS = 0
typedef int HGdevice;
typedef unsigned long long HGdeviceptr;
typedef void* HGcontext;
typedef void* HGstream;
typedef void* HGfunction;

// 与 fitel-sys ppu_sys.rs 的 #[repr(C)] HGLaunchConfig 完全一致：
// grid(3) block(3) shared_mem_bytes [4B pad] stream attrs num_attrs
typedef struct {
    unsigned int grid_dim_x, grid_dim_y, grid_dim_z;
    unsigned int block_dim_x, block_dim_y, block_dim_z;
    unsigned int shared_mem_bytes;
    HGstream stream;
    void* attrs;
    unsigned int num_attrs;
} HGLaunchConfig;

// ---- 厂商库动态解析 ----
typedef HGresult (*FnHgInit)(unsigned int);
typedef HGresult (*FnHgDeviceGetCount)(int*);
typedef HGresult (*FnHgDeviceGet)(HGdevice*, int);
typedef HGresult (*FnHgDevicePrimaryCtxRetain)(HGcontext*, HGdevice);
typedef HGresult (*FnHgCtxSetCurrent)(HGcontext);
typedef HGresult (*FnHgCtxGetCurrent)(HGcontext*);
typedef HGresult (*FnHgCtxSynchronize)(void);
typedef HGresult (*FnHgMemAlloc)(HGdeviceptr*, size_t);
typedef HGresult (*FnHgMemFree)(HGdeviceptr);
typedef HGresult (*FnHgMemcpyHtoD)(HGdeviceptr, const void*, size_t);
typedef HGresult (*FnHgMemcpyDtoH)(void*, HGdeviceptr, size_t);
typedef HGresult (*FnHgMemcpyDtoD)(HGdeviceptr, HGdeviceptr, size_t);
typedef HGresult (*FnHgStreamCreate)(HGstream*, unsigned int);
typedef HGresult (*FnHgStreamDestroy)(HGstream);
typedef HGresult (*FnHgStreamSynchronize)(HGstream);
typedef HGresult (*FnHgLaunchKernelEx)(const HGLaunchConfig*, HGfunction, void**, void**);

typedef struct {
    void* handle;
    FnHgDeviceGetCount DeviceGetCount;
    FnHgDeviceGet DeviceGet;
    FnHgDevicePrimaryCtxRetain DevicePrimaryCtxRetain;
    FnHgCtxSetCurrent CtxSetCurrent;
    FnHgCtxGetCurrent CtxGetCurrent;
    FnHgCtxSynchronize CtxSynchronize;
    FnHgMemAlloc MemAlloc;
    FnHgMemFree MemFree;
    FnHgMemcpyHtoD MemcpyHtoD;
    FnHgMemcpyDtoH MemcpyDtoH;
    FnHgMemcpyDtoD MemcpyDtoD;
    FnHgStreamCreate StreamCreate;
    FnHgStreamDestroy StreamDestroy;
    FnHgStreamSynchronize StreamSynchronize;
    FnHgLaunchKernelEx LaunchKernelEx;
} PpuLib;

static PpuLib g_ppu;
static int g_ppu_state = 0;  // 0=未初始化 1=就绪 -1=不可用

static void* g_cuda = NULL;  // libcuda.so.1（cuModuleLoadData / cuModuleGetFunction / cuStream*）
typedef int (*FnCuModuleLoadData)(void*, const void*);
typedef int (*FnCuModuleGetFunction)(void*, void*, const char*);
static FnCuModuleLoadData p_cuModuleLoadData = NULL;
static FnCuModuleGetFunction p_cuModuleGetFunction = NULL;

static void* try_open(const char* dir, const char* name) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/lib/%s", dir, name);
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (h) return h;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    return dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
}

#define RESOLVE(field, sym)                                            \
    do {                                                               \
        g_ppu.field = (FnHg##field)dlsym(g_ppu.handle, "hg" #field);   \
        if (!g_ppu.field) { g_ppu_state = -1; return -1; }             \
    } while (0)

static int ppu_init(void) {
    if (g_ppu_state != 0) return g_ppu_state;
    // 厂商库分层：libhggc.so 是 API 门面（内部经 PLT 依赖 libhgl/libcudart，
    // dlsym 其句柄会拿到被间插的前向符号，实测形成 hgStreamCreate 递归崩溃）；
    // libhg_wrapper.so 才是 16 个 hg* 函数的真实实现（含 hgcrt_sit_stream.cpp）。
    static const char* lib_names[] = {"libhg_wrapper.so", "libhggc.so", "libhggc.so.1", "libhggc.so.2"};
    static const char* env_roots[] = {"PPU_SDK", "PPU_HOME", "PPU_ROOT"};
    static const char* def_roots[] = {"/usr/local/PPU_SDK", "/opt/ppu", "/usr/local/ppu"};

    for (size_t i = 0; i < 3 && !g_ppu.handle; ++i) {
        for (size_t e = 0; e < 3 && !g_ppu.handle; ++e) {
            const char* root = getenv(env_roots[e]);
            if (root && *root) g_ppu.handle = try_open(root, lib_names[i]);
        }
        for (size_t d = 0; d < 3 && !g_ppu.handle; ++d)
            g_ppu.handle = try_open(def_roots[d], lib_names[i]);
    }
    if (!g_ppu.handle) { g_ppu_state = -1; return -1; }

    // 厂商 driver API 要求先 hgInit(0)（见 SDK 头文件 ::hgInit 注释）
    FnHgInit fnInit = (FnHgInit)dlsym(g_ppu.handle, "hgInit");
    if (fnInit && fnInit(0) != 0) { g_ppu_state = -1; return -1; }

    RESOLVE(DeviceGetCount, DeviceGetCount);
    RESOLVE(DeviceGet, DeviceGet);
    RESOLVE(DevicePrimaryCtxRetain, DevicePrimaryCtxRetain);
    RESOLVE(CtxSetCurrent, CtxSetCurrent);
    RESOLVE(CtxGetCurrent, CtxGetCurrent);
    RESOLVE(CtxSynchronize, CtxSynchronize);
    RESOLVE(MemAlloc, MemAlloc);
    RESOLVE(MemFree, MemFree);
    RESOLVE(MemcpyHtoD, MemcpyHtoD);
    RESOLVE(MemcpyDtoH, MemcpyDtoH);
    RESOLVE(MemcpyDtoD, MemcpyDtoD);
    RESOLVE(StreamCreate, StreamCreate);
    RESOLVE(StreamDestroy, StreamDestroy);
    RESOLVE(StreamSynchronize, StreamSynchronize);
    RESOLVE(LaunchKernelEx, LaunchKernelEx);

    // CUDA 兼容驱动层（模块加载），与 FlagFit cuda_sys 路径一致
    static const char* cuda_names[] = {"libcuda.so", "libcuda.so.1"};
    static const char* cuda_dir = "/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib";
    for (size_t i = 0; i < 2 && !g_cuda; ++i) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s", cuda_dir, cuda_names[i]);
        g_cuda = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    }
    if (!g_cuda) g_cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (g_cuda) {
        typedef int (*FnCuInit)(unsigned int);
        FnCuInit fnCuInit = (FnCuInit)dlsym(g_cuda, "cuInit");
        if (fnCuInit) fnCuInit(0);
        p_cuModuleLoadData = (FnCuModuleLoadData)dlsym(g_cuda, "cuModuleLoadData");
        p_cuModuleGetFunction = (FnCuModuleGetFunction)dlsym(g_cuda, "cuModuleGetFunction");
    }

    g_ppu_state = 1;
    return 1;
}

// FlagFit BUG-H：launch 前必须绑定设备 primary context
static int ensure_ctx(int ordinal) {
    if (ppu_init() < 0) return -1;
    HGcontext ctx = NULL;
    if (g_ppu.CtxGetCurrent(&ctx) == 0 && ctx != NULL) return 0;
    HGdevice dev = 0;
    if (g_ppu.DeviceGet(&dev, ordinal) != 0) return -1;
    if (g_ppu.DevicePrimaryCtxRetain(&ctx, dev) != 0) return -1;
    if (g_ppu.CtxSetCurrent(ctx) != 0) return -1;
    return 0;
}

// ================= 导出契约（knowhere-FL fl_ppu_adapter.cc） =================

int hgDeviceCount(void) {
    if (ppu_init() < 0) return 0;
    int n = 0;
    if (g_ppu.DeviceGetCount(&n) != 0) return 0;
    return n;
}

int hgSetDevice(int device_id) { return ensure_ctx(device_id); }

int hgDeviceSynchronize(void) {
    if (ppu_init() < 0) return -1;
    return g_ppu.CtxSynchronize() == 0 ? 0 : -1;
}

void* hgMemAlloc(size_t bytes) {
    if (ensure_ctx(0) != 0) return NULL;
    HGdeviceptr p = 0;
    if (g_ppu.MemAlloc(&p, bytes) != 0) return NULL;
    return (void*)p;
}

int hgMemFree(void* ptr) {
    if (ptr == NULL) return 0;
    if (ppu_init() < 0) return -1;
    return g_ppu.MemFree((HGdeviceptr)ptr) == 0 ? 0 : -1;
}

int hgMemcpyH2D(void* dst, const void* src, size_t bytes, void* stream) {
    (void)stream;  // 同步版本语义正确（厂商 Async 变体接入见 TODO Phase 2）
    if (ppu_init() < 0) return -1;
    return g_ppu.MemcpyHtoD((HGdeviceptr)dst, src, bytes) == 0 ? 0 : -1;
}

int hgMemcpyD2H(void* dst, const void* src, size_t bytes, void* stream) {
    (void)stream;
    if (ppu_init() < 0) return -1;
    return g_ppu.MemcpyDtoH(dst, (HGdeviceptr)src, bytes) == 0 ? 0 : -1;
}

int hgMemcpyD2D(void* dst, const void* src, size_t bytes, void* stream) {
    (void)stream;
    if (ppu_init() < 0) return -1;
    return g_ppu.MemcpyDtoD((HGdeviceptr)dst, (HGdeviceptr)src, bytes) == 0 ? 0 : -1;
}

void* hgStreamCreate(void) {
    // 必须转发到厂商 hg 层：libcudart 的 cudartStreamCreate 内部回调
    // hgStreamCreate（generated_hgcrt.cpp:1756），若我们走 cu 层会形成
    // 符号间插递归（gdb 回溯证实）。dlopen 句柄内解析拿到厂商真身。
    if (ensure_ctx(0) != 0) return NULL;
    HGstream s = NULL;
    if (g_ppu.StreamCreate(&s, 0) != 0) return NULL;
    return (void*)s;
}

int hgStreamDestroy(void* stream) {
    if (stream == NULL) return 0;
    if (ppu_init() < 0) return -1;
    return g_ppu.StreamDestroy((HGstream)stream) == 0 ? 0 : -1;
}

int hgStreamSynchronize(void* stream) {
    if (ppu_init() < 0) return -1;
    return g_ppu.StreamSynchronize((HGstream)stream) == 0 ? 0 : -1;
}

// ---- AOT kernel 注册（FlagTree 产物，FlagFit loader.rs 同构） ----

void* hgAotKernelLoad(const char* name, const char* artifact_path) {
    if (!name || !artifact_path) return NULL;
    FILE* f = fopen(artifact_path, "rb");
    if (!f) return NULL;  // Phase 1：fl_kernels/*.ppu 未产出 → CPU 回退
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    char* blob = (char*)malloc((size_t)sz);
    if (!blob) { fclose(f); return NULL; }
    size_t rd = fread(blob, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(blob); return NULL; }

    if (ensure_ctx(0) != 0 || !p_cuModuleLoadData || !p_cuModuleGetFunction) {
        free(blob);
        return NULL;
    }
    void* mod = NULL;
    if (p_cuModuleLoadData(&mod, blob) != 0) { free(blob); return NULL; }
    free(blob);
    void* fn = NULL;
    if (p_cuModuleGetFunction(&fn, mod, name) != 0) return NULL;
    return fn;
}

int hgAotLaunchTopK(const void* kernel, const void* xb, long nb, const void* xq, long nq, long dim, long k,
                    int metric, float* out_dist, long* out_ids, void* stream) {
    if (!kernel) return -1;  // artifact 未加载 → CPU 回退
    if (ensure_ctx(0) != 0) return -1;

    // FlagFit 约定：kernelParams = 声明参数 + global_scratch；3 个 HIDDEN_PARAM
    // 槽由 SDK 在 hgLaunchKernelEx 内填充。block=(128,1,1) 为 FlagFit 多数算子
    // 共享档位（见 loader.rs 注释），grid.x 按 nq 分块。
    int64_t nb_v = nb, nq_v = nq, dim_v = dim, k_v = k;
    int32_t metric_v = metric;
    void* global_scratch = NULL;
    void* kp[] = {(void*)&xb, &nb_v, (void*)&xq, &nq_v, &dim_v, &k_v,
                  &metric_v, &out_dist, &out_ids, &global_scratch};

    HGLaunchConfig cfg;
    cfg.grid_dim_x = (unsigned int)(nq);
    cfg.grid_dim_y = 1;
    cfg.grid_dim_z = 1;
    cfg.block_dim_x = 256;
    cfg.block_dim_y = 1;
    cfg.block_dim_z = 1;
    cfg.shared_mem_bytes = 0;
    cfg.stream = (HGstream)stream;
    cfg.attrs = NULL;
    cfg.num_attrs = 0;

    return g_ppu.LaunchKernelEx(&cfg, (HGfunction)kernel, kp, NULL) == 0 ? 0 : -1;
}
