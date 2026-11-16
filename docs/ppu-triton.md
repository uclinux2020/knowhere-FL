# PPU：FlagOS Triton / Triton-TLE 实现与验证

## 归因结论

FlagOS 是多算子库生态，**不等于 FlagGems**。FlagTree 是编译链；使用它不自动意味着调用了 FlagGems 或其他现成算子库。

修改前的 `fl_ppu_adapter.cc` 用手写 `hgAot*` 声明和不存在于仓库的 `.ppuo` 路径表示 brute-force 接口；IVF_FLAT、IVF_PQ、kmeans assign、graph search 返回失败并依赖 CPU 回退。三个旧 Python composition 示例通过 ATen 使用 PyTorch，且 FlagGems enable 失败后继续运行 stock backend。因此，注释、示例 QPS 或成功的 CPU UT 都不能证明 Knowhere 的 PPU 加速来自 FlagOS 算子。

提交前同步远程 `main` 时发现新增提交 `823671fe`：其中 `ppu/kernels/brute_force_topk.cu` 是 CUDA-C 融合算子，`ppu/ppu_runtime.c` 是兼容 shim，`docs/build-verification-report-20261115.md` 和 `ppu/README.md` 记录的 6.76x–8.26x 加速属于该历史 CUDA-C 路径，**不是 FlagOS Triton/Triton-TLE 算子库的性能证据，也不是本次实现的成绩**。这些历史文件保留供对照；本次 `WITH_FL_PPU` 构建改为原生 HG + FlagTree AOT，不编译或链接该 legacy shim/kernel。

本次路径为：

```
离线：检索 Triton / Triton-TLE 源码 -> FlagTree PPU backend -> .hgbin + manifest.json
运行：Knowhere FlDeviceOps -> 原生 HG context/memory/module/launch -> PPU -> host result
```

本次没有引入、导入或链接 PyTorch；不使用 ATen、CUDA-C kernel、CUDA runtime 或兼容层 `libppu_runtime.so`。离线 Python 是构建工具，不嵌入 C++ 服务。TLE 的可选 raw integrations 会尝试导入 Torch，构建工具通过 import blocker 将它视为未安装，并在编译结束检查 Torch 未被导入。现有三个 PyTorch composition 文件是旧示例，不参与本次 AOT 和 C++ 路径，不应拿它们证明本次性能。

这属于 **FlagOS/FlagTree 的自实现 Triton/Triton-TLE 检索算子**，不是声称复用了 FlagGems 的 mm/topk，也不声称调用了未经核实的其他算子库。后续复用其他 FlagOS 库，应核实其无 Torch 接口、PPU 支持和实际执行链。

## 当前实现范围

- FP32 L2 / IP brute-force：分块距离，设备 top-k，设备合并；并列分数按全局 ID 升序。
- IVF_FLAT：上传 host CSR candidate lists，在设备按候选计算距离与 top-k。
- IVF_PQ：按 subquantizer codebook 计算 ADC 分数，支持单字节 codes；非 residual 的现有 FL 编码布局。
- kmeans assign：centroids 作为 base vectors，设备 top-1，返回 host centroid IDs；centroid update 仍在 CPU。
- Triton-TLE：向量加载实际使用 `tle.load(..., is_async=True)`，不是仅添加名称标签。
- CAGRA 图遍历：`graph_search == nullptr`，保留 CPU 回退，**未实现 PPU 图遍历**。

当前先按 query 顺序处理有界 score chunks，scratch 大小不随 `nb*nq` 增长。不支持非零 global/profile scratch、clustered CTA；不兼容产物明确拒绝。CUDA-named `GPUTarget('cuda', 80, 32)` 是本机 PPU FlagTree plugin 的内部 target 标识；通过 `FLAGTREE_BACKEND=ppu` 选择实际 PPU 编译器，并强制检查输出为 `hgbin`。

## 编译库

原生 SDK 需有 `include/hggc.h` 与 `lib/libhggc.so`；不再链接占位 `ppu_runtime`。

```bash
PPU_SDK_PATH=/usr/local/PPU_SDK make WITH_FL=True WITH_FL_PPU=True WITH_UT=True WITH_DISKANN=False
```

完整库仍需项目 `conanfile.py` 声明的全部依赖。本次开始时，`boost/1.83.0#4e8a94ac1b88312af95eded83cd81ca8` 不在本地缓存，启用的 Conan remote 也未提供该 revision，故完整库及 Catch2 FL UT 未完成。提交前合入的远程 `823671fe` 已移除部分 recipe revision pinning；保留该远程改动，但未针对更新后的依赖图重新完成完整构建，不能沿用旧错误断言当前图仍被同一 revision 阻塞。

## 编译 specialization

以 `dim=128, k=10, metric=L2` 为例（必须与调用匹配）：

```bash
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export PPU_SDK_PATH=/usr/local/PPU_SDK
export FLAGTREE_BACKEND=ppu
export UMD_PLATFORM_TYPE=1
export HGGC_DRIVER_CANDIDATE=UMD
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}

python python/fl_kernels/compile_ppu.py --dim 128 --k 10 --tle \
  --output build/ppu-kernels/d128_k10_l2_m0_s0
export KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-kernels"
```

普通 Triton 不传 `--tle`。IP 增加 `--metric IP`，目录使用 `_ip_`。PQ 传 `--m 16 --ksub 256`，目录使用 `_m16_s256`。Assign 需要对应 `k=1, m=0, s=0` bundle。无 PQ 时目录 `s0` 是运行接口 specialization key，manifest 的 ksub 不参与匹配。`nb/nq` 为运行时值，不需要重新编译；`dim/k/metric/PQ layout` 不匹配或产物缺失会报错并触发调用侧回退。

每个 context 独立缓存模块，资源释放时绑定原 context。首次 `GetDeviceOps/DeviceCount/DeviceAvailable` 会懒注册 PPU adapter；不需要 Milvus 额外初始化胶水。`RegisterDeviceOps(nullptr)` 后不会反复重新注册。

## 独立真实设备测试（无完整 Conan graph）

需 FlagTree PPU Python 和 nlohmann/json C++ 头文件（不需要 Torch 或 NumPy）。从仓库根目录执行：

```bash
cmake -S tests/ppu -B build/ppu-contract \
  -DPPU_SDK_PATH=/usr/local/PPU_SDK \
  -DNLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include -DCMAKE_BUILD_TYPE=Release
cmake --build build/ppu-contract -j2

python python/fl_kernels/test_retrieval_triton.py \
  --output build/ppu-kernels --test-executable build/ppu-contract/knowhere_ppu_test --device 0
python python/fl_kernels/test_retrieval_triton.py --tle \
  --output build/ppu-kernels-tle --test-executable build/ppu-contract/knowhere_ppu_test --device 0
```

本次在 ZW810E 的空闲 device 0 上实测：GNU C++ 13.3，原生 HG SDK，FlagTree PPU Triton 3.6。普通 Triton 和 Triton-TLE 两套均通过：L2/IP brute-force、跨 chunk top-k 合并、IVF candidates、PQ 对解码 codebook 的数值 oracle、nearest-centroid assign、ties、empty/fewer-than-k padding、非法 CSR 拒绝。另通过
`dim=16/k=10/m=4` 与 `dim=128/k=10/m=16` 的 Triton-TLE 真机数值测试（ksub=8）。
测试直接调用原生 PPU runtime，**不存在 CPU 回退兜底**。适配器和 fl_device 懒注册/RAII/pool accounting/unregister 测试也实际通过。

本机 FlagTree interpreter 的 load builtin/InterpreterBuilder ABI 不一致，解释器尝试失败；回归脚本改为实际编译并执行 PPU binary，不对第三方安装包打补丁。

## 不能宣称的结论

- 本次通过编译和数值验证，**未测出性能提升倍数**；没有成对的旧实现/新实现 QPS、latency、recall、训练时间、RSS/显存峰值基准。
- 顺序 query、分块同步、direct distance reduction、重复 PQ codebook 读取目前以正确性和可集成为目标，不是完成调优的性能实现。
- 未验证完整 Knowhere/Milvus E2E、并发多卡、生产规模维度/大 k/筛选组合。现有 FlMemoryPool 追踪 index staging；runtime 私有 workspace 尚未纳入该 pool 的限额，是后续资源预算工作。
- 无产物时的 CPU 回退、Host adapter 和旧 PyTorch composition 的成绩不得归因为本次 PPU Triton/TLE 性能。
