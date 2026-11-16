# Triton（结合 Triton-TLE）PPU 功能及性能验证：2026-11-16

## 结论

**功能通过，当前性能未达到加速目标。** 这是一条统一执行链：

`FlagOS Triton 算子（使用 Triton-TLE 扩展）→ FlagTree AOT → 原生 HG → ZW810E`

不是两条真机路径。本轮所有正式性能数据使用 TLE 异步加载配置，不对两个设备后端做比较。没有引入 PyTorch、NumPy、CUDA-C、CUDA runtime，也不将 FlagOS 等同于 FlagGems。AOT 工具阻止可选 integration 导入 Torch，并检查最终未导入。

## 测试对象与环境

- 待测实现 revision：`5f610aae883e8592a6e2b0d0b5a82428d1585d40`。本轮增加基准、报告及完整编译所需错误码映射；未改检索 kernel 或 runtime 的性能实现。
- PPU：ZW810E device 0，97920 MiB，测试前后空闲；Driver `1.3.2-d7f5a2`，HGGC 13.0，SDK `/usr/local/PPU_SDK`。
- CPU：Intel Xeon family 6/model 207，92 vCPU，固定 CPU 0–7，8 OpenMP线程，`OMP_PROC_BIND=close`、`OMP_PLACES=cores`；基准 `-O3 -march=native`，GNU 13.3。
- FlagTree PPU / Triton 实际版本3.6；block=128，select-block=1024，num_warps=4，num_stages=3。
- 合成 FP32 uniform[-1,1] 数据，固定 seed=42；CPU与PPU使用完全相同输入；维度128。
- 每项2次warmup，7次计时，CPU和PPU计时串行执行，并发1。首轮与完整编译有主机资源竞争，因此报告采用**完整编译停止期间的第二轮复测**；其他设备2–5有既存进程，未终止或使用。
- 此环境不是独占整机。CPU/PPU比较有硬件、缓存、SIMD差异，仅代表本机该基准边界。

## 计时与质量边界

PPU Search 批次计时：base/query已驻留设备，包含runtime workspace分配释放、candidate H2D、所有kernel launch、同步、设备结果合并、result D2H。**不含**AOT、初始base/query H2D、数据生成、CPU oracle。原始JSON另外记录首调用模块加载、input allocation/H2D和样本。

CPU基线是本轮新增的独立 `direct FP32 distance + deterministic heap` OpenMP/SIMD实现，**不是 Knowhere/Faiss CPU索引，不是历史 CUDA-C benchmark**。PQ基线扫描显式解码的codebook向量，**不是优化的CPU LUT-ADC**；因此不能将其速度比当作PQ产品级加速。

Recall是相同候选集/相同表示下的算子正确性：IVF不包含centroid routing质量；PQ对解码后的量化表示比较，**不是对原始向量的量化召回率**。以并列分数按ID升序为规则；无候选约定recall=1，不能把空结果百万QPS当有效检索吞吐。

7次样本不足以估计生产p95/p99，因此表格只给batch p50及样本最大值；JSON中的nearest-rank p95在7样本下就是最大值，不作为尾延迟保证。

## 实测结果

除nq1和assign外，nq=16；全部dim=128。`CPU/PPU`小于1表示PPU更慢。

| Workload | CPU p50 ms | PPU p50 ms | PPU batch max ms | QPS | CPU/PPU | Recall |
|---|---:|---:|---:|---:|---:|---:|
| bf 10k, k10 | 0.411 | 9.663 | 9.689 | 1655.8 | 0.0425 | 1.000 |
| bf 100k, k10 | 3.978 | 93.275 | 97.011 | 171.5 | 0.0426 | 1.000 |
| bf 1M, k10 | 105.517 | 886.161 | 890.015 | 18.1 | 0.1191 | 1.000 |
| bf 100k, k1 | 6.620 | 76.558 | 76.754 | 209.0 | 0.0865 | 1.000 |
| bf 100k, k100 | 6.519 | 268.629 | 268.820 | 59.6 | 0.0243 | 1.000 |
| bf 100k, IP k10 | 17.614 | 95.305 | 97.152 | 167.9 | 0.1848 | 1.000 |
| bf 10k, nq1 k10 | 0.214 | 0.587 | 0.597 | 1703.9 | 0.3649 | 1.000 |
| IVF random 50% eligible | 5.707 | 52.924 | 53.024 | 302.3 | 0.1078 | 1.000 |
| IVF random 1% eligible | 0.061 | 1.068 | 1.084 | 14982.9 | 0.0569 | 1.000 |
| IVF clustered 1% eligible | 0.042 | 1.071 | 1.085 | 14943.6 | 0.0394 | 1.000 |
| IVF fewer-than-k, 5 eligible | 0.003 | 0.947 | 0.951 | 16899.0 | 0.0031 | 1.000 |
| IVF empty | 0.003 | 0.013 | 0.014 | 1243394.5 | 0.2012 | 1.000 |
| PQ 10k, m16/ksub256, all candidates | 0.585 | 8.771 | 8.822 | 1824.2 | 0.0667 | 1.000 |
| Assign 64 vectors to 256 centroids | 0.035 | 3.310 | 3.329 | 19337.3 | 0.0104 | 1.000 |

14项性能/功能工作负载全部成功，ordered IDs及score mismatch均为0。另运行原有128维真机contract test：L2/IP、chunk merge、ties、empty/fewer-than-k、非法CSR、m16 PQ/ksub8、assign通过。ksub256由本次PQ基准验证。

完整原始样本及配置见 [JSON](ppu-validation-results-20261116.json)；包括每项7个CPU/PPU样本，不仅均值。AOT 8个specialization重新生成均成功，每项约1秒级，此次使用已有compiler cache，**不是冷编译时间**。

## 性能原因：基于源码，不冒充profiler实测

当前 `fl_ppu_runtime.cc::Search` 对query和candidate chunks做host串行循环，每块先score、select，再和上一块best合并，重复stream synchronize及同步D2D。

- 1M/1024 = 977 chunks/query，nq16；源码计算的kernel launch数量为 `16*(3*977-1)=46880`，不是设备profiler记录。
- query grid固定为1，多query不在设备并行；score tile数量受每块width=1024限制，难充分利用PPU。
- top-k通过重复reduce选rank，k100明显慢于k10，测量符合代码复杂度，但未做逐kernel计时定位。
- PQ重复读取codebook并直接算sub-distance，没有将LUT计算成本在大量candidate间摊薄。
- Assign也逐向量host调度，对小nlist尤其不划算。

建议后续在同一Triton/TLE链上优化：query/chunk并行、分层批量top-k、一次最终D2H、移除中间host同步、复用受预算约束workspace、PQ独立LUT+ADC、批量assign。必须先保留此基线再做成对回归；本轮**没有暗中修改待测实现来制造提升**。

## 编译与复现

Conan更新后的依赖图已从本地缓存解析成功，旧Boost revision错误不再是当前阻塞项。完整 `WITH_FL_PPU+WITH_UT` 编译首次暴露 `ToSegcoreErrorCode` 未覆盖 `fl_device_error`，触发`-Werror=switch`；本轮补齐permanent KnowhereError映射，并新增`tests/ut/test_fl_error_code.cc`回归。

补齐后 `libknowhere.so` 已生成，并由 `ldd` 确认直接链接原生 `libhggc.so`，没有Torch/CUDA runtime/legacy shim。但真实 `ctypes.CDLL` 加载失败：本地 `libmilvus-common.so` 需要 `F14LinkCheck<F14IntrinsicsMode=2>`，缓存的 `libfolly.so` 仅导出 `F14IntrinsicsMode=1`。这是本地依赖ABI不一致，**共享库生成不能视为完整构建/功能验证成功**。不使用`--allow-shlib-undefined`或伪造F14符号掩盖。后续需用一致的SIMD/编译配置重建milvus-common/Folly及其依赖。完整UT运行仍待该依赖问题解决；独立真机验证不等于完整Milvus E2E。

```bash
cmake -S tests/ppu -B build/ppu-contract -DCMAKE_BUILD_TYPE=Release \
  -DPPU_SDK_PATH=/usr/local/PPU_SDK \
  -DNLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include
cmake --build build/ppu-contract -j2

export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK FLAGTREE_BACKEND=ppu
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
python python/fl_kernels/validate_ppu.py --output build/ppu-validation-results/run \
  --build build/ppu-contract --device 0
python python/fl_kernels/summarize_ppu.py build/ppu-validation-results/run \
  --output build/ppu-validation-results/run-summary.json
```

本机使用 `/root/flagos22-torch/bin/python` 作为已有FlagTree解释器；环境名称不代表本项目依赖Torch，执行过程中Torch被阻止导入。

未测：完整index Build/Add/serialization耗时及峰值RSS、真实数据原始向量recall、生产并发/多卡、所有dim/k/filter组合、冷缓存/冷AOT、端到端Milvus。JSON里的RSS是**整个基准进程生命周期**高水位，包含CPU oracle与解码数组，不是建索引峰值；device bytes为输入存储字节数计算，不是显存峰值实测。
