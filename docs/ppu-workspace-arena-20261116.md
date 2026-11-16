# 下一阶段：请求内单块工作区与选择器消融

## 本轮结果

在统一的Triton（结合TLE）→FlagTree→原生HG→PPU执行链内，将Search原有9次独立设备工作区分配合并为**一次请求内arena分配**，按256字节对齐划分候选、offset、LUT、score、compact score/ID切片。请求结束仍释放，不缓存数据或结果，不改变默认预算和算子精度。

减少HG allocator调用及析构中的重复stream synchronize，主要改善小请求；大规模计算收益接近噪声，**未达到CUDA-C端到端10x目标**。

| 工作负载 | 原请求 p50 μs | arena p50 μs | 改善 |
|---|---:|---:|---:|
| BF nb1000/nq1/k10 | 61.68 | 52.09 | 1.18x |
| 同项独立复测 | 61.01 | 48.10 | 1.27x |
| BF nb10000/nq1/k10 | 85.35 | 75.44 | 1.13x |
| Assign 64→256 | 106.63 | 93.86 | 1.14x |
| PQ 10k候选/nq16 | 211.38 | 197.50 | 1.07x |
| BF nb10000/nq200 | 389.45 | 370.84 | 1.05x |
| BF nb100000/nq200 | 2565.34 | 2536.84 | 1.01x |
| BF nb1M/nq200 | 21493.33 | 21470.60 | 约1.00x |

小请求31次样本、大请求15次，均预热2次，seed42、128维L2、device0、CPU0–7/8线程，工作区64MiB，query-request E2E包含query分配/H2D、Search、结果D2H及释放，索引驻留。使用相同已编译kernel产物比较原binary与新binary，算子源码不变。PQ数据为量化表示下的正确性对照。全项ordered ID/score mismatch为0；完整样本见[JSON](ppu-workspace-arena-results-20261116.json)。

## 未采用的选择器与tile实验

- 联合score/ID tuple归约：百万查询请求21.941ms，没有优于原约21.4ms。
- 首层min/max返回位置归约：21.683ms，无明确收益，撤销。
- 单warp：25.250ms退化，撤销编译选项扩展。
- 32-query tile：64MiB预算39.172ms；128MiB预算22.791ms；256MiB预算30.461ms。没有优于16-query/64MiB配置，未改变默认参数。

上述失败或退化实验的原始样本同样保存；实验算子和开关已清理，避免未验证的复杂度进入实际路径。

## 实现与资源预算

`fl_ppu_runtime.cc::Search`计算9个切片尺寸并检查累加/对齐溢出，使用单个Buffer承担生命周期。候选CSR及offset空间仍与原实现一样额外计入实际分配；score/compact/LUT的预算逻辑不变。每片按256字节对齐会有少量额外填充，故64MiB仍为主要工作区预算目标而非包含所有元数据的精确显存峰值。当前未接入FlMemoryPool全局限额，不引入跨请求长期保留缓冲。

异常展开时由唯一Buffer先同步对应stream再释放，避免已提交kernel访问释放工作区。新增`test_ppu_runtime`的`default-stream`模式，验证nullptr stream路径。

## 验证及复现

- 独立原生C++重新编译通过。
- 128维matrix/implicit组合在默认流完整contract通过；LUT bundle在显式流contract通过，包括codebook更新；5维旧bundle CTest通过。
- `git diff --check`通过，未引入PyTorch或CUDA-C生产算子。
- 基线代码`dbcfe134`，本轮只改C++请求内工作区布局及测试流选项，kernel产物复用上一阶段配置。

```bash
cmake --build build/ppu-contract -j2
# Use a complete bundle set matching d128/k10 plus PQ/assign specializations.
KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-implicit/final/kernels" \
  build/ppu-contract/knowhere_ppu_test 0 128 10 16 default-stream
KNOWHERE_PPU_QUERY_E2E=1 KNOWHERE_FL_PPU_WORKSPACE_MB=64 \
KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-implicit/final/kernels" \
  taskset -c 0-7 build/ppu-contract/knowhere_ppu_bench /tmp/request.json \
  bf 1000 1 128 10 L2 1 random 31 8 0
```

设备HGGC13.0/ZW810E、GNU13.3、既有SDK环境变量沿用前轮；其他卡有既存作业，本轮仅使用device0，基准串行运行。完整Knowhere/Milvus仍受既有Folly ABI问题阻塞，不宣称服务E2E已验证。未做并发、多卡、峰值显存采样、实际服务延迟、冷cache或所有shape验证。

下一优先级仍是设备计算与选择阶段的分阶段计时和降低score物化；这次小请求管理开销改善不能替代大规模10x目标。代码及报告未提交推送。
