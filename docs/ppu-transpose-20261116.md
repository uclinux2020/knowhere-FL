# 下一阶段：原生事件定位与转置布局优化

## 本轮结论

在同一Triton（结合TLE）→FlagTree→原生HG→PPU路径新增显式`--transpose-base`配置：每个Search请求先用Triton转置base为维度优先布局，再进行IEEE FP32矩阵距离计算。**计时包含转置、额外设备分配及释放，不缓存打包结果。** 百万向量查询请求由21.434ms降至14.388ms，约1.49x；逆序复测相对历史CUDA-C约3.62x，尚未达到端到端10x目标。

| nb / nq200 | 原布局query请求 p50 ms | 转置配置 p50 ms | 改善 |
|---|---:|---:|---:|
| 100k | 2.540 | 1.732 | 1.47x |
| 500k | 10.828 | 7.182 | 1.51x |
| 1M | 21.434 | 14.388 | 1.49x |

逆序百万复测：本轮14.415ms，历史CUDA-C52.174ms，约3.62x，全部2000个结果槽ID/score核对通过。没有声称服务RPC全链路性能。

## 测量驱动的选择

新增`KNOWHERE_FL_PPU_PROFILE=1`原生HG事件计时（只有开启时创建事件及输出，不把profile模式成绩当正式性能样本）：

- 原布局百万/nq200：score约17.40–17.44ms，select约3.43ms。说明距离阶段是主要热点，选择器并非唯一瓶颈。
- 转置block32：score约8.85–8.89ms，select约3.44ms。布局改善主要在score阶段。
- 事件区间不含转置、分配或host D2H；正式query请求时间包含这些成本，不将事件耗时相加当端到端。

转置布局使矩阵B沿candidate维连续，减少跨stride读取。block64的正式请求约14.4ms优于block32约15.0ms；block128退化至57.0ms，未采用。原布局block16退化至28.95ms，流水num_stages1约21.43ms无明显收益，未保留该调优开关。

原始计时、设备事件输出及退化实验见 [ppu-transpose-results-20261116.json](ppu-transpose-results-20261116.json)。

## 资源和生命周期边界

- 转置base仍是请求私有Buffer，每次重新打包，不按device pointer缓存内容，输入改变不使用陈旧数据。
- **额外内存约nb*dim*4字节**：百万128维为512MB，位于64MiB score/compact预算之外。源码检查尺寸溢出，并对packed base设置独立1024MiB硬上限；超过明确失败，不能称总工作区仅64MiB。
- 索引原始base仍保留；这是性能与额外显存/打包开销的取舍，不默认启用。
- 此次query请求边界包含query allocation/H2D/free、转置与packing allocation/free、score/top-k、sync及结果D2H；原始index驻留，不含初始index H2D、冷AOT、RPC/服务调度。
- 没有显存峰值采样；512MB是存储字节数计算。未接入全局FlMemoryPool总预算，不承诺并发请求总显存上限。

## 实现及验证

`transpose_base.py`是纯Triton转置算子，`matrix_distance.py`按编译常量选择转置或原布局地址计算；runtime检测对应真实kernel并在请求内执行。老manifest无转置kernel时保持原路径。

原生事件创建、记录和elapsed API直接来自HG SDK；每query batch连续记录score/selection边界，结束sync后读取计时。事件生命周期受RAII管理，禁用profile没有事件创建和日志开销。

真机L2/IP、ties、尾query、默认stream、大公共偏置、IVF/PQ、更新codebook、assign回归通过，5维旧bundle CTest通过，全部基准ID及分数错误数为0。Python语法和`git diff --check`通过。PyTorch和CUDA-C没有加入生产路径；FlagOS不等同FlagGems，本次仍为自实现算子编译链。

环境：ZW810E device0、HGGC13.0、GNU13.3、FlagTree PPU Triton3.6，CPU0–7/8线程，seed42、128维/k10/L2/nq200，2次warmup、15次计时，基准串行运行。基线为`8481ed83`及原有b32/w2/select2048/matrix/implicit产物，本轮transposed使用b64、其余相同。

## 复现

```bash
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK FLAGTREE_BACKEND=ppu
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
python python/fl_kernels/compile_ppu.py --tle --matrix --implicit-ids --transpose-base \
  --num-warps 2 --queries-per-block 16 --block 64 --select-block 2048 --dim 128 --k 10 \
  --output build/ppu-transposed/kernels/d128_k10_l2_m0_s0
export KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-transposed/kernels"
export KNOWHERE_PPU_QUERY_E2E=1
build/ppu-contract/knowhere_ppu_bench /tmp/transpose-request.json \
  bf 1000000 200 128 10 L2 1 random 15 8 0
# Profile separately; these instrumented timings are not acceptance timings.
KNOWHERE_FL_PPU_PROFILE=1 build/ppu-contract/knowhere_ppu_bench /tmp/profile.json \
  bf 1000000 200 128 10 L2 1 random 3 8 0
```

完整Knowhere/Milvus仍受既有Folly ABI问题阻塞；未测生产多卡并发、真实数据near-tie召回、全部shape、冷cache、建索引/Add峰值及服务E2E。下一步根据事件定位继续优化矩阵距离与选择物化，而不是把已测3.62x表述成10x。代码及分析未提交推送。
