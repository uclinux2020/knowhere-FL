# 距离计算与score物化优化：精确局部top-1

## 本轮结果

新增显式 `--compact-score`：在Triton IEEE FP32矩阵距离tile内完成精确局部top-1，只向全局内存输出每tile的最优score/ID，之后设备分层选择得到全局top-1。统一路径仍为Triton（结合TLE）→FlagTree→原生HG→PPU，无PyTorch和CUDA-C生产算子。

**有效范围是k=1，编译工具与runtime均明确限制，不替代k=10路径。** k=10融合实测退化，不能将top-1收益宣传为历史CUDA-C k10对照收益，更没有达到端到端10x目标。

| nb / nq200 / k1 | 旧请求 p50 ms | 局部压缩 p50 ms | 改善 |
|---|---:|---:|---:|
| 100k | 1.3698 | 1.0190 | 1.34x |
| 500k | 5.7070 | 4.2286 | 1.35x |
| 1M | 11.5071 | 8.2525 | 1.39x |

逆序配对使用相同dim128/L2/seed42、CPU0–7/8线程、device0，预热2次、15次请求计时；默认64MiB score预算。计时包含query allocation/H2D/free、每请求base转置及额外分配释放、局部选择、后续选择、同步、result D2H；原始索引驻留。所有ID/score错误数为0，same-representation recall=1。基线e7ac3d37，不修改CPU oracle。

## 减少的物化成本

百万向量QPB16/BLOCK64：旧每query完整score表有1M float，约4MB；新只存15625个tile最优score/ID，单份约187500字节。原runtime两份compact切片仍分配，约375000字节/query；完整score表不分配。64MiB预算可一次处理176条query，而旧k1完整score布局只能处理约16条，减少重复batch启动和D2H。

base转置仍需额外512MB，计时和存储取舍沿用原路径，独立1024MiB打包上限；没有把总额外显存表述为只有64MiB。字节数为布局计算，不是峰值采样。无跨请求结果或打包缓存。

HG事件（instrumented，仅用于定位）显示新百万top1距离加局部选择约5.74ms，后续选择约0.035ms；事件区间排除打包、分配和host传输，不将它称为8.25ms请求的完整分阶段分解。

## 实现与正确性

- `matrix_compact.py`复用原IEEE矩阵距离公式及大公共偏置平移处理，在row维并行求最优值，并列score按全局ID升序。
- runtime跳过完整score/ID表，按tile结果容量计算query batch，对压缩后的真实ID使用显式选择器（不得使用位置ID恢复）。
- 编译要求transpose-base、k1；runtime拒绝没有transpose kernel或k非1的compact bundle。默认编译不启用，旧manifest保持兼容。
- Conan导出新增算子源文件。
- L2/IP、default stream、ties、tail、非法CSR、empty/fewer-than-k、PQ更新codebook、assign、大偏置L2真机contract通过。k10旧路径另行回归通过。

## 未采用的k10实验

相同百万/nq200/k10：BLOCK64约16.43ms（旧转置约14.4ms），BLOCK128约32.51ms，BLOCK32约35.17ms；QPB32约26.59ms。score物化虽减少，逐rank归约与融合编译资源成本抵消收益。因此不将k10方案上线，最终仅保留限制为k1的配置。

全部样本及事件输出见[原始JSON](ppu-compact-top1-results-20261116.json)，包含失败性能实验而非只保留最优值。

## 复现

```bash
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK FLAGTREE_BACKEND=ppu
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
python python/fl_kernels/compile_ppu.py --tle --matrix --implicit-ids --transpose-base \
  --compact-score --num-warps 2 --queries-per-block 16 --block 64 --select-block 2048 \
  --dim 128 --k 1 --output build/ppu-top1/kernels/d128_k1_l2_m0_s0
export KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-top1/kernels"
KNOWHERE_PPU_QUERY_E2E=1 build/ppu-contract/knowhere_ppu_bench /tmp/top1.json \
  bf 1000000 200 128 1 L2 1 random 15 8 0
```

本轮独立C++编译、Python语法、git diff --check通过；完整Knowhere/Milvus仍受既有Folly ABI问题阻塞。未测生产服务全链路、多卡并发、所有维度、原始真实数据near-tie、冷cache、显存峰值。报告只比较相同k1新旧PPU请求，不用历史k10 CUDA-C成绩做不公平对照。代码及分析尚未提交推送。
