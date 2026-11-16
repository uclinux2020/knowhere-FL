# PPU Triton/TLE 性能差距分析与优化（2026-11-16）

## 结论

差距确实不是“换成Triton就必然慢几十倍”，而是第一版实现的调度和资源使用不合理。已保留旧基准并修改代码，统一路径仍是 **Triton（结合TLE）→ FlagTree → 原生HG → PPU**，没有使用CUDA-C实现加速算子、没有引入PyTorch。

核心BF在nq16时改善54–80倍；k100改善165倍。所有14项回归及128维contract tests通过，ordered ID/score mismatch均为0。尚未追平历史CUDA-C在nq200的吞吐，不能宣称完成全部优化。

## 实测原因与证据

1. **主机串行启动及同步是首要问题。** 旧版对每个query的1024候选chunk逐一启动score/select/merge，还穿插同步D2D。1M、nq16按源码有46880次launch。改为二维query×candidate grid及设备分层top-k后，旧kernel不变，100k从93.275ms降至3.399ms，1M从886.161ms降至31.875ms。这是有效的调度消融，不是推测语言性能。
2. **对大base tile强制TLE异步加载造成资源膨胀。** manifest记录distance的shared从66048字节降到2048字节：小query向量仍使用TLE异步加载，大base tile使用常规Triton load。nq200/100k同条件从39.806ms降到13.068ms。说明不能将“全部异步加载”机械等同更快；occupancy影响是推断，shared资源和耗时变化是真实测量。
3. **query间复用不足。** 新增batched_distance，每个program复用同一base tile处理4条query，尾query被mask。nq200/100k进一步降到12.276ms；避免重复读取base，仍保持FP32直接距离和相同排序语义。
4. **融合不是自动更快。** 试验distance+local-topk融合：最初shared达131584字节，nq16/100k耗时53.073ms；低shared版本nq200/100k仍耗85.573ms，远慢于默认12.276ms。该配置只保留`--fused`显式实验开关，不默认启用；没有用失败实验替代已经验证的快配置。
5. **先前历史数据并非同条件。** 旧报告CUDA-C是nq200，第一版Triton报告是nq16；CPU基线实现、线程、测量边界也不同。历史t7仅核对前20 query，本轮同条件比较核对全部200条query。不能直接用两份报告的CPU加速倍数判定Triton编译器效率。

## 代码改动

- `fl_ppu_runtime.cc`：二维query/candidate启动，批量分层top-k在stream内连续执行，无chunk级host同步/D2D，最终一次批量同步及D2H；保留旧manifest兼容。
- `retrieval_triton.py`：大base tile不用强制async promotion；query仍结合TLE。
- `batched_distance.py`：4-query共享base tile，支持L2/IP、尾query mask。
- `compile_ppu.py`：生成batched_distance、登记QPB，fusion改为可选实验；检查manifest相关参数。
- `bench_ppu_runtime.cc`：历史CUDA-C产物仅通过环境变量作为**比较用基线**；主库不编译链接该算子，不作为fallback。加载的是本机已有`/root/topk.cubin`，没有新增CUDA-C生产算子。

score工作区现在随query batch×候选width增长，不再是旧版单chunk容量；默认以64MiB的score/compact工作区目标控制batch。`KNOWHERE_FL_PPU_WORKSPACE_MB`允许1–1024MiB，预算连单query都不能容纳时明确失败。candidate CSR额外上传，不计入该预算；工作区仍未接入FlMemoryPool全局限额。新旧内存策略不能混淆。

## 前后同一基准比较

dim128，nq16（assign nq64、single query例外），CPU0–7/8线程，seed42，预热2次，7次计时；base/query已驻留设备，含workspace分配释放、候选H2D、结果D2H。正确性oracle不在计时区。基线为之前保存的同工作负载原始JSON。不是端到端Milvus测量。

| 工作负载 | 旧PPU p50 ms | 新PPU p50 ms | 改善倍数 |
|---|---:|---:|---:|
| BF 10k k10 | 9.663 | 0.196 | 49.36 |
| BF 100k k10 | 93.275 | 1.170 | 79.72 |
| BF 1M k10 | 886.161 | 16.206 | 54.68 |
| BF 100k k1 | 76.558 | 1.137 | 67.35 |
| BF 100k k100 | 268.629 | 1.630 | 164.80 |
| BF 100k IP | 95.305 | 1.498 | 63.64 |
| IVF 50% candidates | 52.924 | 1.189 | 44.51 |
| IVF random 1% candidates | 1.068 | 0.080 | 13.42 |
| PQ 10k m16/ksub256 | 8.771 | 0.453 | 19.36 |
| Assign 64→256 | 3.310 | 0.099 | 33.36 |

仍不适合PPU的边界：不足k的5候选、很小的centroid assignment、某些1% clustered candidates；CPU只需微秒，PPU启动成本仍明显。空候选直接返回host padding，不把空结果吞吐当有效加速。

完整14项、CPU对照和每次样本见 [优化原始JSON](ppu-optimization-results-20261116.json)。相对本次独立8线程CPU对照，新BF100k为4.91x、1M为6.45x；不是历史Knowhere CPU baseline的倍数。PQ对照为解码表示直接距离，并非优化CPU LUT-ADC。

## 与历史CUDA-C产物的真实同条件对照

同程序、同数据seed、dim128/k10/L2、nq200、device0、相同host结果读取边界，全部2000个结果ID核对、分数核对通过。legacy经原生HG直接加载，仅作比较。

| nb | 历史CUDA-C p50 ms | 优化Triton/TLE默认64MiB p50 ms | Triton/CUDA-C耗时比 |
|---|---:|---:|---:|
| 100k | 6.697 | 12.276 | 1.83 |
| 1M | 52.321 | 180.725 | 3.45 |

扩大1M工作区预算：256MiB为127.015ms，1024MiB为115.886ms，仍慢于52.321ms。不将1GiB配置偷偷设置为默认，也不隐瞒显存取舍。原始样本见 [同条件对照JSON](ppu-matched-comparison-20261116.json)。

这些nq200数据说明历史CUDA-C成绩可复现，先前几十倍差距主要来自初版实现。剩余差距确实存在：Triton方案仍物化score/ID并做多级全局选择；legacy每block处理2条query且在寄存器/共享存储维护局部top-k，不物化整个距离表。后续需要专门优化选择算法、PQ LUT及持久query tile，而非仅加TLE标签。

## 验证与限制

- 全部14项功能/性能回归、L2/IP、empty/fewer-than-k、ties、非法CSR、PQ/assign通过。
- 新query tile存在padding时（原矩阵nq1、batch边界、nq200的分批末尾）正确。
- 无Torch import，Python语法及`git diff --check`通过。
- 原生C++独立测试已重新编译。完整Knowhere仍受缓存milvus-common/Folly F14 ABI不一致阻塞，未宣称端到端通过。
- 没有GPU profiler逐指令/阶段测量；没有多卡并发、峰值workspace/RSS实测、真实数据召回、冷AOT/冷cache评估。共享资源变化来自manifest，launch计数来自源码。
- 基准host输入H2D不包含在Search时间。1M初始上传有明显成本，不能拿resident Search成绩承诺一次性查询的端到端延迟。

复现：按`docs/ppu-validation-20261116.md`运行`validate_ppu.py`，新编译bundle自动含低shared+batched_distance。对照用`KNOWHERE_PPU_LEGACY_CUBIN=/root/topk.cubin`执行`knowhere_ppu_bench ... bf ... 128 10 L2 ...`，切勿将该变量用于生产加速路径。代码及报告尚未提交推送。
