# R6 后续优化：分块质心路由与高探针选择（2026-11-18）

> 更新：本文保留第一阶段组合候选的历史实验。随后已完成隔离RPC和消融，
> 当前工作区撤回partition/sort，只保留分块SIMD；最终判定见
> [RPC结果](ppu-r6-rpc-results-20261118.md)及[消融结果](ppu-r6-ablation-20261118.md)。

## 1. 范围与实现

输入报告：`/tmp/milvus-ppu-e2e-验证报告-r6-20261118.md`。基线为
`6b01a2bf48cf4e13ccef92bb1a38147a62e4ac00`；候选为该基线加本轮工作区修改。
本轮没有提交/推送，没有重启或替换在线 Milvus，没有创建/删除集合或调整设备状态。

- IVF_FLAT Search 新增 16 质心一组的 dimension-major 派生缓存。
  内层 SIMD 作用于独立质心的点积，dimension 累加顺序不变；不引入跨维 reduction、fast-math 或 BLAS。
  nlist<16 或 dim<16 保留标量打分。非整块 list 进行 padding，输出仅取有效 lanes。
- 缓存在 Train 完成和成功 Deserialize 后构建，只读用于 Search，不改变 Train/Add 的 assignment。
  不进入序列化，`Size()` 纳入新增缓存字节。
- nprobe<=floor(nlist/8) 保留 partial_sort；更大的前缀使用 nth_element+sort。
  全探针使用 sort（不对 end 调用 nth_element）。保持分数/低 list ID 的全序及有序 CSR。
  原有无过滤全探针设备 BF shortcut、32 路由线程上限、8 拷贝线程上限保持不变。
- 扩展 CSR 回归：L2/IP、dim1/17/128、nlist64/65、nprobe1/8/9/40/42/63、1/2/8/88线程、过滤与扫描失败回退。
  独立标量 oracle 检查每个 CSR ID 的顺序，覆盖重复分数、非整块质心及 dispatch 边界。
- `bench_fl_ivf_host.cc` 在第一次非计时预热检查完整 CSR 与独立标量质心路由的一致性。
  原生模式继续逐样本检查有序结果 ID 和分数。CLI 增加可选 nq/dim：
  `nb nprobe repeats [nlist [nq [dim]]]`，旧调用保持兼容。

缓存内存公式：`ceil(nlist/16)*16*dim*4` 字节。nlist1024/d128 每索引增加 **512 KiB 主机内存**，
不增加设备常驻或序列化字节。这是布局计算，不是峰值 RSS 实测；未测本轮 Train/Add wall-time、构建峰值 RSS。
Train/Deserialize 多一个 O(nlist*dim) 转置步骤，构建性能待验证。

## 2. 实验环境与可复现边界

所有产物放在 `/root/knowhere-r6-opt-20261118/`，避免 `/tmp` 仅余约117MiB的问题。
复用 `build/full-validation/Release/compile_commands.json` 中 test_fl_index 的编译参数与已有真实 Folly CRC 库。
`build_bench.py` 用 HEAD 节点头生成独立 baseline include override，candidate 使用工作区节点头。
两版复用相同 libknowhere/adapter/runtime；**比较的是内联 IVF_FLAT 节点实现，不是两份不同完整 Milvus revision**。

环境：本机 88 核 x86、GCC13、Release -O3、-msse4.2；CPU affinity0–87；
`OMP_WAIT_POLICY=PASSIVE, OMP_PROC_BIND=true, OMP_PLACES=cores`；OMP上限1/8/88按表设置。
记录 `cpu.json`、编译日志、binary SHA256。设备为 native HG PPU adapter，device0，workspace64MiB，
bundle `/root/milvus_run/fl_kernels_tl`，`UMD_PLATFORM_TYPE=1, HGGC_DRIVER_CANDIDATE=UMD`。
没有修改 bundle/kernel/runtime。

fixture：随机质心和 query，seed42；均匀倒排表、全零 base；L2/d128/nq100/k10、无过滤。
全零 base 用于检查平分时低ID规则，**不是实际数据召回/质量测试**。
每进程2次预热；性能轮profile-off、每版两个进程各31样本，串行ABBA；表格取每版两个进程p50均值。
独立profile-on证明轮各3样本（另含2预热），不混入正式性能样本。
不报告单query RTT、饱和QPS或新的Milvus RPC加速。

当前在线Milvus仍在运行，其他卡有现存大显存任务。device0只读采样无活跃计算后运行独立测试，
但没有停止背景服务或获得全机独占。ABBA有系统漂移；小幅变化不能认定统计显著。

复现本次配对的保留脚本：

```bash
TMPDIR=/root/gotmp python -I /root/knowhere-r6-opt-20261118/build_bench.py
python -I /root/knowhere-r6-opt-20261118/run_host.py
python -I /root/knowhere-r6-opt-20261118/run_native.py
python -I /root/knowhere-r6-opt-20261118/run_shapes.py
python -I /root/knowhere-r6-opt-20261118/analyze.py
```

脚本原始输出目录已经存在；重跑 native 脚本应改用**新的**目录，不能覆写本轮证据。
正式部署/RPC应另建隔离环境并确认端口、存储与服务切换范围。

## 3. 真实 PPU 索引级结果

包括 host route/CSR、query alloc/H2D/free、runtime scan 和结果分配；不含训练、Milvus RPC/多segment merge。

| NB | OMP上限 | nprobe | 基线p50均值 ms | 候选p50均值 ms | 延迟降低 |
|---:|---:|---:|---:|---:|---:|
| 100k | 8 | 16 | 1.142 | 0.625 | 45.2% |
| 100k | 8 | 512 | 9.185 | 8.338 | 9.2% |
| 1M | 8 | 16 | 3.323 | 2.645 | 20.4% |
| 1M | 8 | 512 | 75.394 | 72.362 | 4.0% |
| 1M | 88 | 512 | 77.385 | 76.035 | 1.7% |

- 100k/p16 两轮候选 p50=0.629/0.622ms，p95=0.642/0.681ms；基线p50=1.140/1.144ms。
- 1M/p16 两轮候选 p50=2.670/2.619ms，p95=2.718/2.664ms；基线p50=3.497/3.150ms，已有漂移。
- 1M/p512/8线程基线p50=78.383/72.405ms，候选72.758/71.967ms；4%只作观测，不夸大显著性。
- 1M/p512/88线程候选第二轮p95=81.209ms，高于基线两轮79.828/80.094ms；**尾延迟未一致改善**。
- 证明轮 **50条PPU_PROFILE、50条device_done=1、CPU fallback=0**。
  这些是含预热的节点/设备事件，不是50个Milvus RPC；所有运行通过CSR标量oracle和逐样本结果核验。

证明轮用于相位定位：100k/p16路由中位数1.082→0.397ms；1M/p16为0.894→0.305ms。
1M/p512/8线程候选 route≈0.711ms、materialize≈8.739ms、scan≈61.487ms。
这说明高探针下本夹具扫描/传输和CSR成本更大；**不能用本夹具分解倒推R6多segment RPC的瓶颈比例**。
phase数据来自profile-on，不与profile-off总时间直接相减。

证据路径：`native-blocked/summary.json`、每运行的stdout/log、`analysis.json`、
`native-blocked-abba.log`、`blocked-binary-sha256.txt`。最终CLI扩展仅改变benchmark参数解析，
新增 shapes 测试使用 `build-final-bench.log` 编译出的二进制；上表二进制指纹保留在blocked文件中。

## 4. 主机专用测量与边界

Scan callback不计算距离，包含host route/CSR和Host adapter query staging；不是PPU/E2E结果。

| NB | OMP上限 | nlist/nprobe | 基线 ms | 候选 ms |
|---:|---:|---:|---:|---:|
| 100k | 1 | 1024/16 | 5.852 | 1.951 |
| 100k | 1 | 1024/128 | 8.611 | 4.637 |
| 100k | 1 | 1024/129 | 8.568 | 3.399 |
| 100k | 1 | 1024/512 | 16.112 | 8.909 |
| 100k | 8 | 1024/16 | 0.979 | 0.400 |
| 1M | 8 | 1024/16 | 1.146 | 0.503 |
| 1M | 8 | 1024/512 | 10.084 | 12.045 |
| 1M | 88 | 1024/512 | 13.492 | 14.150 |

高候选量主机wall出现回退，不能声称所有配置加速；分配/内存带宽漂移仍需专项隔离。
保留 `host_blocked_abba.json` 的完整ABBA样本，不以设备层收益掩盖该结果。

补测100k/8线程：nq1/d128/nlist1024/p16=0.0588→0.0211ms，nq8=0.1320→0.0524ms；
nq100/d16/nlist128/p16=0.3890→0.3573ms；d17/nlist17/p8=1.0370→0.9368ms；
d128/nlist4096/p16=3.6066→1.0472ms；d128/nlist128/p4=0.2616→0.1789ms。
对应 `shapes.json`，全部通过独立CSR oracle，不含设备scan。

最初单独尝试partition+sort版本：主机单线程高探针有收益，但1M/p512真机8线程72.675→73.766ms，
不能单独作为明显设备收益；保留 `host_abba.json`、`native-partition/`，后续增加分块打分形成上表最终候选。

## 5. R6 报告需修正的事实

1. “E2E 90%+ 全是路由”证据不足。设备score/select事件不含CSR H2D、分配、launch/wait、D2H/free、
   query staging、segment调度/merge和RPC。现有日志应按节点/请求并发时间窗关联，不能简单相减或相加。
2. 普通 IVF Search **不只上传query**，还按批次上传host CSR候选和offsets。
   仅BF或无过滤全探针implicit-ID shortcut免候选上传。
3. 当前源码已有质心tie-break和train-only默认nlist截断修复，并有逐探针单调性回归。
   上轮真实RPC报告 `ppu-r5-followup-20261118.md` 第5节记录全探针recall1.0和单调probe sweep。
   因此不能直接把R6“已知BUG仍待修”当成当前仍未修复的事实；若仍有非单调样本，需要实际nprobe日志、
   固定artifact/query、原始IDs/分数复现。本轮不声称用零base基准完成真实recall验收。
4. SDK显存查询工具实际可用，不在SDK/bin而在：

```bash
LD_LIBRARY_PATH=/usr/local/PPU_SDK/ppu-smi/lib:/usr/local/PPU_SDK/lib \
  /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi -q -d MEMORY,UTILIZATION,PIDS
LD_LIBRARY_PATH=/usr/local/PPU_SDK/ppu-smi/lib:/usr/local/PPU_SDK/lib \
  /usr/local/PPU_SDK/ppu-smi/bin/ppu-smi \
  --query-ppu=index,memory.total,memory.used,memory.free,utilization.ppu --format=csv
```

只读实测总显存97920MiB/卡；测试前device0 used229MiB，查询时device0 used949MiB。
这是**全卡快照**，包含现存进程/上下文，不等于本索引独占显存、增量驻留或峰值。
保留 `ppu-before.txt`、`ppu-query.csv`；未reset、改clock/power或管理设备。

## 6. 检查与下一步

- 完整native PPU配置Knowhere/UT增量构建成功。
- `[fl]` 全部 **29用例、7493断言通过**（Host adapter/CPU正确性测试，不能代替真实设备证明）。
- clang-format-18检查、git diff --check通过；初次格式检查发现一处换行，修正后通过。
- pre-commit未安装，未运行其完整hook；未运行全仓库Catch2、ASAN、生产并发或新Milvus RPC。

优先级：

1. 在授权的隔离Milvus实例做同artifact/query/segment布局、profile-off ABBA与matched-recall验收，
   保留当前nprobe16点并扫0.90/0.95质量目标。新增请求关联字段以拆解多segmentcritical path。
2. 高探针优先验证CSR分配/上传与runtime回收：runtime已是每请求单arena，不是每数组独立分配；
   仍每请求malloc/free。跨请求复用需每context/device有界lease池、inflight独占及故障回收，不能单共享buffer加锁。
3. 用ppu-smi按进程和相位采样主机RSS/设备显存，记录新增缓存构建时间、峰值、并发scratch容量。
4. 在SDK异步拷贝/default-stream语义得到验证后，才考虑显式stream/双缓冲；不能凭CUDA类比删除同步。

本轮保留的优化提升了低探针真实索引级性能，没有证据证明R6的7.62/10.92ms Milvus RPC会获得相同幅度。
