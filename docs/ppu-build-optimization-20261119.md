# PPU 构建链路优化与未开发模块复核（2026-11-19）

## 1. 范围与结论

依据 `/tmp/milvus-ppu-GPU加速未开发模块分析-20261119.md`，以当前 knowhere `eaeb5947` 为基线核验源码并实施两项构建优化。

- **不沿用“IVF 设备侧收益接近零”与“scatter 是 1M 构建慢的直接原因”**。前者未覆盖高召回、大 nprobe 与过滤负载；后者没有逐阶段构建计时支持。
- `kmeans_assign` 已在 native PPU 接线；`graph_search` 仍为空。PQ coarse/subspace Train 与 Add 编码目前仍走 CPU，不能将适配器支持等同于 PQ 节点已经调用设备训练。
- 当前已存在全探针、无过滤精确 BF 快路径、SIMD 质心路由和可选有界 workspace 缓存，不应再次当成待开发功能。
- 本轮优先优化真实 PPU 路径周围的串行计算和重复传输，不新增 SQ/图算法/segcore 接口，保持索引格式与检索算法不变。
- 没有重启或换库在线 Milvus、没有创建/删除集合、没有提交或推送。本轮测量不是新 Milvus RPC/flagfit E2E 成绩。

原分析文档未修改；源码核验勘误与新证据保存在本文件。

## 2. 源码核验与优先级调整

| 模块 | 当前事实 | 下一步 |
|---|---|---|
| IVF kmeans assign | `fl_ppu_adapter.cc` 指向 `ppu::Assign`，设备 top-1 返回 host IDs；CPU 更新质心 | 先优化更新与传输，再按 profile 决定设备归约 |
| IVF kmeans update | 原先每轮串行扫描 `nb × dim`、double 累加 | 本轮稳定分组并行更新；不改变每个维度的累加顺序 |
| IVF_FLAT Add | 原先为 assign 上传 xb、释放，再为常驻搜索上传 xb | 本轮复用最终 resident xb，只上传临时 centroids |
| PQ Train/编码 | `LloydKmeansCpu` 与 `NearestIn`；未复用设备 assign | 后续独立实验，先测子空间 pack/H2D/launch/D2H 成本；需对应 dsub/k1 bundle |
| PQ 查询 | native ADC 接线，已有可选 LUT specialization | 不能将适配器接线当作生产 PQ 请求设备执行证据 |
| SQ | 无 FL SQ 节点 | 新功能，需节点/配置/格式/白名单与 recall 验证，不是当前 IVF_FLAT 加速捷径 |
| 倒排表 scatter | host 按 assign 顺序 push_back；设备仍按原始 ID 访问 resident xb | 先测 Add 分组占比；若布局仍在 host/需要序列化，设备 scatter+D2H 可能得不偿失 |
| 标量过滤 | segcore 表达式求值在 Knowhere 外 | 收集过滤 critical-path 占比后再推进；bitset 与候选压缩端到端联合评估 |
| CAGRA 图构建/遍历 | CPU 构图，PPU `graph_search=nullptr` | 设备遍历本身也是缺口；批量建图需算法与召回专项，不只解除规模限制 |
| 加载/序列化 | host 格式与 IO；部分可能计算密集 | 按阶段测量，不能概括成“纯 IO 无可加速部分” |
| 多卡 | router 分配索引节点，不等同请求内跨卡分片检索 | 不将轮询称为已完成跨卡检索/归并；是否改善延迟取决于分片和通信 |

原文需要修正的另外两点：PQ **编码**是子向量与码本的 nearest-codeword distance/argmin，并非只做 gather+sum；将 kmeans assign 接上并不自动实现 SQ 编码。过滤排除90%并不保证距离计算或 RPC 延迟下降10倍，必须确认 kernel 是否实际跳过、压缩开销和其他阶段占比。

## 3. 实现

### 3.1 稳定分组质心更新

`src/index/fl/fl_centroid_update.h` 提供共享 `AccumulateCentroids`，供 IVF_FLAT 与 CPU Lloyd/PQ 使用：

1. 小工作量、单线程、嵌套 OpenMP 或 dim<16 保留串行路径。
2. 大工作量对 owner 做计数/前缀和，稳定分组 row IDs。
3. 每个 cluster/dimension tile 独立 double 归约，最多8线程；同一维度严格沿原始行顺序累加，无浮点原子、无跨线程 partial sums。
4. 空簇 reseed、随机种子、迭代上限、assign 与收敛判定不变。

初次32维 tile 导致重复读向量、偏斜场景明显退化，已撤回；最终128维 tile 的偏斜性能接近持平。不能称所有分布都更快。

新增临时 host 空间估算：`8*nb + 16*(nlist+1)` bytes（row IDs、offsets/cursor），不包括原有 owner、sums/counts 或输入。1M 约8MB，不是实测峰值 RSS。

### 3.2 Add 复用 resident xb

先 `TryStageToDevice()`，再调用 assign；`AssignAll` 绑定 resident buffer 所属 `device_id_`，不重新轮询获取另一张卡。临时质心通过 `TryReserve` + RAII 管理；stage、分配、复制或 assign 失败仍执行原 CPU nearest-centroid 路径。

正常设备路径全量 xb H2D 次数 **2→1**。1M/d128 减少512,000,000 bytes，100k减少51,200,000 bytes。只是移除重复上传与临时分配；不是把向量移出 host，也不宣称最终常驻显存下降。

旧临时 xb 在最终 resident xb 分配前已释放，所以不能将此次改动称为“消除两个完整 xb 同时驻卡”的峰值显存收益。

## 4. 验证与性能证据

### 4.1 完整库与回归

- `PPU_SDK_PATH=/usr/local/PPU_SDK cmake --build build/full-validation/Release -j4` 成功。
- 全部 `[fl]`：**35用例、7728断言通过**。包括新增 `[centroid-update]` 和 `[add-resident]`。
- 质心更新覆盖1/2/8/88线程、dim129尾 tile、空簇、均匀/单簇偏斜，double sums 与独立串行 oracle **逐位一致**。
- Add 回归覆盖正常设备 callback 与注入 assign 失败：全量上传1次、结果一致、临时与最终 pool accounting 释放正确。此类 UT 使用 Host adapter，不是设备性能证明。
- 独立微基准在无 OpenMP 构建下通过。
- clang-format-18 dry-run、`git diff --check`、Python语法检查通过；本机没有 pre-commit，未执行 hooks，也未运行全仓库 Catch2。

### 4.2 CPU 更新阶段 ABBA（不是 PPU/RPC 成绩）

`tests/ppu/bench_fl_centroid_update.cc` + `run_centroid_update_abba.py`：seed42、相同输入/owner，每进程2次预热+30次计时，serial/grouped/grouped/serial 顺序，CPU affinity0–7、8线程、PASSIVE、PROC_BIND/PLACES 固定。oracle 在计时外，计时包含分组、分配与归约。

每版取两个进程 p50 的均值：

| NB | nlist | owner分布 | 串行 ms | 分组 ms | 串行/分组 |
|---:|---:|---|---:|---:|---:|
| 100k | 128 | 均匀 | 4.815 | 2.607 | 1.85 |
| 100k | 128 | 90%同簇 | 3.824 | 3.787 | 1.01 |
| 1M | 128 | 均匀 | 68.341 | 35.729 | 1.91 |
| 1M | 128 | 90%同簇 | 65.657 | 66.716 | 0.984 |
| 1M | 1024 | 均匀 | 67.840 | 36.656 | 1.85 |
| 1M | 1024 | 90%同簇 | 65.547 | 67.457 | 0.972 |

偏斜百万点有1.6%–2.8%退化，保留并披露；不代表完整训练退化同样比例。尚未验证低维PQ子空间、生产多segment并发、NUMA扩展或全部分支阈值的性能。

### 4.3 真实 PPU Add 配对

真实 native HG/FlagTree Triton-TLE 路径，物理设备1（设备0/6/7有现存进程，未操作），GNU C++13.3、CPU0–7/OMP8/PASSIVE，workspace默认64MiB。两版同依赖与k1 bundle，基线用 HEAD 的 IVF_FLAT 节点头覆盖，候选用当前节点；共享其余完整库。因此是“eaeb5947节点/候选节点”对照，不是两版完整Milvus revision。

每规模baseline/candidate/candidate/baseline，每block15个全新进程，共120个Add样本。每进程先做相同4096行CPU训练（不计时），Add计时含host payload copy、native staging/assign及invlists；Serialize在计时外验证全artifact FNV1a64与size，两版完全一致。callback失败立即中止，禁止CPU fallback冒充加速。proof-on独立100k日志有PPU_PROFILE设备事件，正式性能轮完全unset profiling。

| NB | 基线两block p50均值 ms | 候选 ms | 基线/候选 | xb上传次数 |
|---:|---:|---:|---:|---|
| 100k | 49.977 | 44.686 | 1.118 | 2→1 |
| 1M | 486.292 | 439.476 | 1.107 | 2→1 |

**不能称稳定尾延迟改善**：合并每版30个fresh-process样本的离散p95（index28）分别为100k基线52.748ms、候选787.083ms；1M基线1258.82ms、候选1195.90ms。存在明显驱动/主机背景波动，p50收益仅代表本轮限定观测，不证明统计显著性或服务生产收益。

Add结束、Search/Serialize之前采进程高水位，中位数（KiB）：100k基线227270、候选227300；1M基线1143280、候选1143296。包含输入与此前CPU训练，不是Add阶段单独峰值RSS；未显示主机内存下降，也未采设备构建峰值。final serialized bytes/hash一致，pool used与最终resident bytes匹配且析构后归零。

第一轮驱动因结果记录重复传入`nb`抛TypeError，退出1；失败日志保留为`add-abba.log`，修复后新文件`add-abba-v2.log`完整重跑，失败样本不混入统计。

计时边界为 Add 的 host payload copy、设备 staging、nearest-centroid assign、倒排列表分组；Train、Serialize 与 Milvus RPC 均在外。

## 5. 下一轮 E2E 验收

1. 先在隔离实例中测现有matrix+implicit bundle、修正nprobe与profile-off效果；不把早期direct bundle/低recall数据作为当前上限。
2. 查询代表点：100k/1M，nq1/100，k10/100，nprobe16/64/512/全探针，至少一个高选择性过滤点。质量目标先声明0.90/0.95，同目标比较，保留不足k/全过滤正确性检查。
3. 构建分开记录 Train assign/update、Add staging/assign/group、Serialize、create-index completion 与 load-ready。构建期RSS采样与设备实测分开；不以搜索后进程高水位替代构建峰值。
4. 固定artifact、segment布局、CPU线程/亲和、物理设备 allowlist和后台compaction。profile-on证明轮与profile-off性能轮分离；ABBA每点至少30样本，记录p50/p95/原始结果与hash。
5. 本轮尚未实施新RPC测量；启动隔离实例、写集合或替换在线库须另行确认。现有 `/tmp` 剩余空间很小，新增构建/基准产物放 `/root` 独立目录，不清理他人文件。

## 6. 继续阶段：完整索引构建与隔离RPC验收

### 6.1 Train + Add 真机配对

扩展 `bench_fl_ivf_add.cc` 的 `build` 模式：全部输入参与 native PPU 训练，计时包含完整 Train + Add；Serialize单独计时/验证，输入生成在外。两版同kernel、依赖、seed、OMP8、CPU0–7、物理卡1；baseline/candidate/candidate/baseline，每block3个新进程，每版每规模6样本，属于方向验证，不足以支持尾延迟统计。

下表为每版全部6样本中位数（不与上文“两block p50均值”混称）：

| NB | 阶段 | 基线 ms | 候选 ms |
|---:|---|---:|---:|
| 100k | Train | 787.305 | 721.954 |
| 100k | Add | 45.736 | 40.781 |
| 100k | Train+Add | 832.794 | 762.778 |
| 1M | Train | 8228.630 | 7369.075 |
| 1M | Add | 477.268 | 427.583 |
| 1M | Train+Add | 8705.895 | 7795.975 |

两规模两版全部25次train assign，serialized artifact bytes/hash相同。1M设备assign host-wall包围时间中位数6337.845/6338.050ms，约占原Train的77%；该callback包括runtime launch/wait/D2H，不能等同纯kernel时间。下一步优先改善native assignment或其bundle，而不是假定scatter主导构建。

### 6.2 已授权的隔离RPC运行

2026-11-19用户明确选择允许隔离验收：临时RPC19540、空闲metrics9093、物理卡1；共享etcd/MinIO但仅新前缀/新100k和1M集合，产物/root独立目录，结束只停止本次临时PID并保留数据，不改在线实例。

在线预检为flagfit-RAG Milvus PID2434541、RPC19620、metrics9092、device0；它不是历史19530实例。新驱动不使用已有 `start_milvus.sh`，不覆写历史RPC路径指针、库或数据。

配对库仅替换IVF_FLAT TU（HEAD eaeb5947节点与本轮候选节点），其余为同一Milvus-ABI对象快照；保存shared-object SHA256，不能称两版全仓库新构建。固定matrix+implicit bundle、workspace64MiB/cache0、CPU0–7/OMP8、GOMAXPROCS8、关闭自动compaction，profile-on证据轮与正式off ABBA分离。该轮检索代码不变，主要用于验证构建优化不会改变reload查询结果，不预期直接改善Search耗时。

运行驱动 `/root/knowhere-ppu-build-opt-20261119/run_rpc_v2.py`；结果路径写入同目录 `current-rpc-v2-path.txt`，完成后补充实际结果。100k/1M、nq100、L2/k10/nlist1024，探针16/64/512/全探针，正式每点30次，保留全部样本/ordered IDs/原始向量分数校验与segment布局。

首次proof中断：最初只重编IVF_FLAT TU，复用的BF/router/runtime对象仍为11月18日版本，早于物理白名单修复。此轮不作为验收成绩；仅精确停止通过`/proc/PID/environ`确认本次MILVUSCONF的临时PID，保留 `/root/knowhere-buildopt-rpc-1795098152549434625/`，未删除集合。刷新整个Milvus-ABI构建后在全新`rpc-libraries-v2`和存储前缀重跑，并核查每轮实际PID环境与loaded-library路径。设备快照里的native child PID不能直接当作Milvus主PID，未对任何在线PID发停止信号。

### 6.3 2026-11-20 恢复验收

上次会话结束后，v2驱动未完成：`/root/knowhere-buildopt-rpc-1795098511130053742/` 中100k proof完整，1M JSON仅含CPU FLAT一项；没有正式ABBA结果。恢复时核查只剩在线Milvus PID2434541，无本次临时实例或驱动，物理卡1无计算进程。不将部分JSON称为验收通过。

为避免覆盖不完整集合/日志或混合跨日样本，重新创建全新命名空间执行完整proof及off-ABBA；仍遵循已确认的端口、物理卡、存储及在线保护范围。驱动 `/root/knowhere-ppu-build-opt-20261119/run_rpc_20261120.py`，指针 `current-rpc-20261120-path.txt`，日志 `rpc-20261120.log`。每个client工作负载设1800秒超时，退出路径只停止本次驱动创建的server PID，保留全部数据。新旧实验不混合计算统计。

当前完整新实验目录 `/root/knowhere-buildopt-rpc-1795135671970639304/`：两规模proof及baseline-A/candidate-A完成，候选B进行中。A轮input hash、segment、有序ID及逐query recall一致；proof有306条PPU_PROFILE、无CPU fallback提及、无` [ERROR] `级日志，物理白名单日志确认为1。尚未完成全部ABBA，不能报告最终性能验收通过。

Bundle核验：同root的k10是matrix-ieee-fp32/QPB16/block32/implicit IDs，k1仍为direct-fp32/QPB4/block128/implicit IDs。因此“使用matrix root”不代表训练assign已用矩阵kernel；后续k1矩阵实验需新bundle root，不能覆盖运行中已缓存产物，也不能与本次RPC并行竞争资源。

### 6.4 完整RPC结果（2026-11-20）

新实验proof + baseline-A/candidate-A/candidate-B/baseline-B全部完成，正式20个工作负载×4轮×30样本，共2400个计时RPC。完整分析校验输入hash、工作负载矩阵、proof与正式轮segment布局、有序ID、逐query recall及每个样本分数，全部通过。两规模全探针recall=1；所有正式轮profile完全关闭，无fallback提及。

以下每版为两轮p50的均值，单位ms/batch，nq100：

| NB | 路径 | nprobe | recall@10 | 基线 | 候选 |
|---:|---|---:|---:|---:|---:|
| 100k | FL BF | — | 1.000 | 5.715 | 5.571 |
| 100k | FL IVF | 16 | 0.149 | 4.874 | 4.940 |
| 100k | FL IVF | 64 | 0.365 | 5.988 | 5.975 |
| 100k | FL IVF | 512 | 0.912 | 11.644 | 11.588 |
| 100k | FL IVF | 1024 | 1.000 | 5.529 | 5.543 |
| 1M | FL BF | — | 1.000 | 17.535 | 17.833 |
| 1M | FL IVF | 16 | 0.200 | 10.514 | 11.045 |
| 1M | FL IVF | 64 | 0.415 | 17.044 | 17.256 |
| 1M | FL IVF | 512 | 0.927 | 73.196 | 73.795 |
| 1M | FL IVF | 1024 | 1.000 | 17.580 | 17.765 |

**结论：本轮构建优化未带来Search RPC加速。** 多数点接近持平；1M/nprobe16候选约慢5.1%，如实保留，不声称查询无性能回归。BF与未修改CPU控制项亦有波动，不能将全部变化归为本轮构建代码；未做统计显著性或生产并发验收。候选100k/nprobe64两轮p95为23.965/21.732ms，基线6.988/7.104ms；1M全探针基线B p95达46.882ms，尾延迟问题两版均存在。

全探针精确BF快路径显著快于高nprobe CSR：候选1M 17.765ms、recall1，对比nprobe512 73.795ms、recall0.927。这是既有快路径的结果，不归因于本轮改动。若业务目标≥0.90/0.95，应评估直接精确路径，而不是以低召回nprobe16的11ms作为验收。

proof有306条runtime events/120条FL_IVF_SCAN；所有轮server顶层ERROR=0、fallback提及=0。baseline-A有28条嵌于INFO中的Storage-AWS `[ERROR]`，内容包括不存在的DMI文件等环境探测，单独记录，不冒充服务ERROR也不隐去。原始p95、CPU控制项及全部样本见 `paired-summary.json` 与各轮JSON。

本轮使用固定candidate建立的artifact后跨版reload，**不包含基线/候选重新构建RPC配对**；proof的create-index完成时间不能证明构建加速。Train+Add收益仍以第6.1节独立真机数据为证据。下一步应优先实验k1 assign矩阵bundle，再做构建阶段matched-artifact/recall验证。

本轮临时PID全部退出，在线PID2434541仍运行。集合、存储、旧中断证据保留；未提交/推送。未完成过滤/IP/大k/生产并发、真正flagfit应用全链路或设备构建峰值显存验收。

产物目录：`/root/knowhere-ppu-build-opt-20261119/`，包含构建/测试日志、CPU ABBA原始JSON、设备proof、Add与完整Build两版构建脚本和二进制指纹。完整RPC产物 `/root/knowhere-buildopt-rpc-1795135671970639304/`。测试工具RTK在本机不可用，使用原生命令；不调整用户工具安装。
