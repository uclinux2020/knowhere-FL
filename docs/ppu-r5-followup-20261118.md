# R5 后续优化与复测指令（2026-11-18）

## 1. 本轮范围与结果

基于 `/tmp/milvus-ppu-e2e-验证报告-r5-20261117.md`，基线为 knowhere `6725671b`。
本轮没有重启 Milvus、替换在线库或删除集合；没有提交或推送。第一阶段仅做主机微基准，继续阶段补做了独立 native PPU 与 IVF 索引级真机配对，见第4节；尚未执行新的 Milvus RPC 验收。

改动：

- `FlIvfFlatIndexNode::BuildCandidateLists` 的路由线程数按 query 数、OpenMP 上限和工作量决定，当前最多32个。第一阶段最多8个，真机发现100k/88线程场景回退，后调整为32并复测；下表保留第一阶段8线程候选结果。旧实现超过8,388,608次标量乘积即启动整个OpenMP team，R5的nlist1024/d128/nq100会触发。
- 每个 worker 复用一个 centroid score buffer，probe IDs 改成连续数组，消除每 query 的 score/probe 多次分配。
- 保留原浮点累加顺序、质心平分时低 ID 优先规则、partial_sort、候选顺序和过滤语义。未改变 nprobe、训练算法、设备常驻、bundle、CPU fallback 策略或序列化格式。
- 扩展 CSR 回归测试：L2/IP，dim1/128，1/2/8/88 线程，串行/并行路由与候选拷贝边界，过滤和失败回退。
- `milvus_nprobe_sweep.py` 使用独立 query RNG（seed43），跨 NB 保持相同 queries；保存 base/query SHA256、p50/p95/min/max，将 create-index completion、insert+flush、load-ready 分开命名计时。

### 主机专用微基准，不是设备/E2E成绩

使用 `tests/ppu/bench_fl_ivf_host.cc`：synthetic 均匀倒排表，dim128/nq100，scan callback 不计算距离。
包含路由、CSR、Host adapter query staging 与结果分配，不包含真实训练、设备 kernel 或 Milvus RPC。
每进程预热2次、计时21次，顺序 baseline/candidate/candidate/baseline，表格取同版两个进程 p50 的均值。
未固定 CPU affinity，未测并发 segment 下的线程池竞争；仅是本机优化方向证据。

| OMP 上限 | NB | nlist | nprobe | 基线 ms | 候选 ms | 基线/候选 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 100k | 128 | 4 | 0.865 | 0.257 | 3.37 |
| 8 | 1M | 128 | 1 | 1.095 | 0.520 | 2.10 |
| 8 | 1M | 128 | 4 | 0.933 | 0.507 | 1.84 |
| 8 | 1M | 128 | 16 | 3.608 | 2.081 | 1.73 |
| 1 | 1M | 128 | 16 | 11.133 | 11.300 | 0.985 |
| 88 | 1M | 128 | 4 | 1.109 | 0.441 | 2.52 |
| 88 | 1M | 1024 | 16 | 8.872 | 0.861 | 10.30 |
| 8 | 1M | 1024 | 16 | 1.085 | 1.023 | 1.06 |

单线程点约慢1.5%，不能称所有配置都加速。R5 nlist1024 的8线程点仅约6%收益；不要将88线程微基准的10.30倍套用到11.44ms RPC上。
初次配对也显示相同方向，但样本存在变动，不能宣称统计显著的生产收益。

产物位于 `build/ppu-r5-followup/`：`build_host_bench.py`、`run_abba.py`、`host_abba.json`、构建日志、`fl-tests-final.log`。
完整 Knowhere/UT 构建成功；全部 `[fl]` **27用例、4440断言通过**。测试使用 Host adapter，不能作为 PPU 执行证明。
Python `py_compile`、脚本 `--help` 和 `git diff --check` 通过；未运行全仓库 Catch2、生产并发、建设峰值 RSS/显存或真机候选 E2E。

## 2. R5 必须修正的测试与结论

### 2.1 不得把 RPC 减去 kernel 小区间后的全部时间归为路由

R5 的真实 kernel 执行证据成立，但“90%+在质心路由/候选收集”尚未证明。
`score_ms/select_ms` 是设备事件；candidate H2D/D2H 是 host wall-clock 包围同步复制。
还需计入 allocation、launch/wait、bundle 首次加载、同步/析构/free、query staging、segment 调度、merge/reduce、RPC 和客户端开销。
`launch_wait_ms` 包含 score/select，不可重复相加。
1M 的实际请求跨5个 segment，不能把一条 segment kernel 时延与整个集合 RPC 相减；也不能简单求和当并发 critical path。

**指令**：证据轮保存 `FL_IVF_SCAN` 的 route/materialize/query_stage/scan/elapsed 与 runtime 日志，按请求时间窗、node/base_ptr/device/nb/nq/nlist/nprobe 关联；另采 server search/reduce 和客户端 wall-clock。
若同窗口多 segment 重叠不能唯一关联，应添加 request/segment trace，而不是强行归因。
新日志只可解释新实验，不能倒灌解释 R5。

### 2.2 profiling 与正式性能轮分离

R5 启动脚本硬编码 `KNOWHERE_FL_PPU_PROFILE=1`。
当前实现按环境变量是否存在启用：**`=0`、`=false`、空字符串仍开启**。

**指令**：一轮 profile-on 作执行证据，正式计时轮在**服务进程启动环境**执行 `env -u KNOWHERE_FL_PPU_PROFILE`；只在 Python client unset 无效。
固定同 binary/library、bundle、workspace、CPU affinity/线程、query/mask、segment布局，并做 on/off/off/on 配对，量化 instrumentation 开销。
不要直接调用既有 `/root/start_milvus.sh`，其 profile 硬编码且会停止服务并覆写旧日志。
重启或另起实例应先取得授权，保留旧证据；推荐独立端口、etcd/minio/localStorage/rocksmq 命名空间。

### 2.3 执行证据按阶段与 segment 计数

对现存 `/root/milvus_run/milvus_stdout.log` 的只读审查：228条 `PPU_PROFILE` 中156条属于 kmeans/Add assign，36条 BF、36条 IVF search。
搜索72条是 segment级调用，包含预热；不是228次计时Search，也不是72个RPC。
精确 FL vector staging 匹配得到24条，不是报告14条；泛搜 staged 会混入 Milvus 自身日志。

**指令**：分训练/添加、预热、计时请求统计，并关联到 segment 与请求窗口；按 severity 统计 ERROR，不能泛搜字符串 error。
无 fallback + staging/注册日志不充分，仍需 kernel 执行或等效硬件 trace。
禁止用 R5a 的同配置复现单独证明 R1–R4 每个历史请求均回退；历史结论需各轮原始配置与日志。

### 2.4 IVF 必须做同质量比较

现存 E2E 脚本两个 IVF 均为 **nlist1024/nprobe16、L2/k10、无过滤、nq100**；应补入R5表格。
100k CPU recall0.176 与 FL0.171 差0.5个百分点，1.58倍仅是同参数、略低recall的观测，不是无损同质量加速。
CPU/FL 使用不同训练算法，同 nlist/nprobe 不代表相同候选或相同训练工作量。

**指令**：先固定nlist1024，扫nprobe4/16/64/256/1024，同时报告recall和延迟；预先声明质量目标（例如0.90/0.95），按达到相同目标的点比较。
若不能达到目标，明确失败，不以低召回最快点替代。全探针检查recall1与score误差。
另外选nlist256/4096代表点；记录各segment实际nlist，特别注意CPU小segment训练可能调整nlist。
过滤/大k/IP/near-tie/不足k/全过滤和并发需专项，不能将当前“双1.0”泛化为这些场景已验证。

### 2.5 build_s 和吞吐必须改名

现存 `create_index` 默认同步等待完成，并非仅异步提交，但包含RPC、排队与约0.5秒粒度状态轮询；load/staging在计时外。

**指令**：称 `create_index_complete_s`，不称纯Train/Add构建时间；同时报告insert_flush、load_ready、first-search与server逐segment构建阶段。
建库峰值RSS按构建阶段采样，不能用搜索后的process-lifetime高水位。
`batch_ms/nq`是摊销时间，不是单query RTT；`nq/batch_seconds`是concurrency1批吞吐，不是饱和服务QPS。
至少30次原始样本，报告p50/p95与变动；固定数据、保存每次shape/唯一ID/score，oracle检查放计时外。

## 3. 可执行复测步骤

以下操作由测试执行者在已授权的隔离实例执行；本轮没有执行这些服务操作。

### 3.1 基线/候选构建与验收

在仓库根目录，使用与Milvus依赖ABI一致的构建环境。当前本机UT环境可复用：

```bash
cmake --build build/full-validation/Release -j8
LD_LIBRARY_PATH="$PWD/build/full-validation/Release:/root/milvus/internal/core/output/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}" \
  OMP_NUM_THREADS=8 build/full-validation/Release/tests/ut/knowhere_tests '[fl]'
python -I -m py_compile tests/ppu/milvus_nprobe_sweep.py
```

不要将UT专用库直接覆盖在线Milvus库。记录实际加载libknowhere/可执行文件SHA256、编译选项、依赖和设备驱动版本。
基线6725671b与候选使用独立构建输出，相同Go/core executable、配置、bundle及集合，分别reload，ABBA配对。

### 3.2 bundle 必须按正确 specialization/root 部署

R5 L2/d128/k10搜索、k1 assign 默认TLE bundle本身正确。BF可选matrix/implicit是另一种同一Triton/TLE真机路径的实现配置，不是第二设备路径。
每个variant使用独立root；runtime按context/path缓存，禁止运行中原地覆盖旧bundle期待热切换。

默认root保留。创建全新matrix-implicit root（执行前确认目标未存在）：

```bash
export PPU_SDK=/usr/local/PPU_SDK FLAGTREE_BACKEND=ppu
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
ROOT="$PWD/build/r5-matrix-implicit-$(date +%s%N)"
/root/flagos22-venv/bin/python python/fl_kernels/compile_ppu.py \
  --dim 128 --k 10 --tle --matrix --implicit-ids --num-warps 2 \
  --queries-per-block 16 --block 32 --select-block 2048 \
  --output "$ROOT/d128_k10_l2_m0_s0"
/root/flagos22-venv/bin/python python/fl_kernels/compile_ppu.py \
  --dim 128 --k 1 --tle --output "$ROOT/d128_k1_l2_m0_s0"
```

transpose variant另建root，k10使用 `--matrix --implicit-ids --transpose-base --num-warps 2 --queries-per-block 16 --block 64 --select-block 2048`，也补k1。
transpose要求matrix；不能只给默认编译命令附加transpose。
修改k/metric时额外编译对应specialization，IP不能复用L2。
记录manifest/hgbin SHA256、matrix/implicit/transpose和tile参数，服务启动设置root后检查实际manifest。
文档约21.4ms是nq200的native query-request边界，不能与R5 nq100的Milvus RPC直接比较。

### 3.3 服务启动环境（需要事先授权）

保留原配置的必要MILVUSCONF、依赖库路径和独立storage配置。不是可直接覆盖当前实例的重启脚本。

```bash
# Proof process: add KNOWHERE_FL_PPU_PROFILE=1 to its launch environment.
# Acceptance process: completely remove the variable from the server environment.
env -u KNOWHERE_FL_PPU_PROFILE \
  KNOWHERE_FL_PPU_KERNEL_DIR="$ROOT" KNOWHERE_FL_PPU_WORKSPACE_MB=64 \
  /root/milvus/bin/milvus run standalone
```

日志输出另存新文件，不截断R5日志；验证服务PID环境而不是client环境。
真实执行失败时将该次性能验收判FAIL；不必把生产fallback默认策略擅自改成FATAL。
运行前确认设备/主机无其他benchmark占用，固定CPU affinity、OpenMP/Milvus线程池与device routing，多segment并发不能只写“88核同机”。

### 3.4 使用修正后的RPC脚本

脚本 `--prepare` 只创建新的隔离集合，拒绝覆盖；之后同prefix reload不重建，适合跨revision固定artifact比较。
创建后记录各索引segment布局；若不同索引布局差别明显，不能归为纯index速度比较。监控compaction/重建，变动则废弃该配对。
**query seed现改为43，不能与旧脚本seed42的query结果直接配对；两版必须都用本次脚本。**

```bash
mkdir -p build/ppu-r5-rpc-results
for nb in 100000 1000000; do
  python -I tests/ppu/milvus_nprobe_sweep.py \
    --port 19540 --nb "$nb" --nq 100 --nlist 1024 \
    --probes 4 16 64 256 1024 --repeats 30 \
    --prefix r5audit_20261118 --revision candidate-proof --prepare \
    --output "build/ppu-r5-rpc-results/proof-$nb.json"
done
```

在授权切换为profile-off基线或候选服务后，使用同命令但**去掉 `--prepare`**、修改 `--revision` 与新输出文件名。
按 baseline-A/candidate-A/candidate-B/baseline-B 顺序分别执行两规模，所有benchmark串行运行。
每轮保存完整服务日志、实际环境、manifest、样本和query hash。profile轮只作证据，正式速度使用off轮。
脚本覆盖L2/d128/k10无过滤；其他工作负载仍须补测。

## 4. 继续阶段：真实 PPU 验证与修正（2026-11-18）

### 4.1 BF bundle 对比

重建独立 `knowhere_ppu_bench`，设备0，CPU affinity0–7、8线程、workspace64MiB，nq100/d128/L2/k10。
两种现有bundle使用同一runtime二进制：R5默认direct/TLE root与 `build/ppu-implicit/final/kernels` 的matrix+implicit root。
每次新进程2次预热，profile-off每版两次30样本，ABBA；profile-on证明轮各5条事件（2预热+3样本），没有CPU fallback。
计时为resident-base native query-request（query alloc/H2D/free、runtime alloc/launch/sync/D2H/free），不是Milvus RPC。

| NB | direct 两轮p50均值 ms | matrix+implicit 两轮p50均值 ms | direct/matrix |
|---:|---:|---:|---:|
| 100k | 5.981 | 1.334 | 4.48 |
| 1M | 61.117 | 11.592 | 5.27 |

全部最终结果ordered-ID/score mismatches=0；随后改bench逐个timed sample在计时外检查，1M/matrix的30个样本全部通过，p50=11.605ms、p95=11.646ms。
默认direct单卡1M值不能与R5多segment/多卡RPC的23.25ms横比。这里只说明相同native边界下bundle选择收益。
产物 `build/ppu-r5-followup/device/` 包含JSON原始样本、proof日志、manifest/hgbin与binary SHA256。
此ABBA不是profile开销实验（proof样本数不同且非on/off/off/on），不报告profiling开销百分比。

### 4.2 IVF 含路由与真实扫描的索引级配对

扩展 `tests/ppu/bench_fl_ivf_host.cc`：默认仍为HOST_ONLY，`KNOWHERE_IVF_BENCH_NATIVE=1` 切换真实native adapter。
失败立即终止，检查每次实际native callback执行、候选CSR oracle的有序IDs与L2分数；oracle仅首次非计时预热构建，每样本验证在计时外。
fixture为随机质心/queries、全零base和均匀倒排表；全零base用于可验证的平分低ID规则，**不是代表真实数据recall的质量测试**。
边界含host route/CSR/query staging与实际PPU scan，不含Train、Milvus RPC或多segment merge。

第一次8线程上限候选虽减轻1M/88线程的极端尾延迟，却使100k/88线程profile-off从约0.597ms回退到1.438ms。
控制 `OMP_WAIT_POLICY=PASSIVE` 后仍从0.814ms回退到1.152ms，故不能保留“最多8线程普遍更优”结论。
将路由上限改为32，候选copy上限仍为8；最终配对固定PASSIVE、OMP_PROC_BIND=true、OMP_PLACES=cores、CPU affinity0–87，ABBA各进程30样本：

| NB | OMP上限 | 基线两轮p50均值 ms | cap32候选 ms | 基线/候选 |
|---:|---:|---:|---:|---:|
| 100k | 8 | 1.380 | 1.241 | 1.11 |
| 100k | 88 | 0.830 | 0.786 | 1.06 |
| 1M | 8 | 3.561 | 3.214 | 1.11 |
| 1M | 88 | 7.217 | 3.850 | 1.87 |

8线程点不同轮有系统漂移，前次8线程候选曾略回退；小幅收益不能当统计显著结论。
32线程上限消除了此fixture里原8线程上限的100k明显回退，但仍待真实多segment并发与不同query数测试。
所有成功的native运行均无fallback，proof日志 `device_done=1`。产物在 `native-ivf/`、`native-recheck/`、`native-cap32-final/`。
一次cap32复测失败exit127，发生在完整库重链期间；构建结束后改新输出目录复测成功，失败不得当性能样本。
未证明具体缺失symbol原因，也不将该失败归为设备失败。

完整库重建与 `[fl]` 再次通过：27用例、4440断言；格式与diff检查通过。
在线Milvus仍保留，未重启、未换库、未删除集合。其他卡有现存大内存任务，测试只用空闲device0；未来严格验收仍须隔离host背景负载。

### 4.3 当前执行建议

1. BF优先把matrix+implicit独立root纳入隔离Milvus的profile-on证据轮，再profile-off复测，不能直接替换运行中缓存bundle。
2. IVF保留当前cap32候选进行固定artifact RPC ABBA；不要再用cap8微基准结果作为当前版本成绩。
3. 服务环境明确设置并记录OMP等待策略与亲和，默认等待行为可能放大88线程team的尾延迟；PASSIVE效果同样需按真实生产并发验收。
4. 下一步操作涉及新Milvus实例/存储命名空间与集合写入，须先确认实例隔离范围；现存实例不用于覆盖或删除测试。

## 5. 隔离 RPC 验收：发现探针截断错误

已获得明确授权：临时19540实例、全新etcd/MinIO前缀，创建100k/1M集合，结束仅停止本次临时进程并保留数据，不操作原19530实例。
/tmp盘仅余约300MiB，因此实际测试配置、localStorage/rocksmq、JSON与日志使用 `/root/knowhere-r5-rpc-<timestamp>/` 独立新目录。

第一轮 `/root/knowhere-r5-rpc-1794964075244960623/` 在100k全探针精确性检查失败，未进入正式ABBA。
实际日志显示 nlist1024，但请求nprobe256/512/768/1024均变成128，recall停在0.519；本轮全探针不精确不是PPU kernel错误，而是 `FlIvfFlatConfig::CheckAndAdjust` 把搜索参数与train-only的nlist默认128比较并截断。
搜索配置并不携带真实训练nlist，节点Search本已有 `min(request_nprobe, nlist_)`，因此移除配置层错误截断，保留节点按实际nlist限制。

新增 `[fl][ivf-nprobe]` 配置回归测试，完整库构建成功，28用例/4442断言通过。
两版RPC对照库共享此正确性修复；只替换IVF_FLAT translation unit的节点头对比6725671b路由与当前cap32路由，故基线应称“6725671b路由+探针修复”，不能称未修改的6725671b完整revision。
首次构建因未设置PPU_SDK_PATH失败，设置 `/usr/local/PPU_SDK` 后成功，未更改ABI或Folly检查。

复测固定matrix+implicit bundle、OMP上限88、PASSIVE等待、CPU affinity0–87、GOMAXPROCS8，nprobe4/16/64/256/512/768/1024，质量目标0.90/0.95。
proof-on与正式off轮分离，ABBA只使用off；逐样本shape/唯一ID/ID域/原始向量L2分数检查位于计时外，跨revision要求segment/artifact与有序IDs不变。
驱动 `build/ppu-r5-followup/run_rpc_abba.py`，结果路径指针 `rpc-output-path.txt`；原服务保留运行。

修复后证明轮已完成，产物 `/root/knowhere-r5-rpc-1794964331277474673/`：
- 100k FL_IVF_FLAT recall@10在nprobe4/16/64/256/512/768/1024分别为0.065/0.149/0.365/0.724/0.912/0.988/1.0。
- 1M对应0.083/0.178/0.391/0.749/0.922/0.988/1.0；CPU IVF两规模全探针也均为1.0。
- 100k每索引1个segment；1M每索引5个segment，不同索引集合segment行数布局并不完全相同，CPU/PPU比较须限定为该服务级布局观测。
- 新增索引级nlist256、nprobe128/256/512回归夹具，检查256以上探针按真实list数量限幅且不丢失最后一个list中的精确邻居。
正式ABBA已完成并通过跨版hash、segment、有序IDs、分数核验，完整结果见 [RPC结果报告](ppu-r5-rpc-results-20261118.md)。
最终回归为29用例/4453断言通过。临时实例均停止，原19530保留，测试数据和中断证据保留。

1M/nprobe16路由候选两轮p50约11.9ms，基线53.4/106.1ms；nprobe512为332.1→86.2ms、recall0.922，nprobe768为502.8→123.4ms、recall0.988。
基线波动及候选尾延迟明显，倍数仅为本次固定工作负载观测，不泛化生产并发。
监控发现每轮HTTP metrics启动冲突：必须在临时进程环境设置 `METRICS_PORT=9092`；YAML common.MetricsPort不足以改变HTTP默认9091。
驱动已修正供下轮使用，本次报告保留真实ERROR并说明监控隔离不足。

## 6. 后续优先级

1. 先按上文完成同artifact、profile-off、matched-recall 的真机RPC验收，确认路由线程限制收益及segment并发影响。
2. 用host/runtime/server时序确认真实critical path，再决定继续优化质心打分、CSR上传或分配/同步；不要从小kernel事件直接推断。
3. BF对比default与matrix-implicit，再单独评估transpose；packing/free也计入请求wall时间。
4. 建库优化需同时报告build wall/RSS、最终存储与recall，不将轮询量化后的create-index倍数当纯计算收益。
