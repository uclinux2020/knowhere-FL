# PPU 未开发模块与 FlagFit 端到端优化续查（2026-11-20）

## 1. 基线与范围

以 Knowhere `593792ba` 和 `/tmp/milvus-ppu-GPU加速未开发模块分析-20261119.md` 为起点。结合已有 [构建优化](ppu-build-optimization-20261119.md)、[k1实验](ppu-k1-matrix-20261120.md) 与本机 FlagFit 当前实现核实，不重做已完成的功能、不把微基准当 RAG E2E 成绩。

初轮不操作在线 Milvus/FlagFit、不创建/删除集合，只进行源码修改和独立 native PPU 测试，设备1。继续阶段用户明确选择允许隔离RPC验收：19540/metrics9093、物理卡1、独立存储前缀与新集合，结束仅停本次PID、保留数据。仍不修改19620在线服务/既有集合、部署配置或质量阈值，不提交/推送。0/6/7有现存进程。产物放 `/root/knowhere-ivf-tile-20261120/`、`/root/knowhere-ivf-rpc-20261120/`，不清理历史文件。

## 2. 实际应用链路改变了优先级

只读核查 `/tmp/flagfit` 的实际入口：

- `tools/bench-rag/run_index_arms.py`：embedding → Milvus REST search → rerank；单条向量 `data:[vector]`，1024维归一化、IP、GPU_IVF_FLAT/nprobe1024、owner过滤，默认召回20再精排取5。C8/C32是独立请求并发，不是nq8/32。
- `tools/bench-rag/recipes/run_paired_rag.py`：上述检索之后组装证据并调用生成；真实请求JSONL已有 embed/search/rerank、assemble、generation/TTFT及整体E2E计时。
- 实际服务配置 `.../validation/home/.flagfit/vector_store.yaml` 为GPU_IVF_FLAT/nlist1024/nprobe1024。REST direct adapter绕过Rust Store，不能把Store的verify-index额外RPC计时套到该基准上。
- 当前进程身份不是旧文档的PID2434541；只读核查 Milvus PID3511159、FlagFit PID3930017。设备报告PID不可直接视为宿主服务PID，未发信号。
- 归档业务bundle `.../validation/kernels/d1024_k20_ip_m0_s0/manifest.json`：direct、block128、select1024、warps4、TLE。不是128维L2 matrix/QPB16配置。

FlagFit已有报告 `docs/analysis/20261119-ppu-milvus-rag-tp2.md` 的1800条固定输出预算请求：

| 并发 | CPU平均E2E秒 | PPU平均E2E秒 |
|---:|---:|---:|
| 1 | 6.555 | 6.580 |
| 8 | 15.027 | 15.002 |
| 32 | 45.550 | 45.440 |

这些是**历史描述性数据**，不是本轮测量；90.06%因256-token预算length stop，QA与独立稳态未验，不称生产业务加速。

`docs/analysis/20261120-rag-rerank-candidate-plan.md` 的候选-only试验中，k20平均search18.986ms、rerank362.302ms；k10为10.937/137.421ms，但配对nDCG@5零容限门未通过，禁止据此默认降k。rerank客户端排序约0.02ms，父段耗时集中于HTTP调用（仍包含服务端排队/编码/模型/JSON），不能归因纯模型。该报告也明确不应因为k10质量不足转回低收益IVF kernel投入。

即便18.986ms search完全消失，其对6.555s RAG均值的量级贡献也仅约0.29%。这是**不同实验边界的量级示例**，不是同请求Amdahl实测；正式收益上限需从同一campaign逐请求critical path计算。

## 3. 原清单的修订排期

| 模块 | 现在的事实 | 对在线RAG的价值/下一步 |
|---|---|---|
| IVF kmeans assign | 已真实接线；k1矩阵已有百万create-index约2.18倍方向证据，但质量变动 | 服务构建/导入链优化，不能宣称查询加速；matrix仍opt-in |
| PQ Train/编码 | PQ节点未调用设备assign，确实是可开发空白 | 先确定真实应用使用PQ；当前IVF_FLAT业务不能因此获得延迟收益。dsub/k1 pack/传输/训练质量专项 |
| SQ8/SQ4 | 无FL节点 | 产品能力新增，不是现有RAG快赢；配置/格式/白名单/ADC与召回需整体交付 |
| IVF分组scatter | 当前host列表+原始ID resident xb | 没有证据表明它主导构建；先阶段计时，不为GPU化引入无收益D2H |
| 全探针过滤检索 | 快路径要求bitset.empty；实际请求有owner条件 | 应验证实际bitset/eligible数，再评估全探针单份过滤候选、设备bitset或有界候选复用；不得忽略租户隔离/删除/可见性 |
| 标量表达式→bitset | 在Milvus segcore，非本仓库节点 | 先同请求分解expr/filter/scan；90%排除不保证10倍E2E。不可自行改上层语义 |
| rerank准备/评分 | 在线检索父段中已有较大占比证据 | 当前RAG优先级高于新增PQ/SQ；保持k20与token完整语义，下探排队、重复编码、prepared/batch评分，单独申请跨仓库实施与服务验收 |
| 图构建/遍历 | CPU构图、PPU graph_search空 | 独立算法/产品项目；不直接解除20万限制冒充加速 |

只读短窗记录的device_done=1/full_scan=0不能单凭owner过滤判断所有阻断项；需要逐条件见证。当前全campaign backend证据仍不足。

## 4. 下一步实施与验收边界

1. **Knowhere当前路径**：100k/1M同编译链tile128/32、固定候选的ABBA已复现收益；继续代表维度/k值。先验随机/平分/过滤候选/不足k与budget边界，再按各负载选择tile，不默认推广32；旧产物差异另列。
2. **Milvus/FlagFit Search**：仅在明确授权的隔离实例中，用同artifact/segment/query/owner条件、1024/IP/k20、C1/C8比较；记录实际bitset状态、eligible count、host route/materialize、candidate H2D、score/select与REST wall。profile证明轮与off计分分离。验证租户隔离/删除/可见性，不用无过滤BF数字冒充业务条件。
3. **应用主critical path**：维持召回20/top5与token策略，优先下探rerank HTTP里的服务等待、tokenization/重复准备与评分。prepared输入复用只复用不变数据，不复用结果；batch/concurrency改变须测逐query分数、排序、nDCG/QA、OOM及C8墙钟。需跨仓库实施与服务操作授权，不在本轮自动部署。
4. **构建/产品能力**：PQ设备训练与编码可作为独立后续任务，但先冻结真实PQ负载和oracle（非residual、dsub/k1、code byte范围、同召回与构建RSS）；SQ/图另立功能验收，不捆绑到当前Search优化。
5. 最终按同一完整campaign的逐请求各阶段与E2E检验收益；不以低召回、减少候选、截断回答或未通过质量门的结果换取性能倍数。独立稳态与QA未通过的结果保持描述性，不入生产默认。

## 5. 本轮限定实验：解耦 dense / IVF score tile

`compile_ppu.py` 新增 `--ivf-block`，默认等于既有 `--block`。相同tile继续输出schema1；split-tile必须输出schema2，使旧runtime明确拒绝，避免忽略新字段后launch错误grid而静默漏扫。新runtime兼容schema1（ivf_block默认block），schema2必须提供ivf_block；仅IVF_FLAT/PQ score grid采用ivf_block，BF/k1密集grid不变，fused grid仍由fused_block控制。不会改变质心路由、nprobe、候选/过滤、k、选择器或索引序列化格式。

目的不是声称找到主要E2E瓶颈，而是让两类访存形态可独立调参，避免优化dense matrix tile时同步改变gather scan。业务旧bundle为block128，故保持k20/select1024/warps4/direct/TLE不变，只比较IVF tile128与32。另有1024/IP/k10/warps2/select2048限定消融，不能混称业务基线。

新增：

- `test_ppu_ivf_tile.cc`：显式设备/维度/k/metric，随机/平分、L2另有10000-offset、空/少于k、tile边界与多层选择、nq17、默认/显式流，对double oracle核对有序ID/分数，禁止回退。
- `run_ivf_tile_abba.py`：顺序ABBA，nq1/17、候选10%/50%/100%、random/skewed，每block30样本、2预热；query allocation/H2D/free、CSR H2D、workspace与D2H均计时。CPU oracle在外；CSR是预先生成的候选，不含真实表达式求值/host路由，也不是Milvus RPC或RAG。

实验结果与实际执行检查在完成后补充。当前不默认改tile或部署bundle。

### 初次试验与归因约束

100k/d1024/IP/k20，旧归档bundle对新编译tile32 bundle，四格两block p50均值：nq1/10%为0.532→0.097ms，nq1/100%为2.312→0.397ms，nq17/50%随机16.485→2.404ms，偏斜10.931→1.314ms。所有计时样本有序ID/分数oracle通过，但**这不是严格tile消融**：虽然编译参数保持选择器相同，新编译select/distance/batched_distance的hgbin hash与旧业务归档都不同。可能包含编译链/产物差异，不应将5.49–8.32倍全部归因tile32。

为缩小归因范围，另建 `app-controlled/`：复制旧业务bundle，只替换新ivf_flat.hgbin及其kernel元数据，保留旧select/distance/batched_distance字节，schema2/ivf_block32。使用同一新runtime、顺序ABBA复测；不覆写旧或在线bundle。这样隔离到“新IVF score算子”而不是其他检索算子，但新旧IVF编译链本身仍可能不同，严格的纯tile归因还需当前编译链同时重编tile128/32。

百万规模旧归档→新全bundle补测也完成（同四格/每版60样本）：nq1/10%为2.318→0.393ms，nq1/100%为19.591→3.055ms，nq17/50%随机164.643→24.153ms，偏斜50.083→7.693ms。仍受同样编译产物归因约束，不称纯tile收益。两规模共32进程/960计时样本，seed42、有序ID与score不匹配全部0；不是ANN质量/真实数据/服务并发验收。

### 同编译链与受控算子替换结果

同一当前编译链重编tile128/32，其余三个hgbin的hash完全一致，仅ivf_flat不同；同一runtime、固定输入/query/候选、k20/select1024/warps4/TLE，CPU affinity0–7、OMP8、物理设备1、workspace64MiB/cache0，profile-off，顺序ABBA、每版每格60样本，两个block p50均值如下（ms/request或batch）：

| NB | nq | 候选比例/分布 | tile128 | tile32 | 加速比 |
|---:|---:|---|---:|---:|---:|
| 100k | 1 | 10%随机 | 0.533 | 0.101 | 5.29 |
| 100k | 1 | 100%随机 | 2.316 | 0.399 | 5.80 |
| 100k | 17 | 50%随机 | 16.493 | 2.403 | 6.86 |
| 100k | 17 | 50%偏斜 | 10.919 | 1.315 | 8.30 |
| 1M | 1 | 10%随机 | 2.321 | 0.392 | 5.93 |
| 1M | 1 | 100%随机 | 19.628 | 3.070 | 6.39 |
| 1M | 17 | 50%随机 | 164.604 | 24.114 | 6.83 |
| 1M | 17 | 50%偏斜 | 50.084 | 7.686 | 6.52 |

保留旧业务select/distance/batched_distance、只替换ivf_flat的100k对照也完成，四格分别5.38/5.81/6.86/8.38倍，方向相符。所有timed-sample有序ID/分数检查通过。**因此100k限定负载的tile收益不是只由其他kernel重新编译造成**；不代表其他维度/k/真实业务分布普遍更快，不等同Milvus RPC或FlagFit E2E倍数。

只读布局确认新旧业务算子compiler参数相同不等于binary相同，本轮专门补同编译链对照而不是忽略hash差异。原始数据在 `same-compiler-abba/`、`controlled-abba/` 与 `fresh-pair-hashes.json`、`controlled-hashes.json`。100k/1M同编译链与100k旧selector受控实验的全部1440个计时样本及指纹另归档为 [JSON](ppu-ivf-tile-results-20261120.json)，不依赖本机/root路径即可复核。最终全部五个campaign共80进程/2400计时样本已审计通过；独立profile-on proof有PPU_PROFILE设备事件、无CPU回退，旧runtime加载schema2明确拒绝。正式性能轮均profile-off。设备1的本轮进程已退出，未操作0/6/7的既有服务。

**最终边界**：已验证1024/IP/k20的native query-request局部加速；未验证Milvus RPC或FlagFit全链路收益，不默认部署tile32。其他维度、大k、生产并发、实际owner/deletion表达式成本及split PQ执行仍待验。

### 复现入口

从仓库根目录执行，输出必须选新目录，不覆盖在线root：

```bash
PPU_SDK=/usr/local/PPU_SDK FLAGTREE_BACKEND=ppu \
  /root/flagos22-venv/bin/python python/fl_kernels/compile_ppu.py \
  --dim 1024 --k 20 --metric IP --tle --block 128 --ivf-block 32 \
  --select-block 1024 --num-warps 4 --output <NEW_ROOT>/d1024_k20_ip_m0_s0

LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib taskset -c 0-7 \
  python3 -I tests/ppu/run_ivf_tile_abba.py \
  --executable <TEST_BUILD>/knowhere_ppu_bench \
  --baseline <FRESH_TILE128_ROOT> --candidate <FRESH_TILE32_ROOT> \
  --output <NEW_RESULTS> --device 1 --nb 100000 --dim 1024 --k 20 --metric IP
```

同编译链基线命令只把ivf-block改为128，其余参数与环境不变。特殊机器路径是本轮复现记录，不是通用依赖安装建议。设备编号使用前须确认空闲；命令不含服务重启/集合操作。

### 已完成的检查与失败记录

- 独立CMake runtime/benchmark/新oracle目标与完整Knowhere/Catch2目标构建成功；源码格式、`git diff --check`、Python AST/`--help`检查通过。pre-commit未安装，未执行hooks。
- 完整FL UT首次运行exit127，glog依赖的gflags符号未解析，尚未执行任何测试。使用本构建对应的Conan libgflags_nothreads显式预加载重跑，**35用例/7728断言全部通过**；不伪造符号或更改依赖源码。UT使用Host adapter，不作为PPU执行性能证明。
- 旧业务bundle与split候选在设备1的IP/nq17/k20 oracle各48组合通过，1MiB预算覆盖batch分割；无CPU fallback。最终L2/d16/k3的72组合通过（含10000 common-offset）；既有d128/k10/m16 native完整合约通过BF/IVF/PQ/assign、ties/padding/invalid CSR。这些独立真机合约不替代RPC/应用质量。新增split PQ产物编译通过，但split PQ执行/性能未测，不将旧bundle的PQ合约冒充新split PQ验证。
- 首次构建新目标失败：修改CMakeLists后未重新configure，提示No rule to make target；重新configure后通过，不计为测试通过样本。
- 首次double oracle测试在大common-offset IP失败：FP32约1e11量级会将不同double分数舍入成同分，不能要求double严格排序。保留bounded IP随机/精确tie，large-offset测试限定L2；失败日志保留。这不等同证明任意IP输入逐位无损。
- app-b32开始编译时仍使用旧草稿schema1，结束后仅在本轮独立目录修正schema2；hgbin未改，在线/旧bundle不动。最终源码直接输出schema2；L2/PQ新split bundle按最终schema2生成。
- 非法`--ivf-block 31`在导入编译器/创建产物前退出2；不能将参数拒绝当设备测试失败。

- 100k初轮每格/版60计时样本。nq17/50%随机基线最大666.286ms（p95仍16.539ms），保留原样，不剔除异常值后声称尾延迟稳定。其余尾部和raw保存在 `app-abba/tails.json`，不将低并发微基准p95外推生产。

## 6. 已授权隔离 Search RPC 继续验收

用户明确允许临时19540/metrics9093、物理设备1，独立存储前缀与新`ivftile_*`集合，不访问业务集合、不修改19620在线实例。预检端口空闲、设备1无计算进程，/root约24GiB可用。driver `/root/knowhere-ivf-rpc-20261120/run_rpc.py`，客户端 `tests/ppu/milvus_ivf_tile_rpc.py`。

两臂共用同一Milvus ABI库（依赖对象快照，刷新PPU runtime/IVF_FLAT TU，保存hash），同编译链k20 tile128/32、同k1/IP训练bundle。proof建立100k×1024归一化随机输入/nlist1024/IP索引，正式轮仅reload同collection/artifact，禁止重新训练；核查segment/index、loaded lib、physical allowlist与profiling环境。

全探针k20/nq1，10个交错owner；无过滤、10%/50%/90%可用与全过滤，C1/C8。32个固定query，每格每轮120请求，ABBA；不是120个独立query。每请求验证owner、有序ID、double精确filtered oracle与score。truth/JSON预先生成，oracle核验在单请求计时外；campaign wall含客户端核验/线程调度，不称纯服务吞吐。proof-on/off分离，finally仅停本次Popen PID，数据保留。结果完成后补充。

### FlagFit其他路线最新负结果（只读）

`/tmp/flagfit/docs/analysis/20261120-rerank-pair-batching.md`：默认k20、无截断、10query/200pairs，跨query合批4次forward替代10次，logits/排序一致，但预编码评分三轮慢8.6–9.5%；不推进此调度或移除模型锁。

`20261120-rerank-gemm-specs.md`：M=3342/6321/15688/20969，上/下投影，BM128/stage3全格慢于BM256/stage5；不是单变量BM对比。oracle/事件门通过不等同候选有收益，不部署该规格。

不重复这些已有限否证的策略。FlagFit工作区另有未提交修改，本轮只读，未覆盖其工作；剩余rerank优化需新的算子/布局证据。

### 隔离验收阻塞与恢复记录

初次`/root/milvus/bin/milvus`完成1024/IP设备训练与索引构建，但Load反复报`CUDA not supported: failed to retrieve GPU memory info or no GPUs found`。该二进制没有PPU版Go显存provider，不能因Knowhere算子已接线就认为GPU_*加载准入也已完成。没有查询结果，不记为性能样本。

只对经cmdline确认的本轮client PID4002034发TERM，driver finally清理临时server PID4001800；19540退出，19620/PID3511159保持。保留collection/artifact及失败日志。恢复改用已有`.../milvus-local-bin/milvus-ppu`（ppu build tag/native HG memory provider），不绕过显存检查，不换GPU索引注册名、不改在线二进制。

第一次恢复在服务就绪后的loaded-library路径检查失败：`/proc/maps`是symlink解析后的真实路径。先停止该临时服务，修复为Path.resolve比较，在新`resume-ppu-v2/`日志目录重跑。先前轮次均无计分样本，且全部恢复仍reload已建索引，不混合跨轮数据。

### 完整RPC结果

`resume-ppu-v2/` 的proof及baseline-A/candidate-A/candidate-B/baseline-B全部完成，四个segment行数28980/29530/30000/11490固定；输入/query hash、index参数、每个计时请求的有序ID/精确filtered score/owner检查通过。10负载×120请求×4轮=4800个计时REST请求；proof另80请求与预热。查询32个独立qid，重复不算新样本。设备proof有512条FL_IVF_SCAN，全部device1/device_done1；无过滤full_scan1，owner筛选full_scan0。正式轮完全关闭profile，未发现CPU fallback。

两block p50均值（ms/request）：

| 可用行数 / 排除比例 | C | tile128 | tile32 | 比值 |
|---|---:|---:|---:|---:|
| 100000 / 无过滤 | 1 | 8.712 | 8.719 | 0.999 |
| 100000 / 无过滤 | 8 | 48.376 | 40.194 | 1.204 |
| 10000 / 90%排除 | 1 | 6.478 | 5.223 | 1.240 |
| 10000 / 90%排除 | 8 | 28.856 | 20.820 | 1.386 |
| 50000 / 50%排除 | 1 | 8.367 | 6.570 | 1.274 |
| 50000 / 50%排除 | 8 | 40.124 | 25.269 | 1.588 |
| 90000 / 10%排除 | 1 | 8.636 | 6.540 | 1.320 |
| 90000 / 10%排除 | 8 | 39.488 | 31.325 | 1.261 |
| 0 / 全过滤 | 1 | 2.606 | 2.632 | 0.990 |
| 0 / 全过滤 | 8 | 10.192 | 7.651 | 1.332 |

**限定判定**：过滤负载C1两候选轮均改善、C8总体p50改善；原native约5–8倍被host路由/候选/segment/HTTP开销稀释。无过滤与全过滤控制项不应有tile计算收益，C8仍变化说明背景调度/服务波动，不能将这些点归因tile。C1控制项接近持平。

**尾延迟未改善**：50%可用C8候选两轮p95=175.40/184.11ms，基线178.70/173.34ms；10%可用候选174.95/148.34ms，基线154.08/155.96ms。不能称稳定tail改善或生产吞吐收益。campaign wall含oracle与线程调度，不作纯服务QPS。

profile-on独立证明轮中，每segment路由中位数约0.25–0.31ms、候选生成约0.06–0.52ms、tile32 scan约0.12–0.17ms（10%/50%可用）。这些阶段是带profiling、并混合C1/C8的诊断，不与off RPC时延直接相加。全探针仍计算所有质心分数/排序只为生成所有list，是下一步可移除的主机工作；候选排序变化须以精确ID/tie/过滤回归验收。

原始[RPC JSON](ppu-ivf-tile-rpc-results-20261120.json)含4800样本、proof、segment与逐qid真值。全部临时PID已退出、19540关闭、19620/PID3511159仍在；数据保留。**不是FlagFit完整RAG成绩**，没有改在线bundle或默认配置、没有提交推送。

## 7. 全探针过滤路径跳过质心路由

基于第6节实际阶段证据，仅在`nprobe == trained nlist`且bitset非空时跳过质心距离/排序/probes矩阵，按list ID顺序单遍生成过滤候选。所有list都被选择，候选集合不变；不复用跨请求bitset，不并行/重复调用过滤器，不修改partial-probe路径或既有无过滤BF快路径。设备/CPU scanner仍以global ID打破同分。

新增`[ivf-full-filter]`：L2/IP、逆序质心与全相同base、nq2、0/3/16/32可用、成功设备Host adapter和注入失败CPU fallback，有序ID/分数/padding检查。完整库构建通过，全部FL **36用例/8392断言通过**。首次新测试错误地要求Host adapter padding为infinity，但现有adapter使用有限极值；按既有contract检查ID=-1与正确符号极值后重跑通过，不修改产品padding语义。

独立候选库 `route-library/`，baseline为第6节schema2库；两臂固定同tile32 bundle与已建collection，只比较host节点变化。`run_route_rpc.py`使用同授权隔离范围，proof+off ABBA、32qid/C1/C8/10负载、每格每轮120请求，逐请求精确filtered oracle不变。未完成前不声明新RPC收益。运行日志及最终结果保存在`route-rpc/`，不混合第6节tile对照数据。

### 完成结果：C1过滤请求获益，C8混合

proof与off ABBA完整结束，新增4800计时REST请求均通过固定segment/input/query/index、每请求filtered exact ID/score/owner检查。proof全部full-probe阶段route_ms=0，确认分支实际生效；设备1执行、正式轮profile-off、无CPU fallback。两库PPU runtime对象字节一致，仅IVF_FLAT对象不同，两臂同tile32。

| 可用行数 | C | 原tile32节点ms | 跳过路由ms | 比值 |
|---:|---:|---:|---:|---:|
| 100000（无过滤） | 1 | 8.728 | 8.720 | 1.001 |
| 100000（无过滤） | 8 | 44.517 | 50.751 | 0.877 |
| 10000 | 1 | 5.221 | 4.084 | 1.278 |
| 10000 | 8 | 27.127 | 27.915 | 0.972 |
| 50000 | 1 | 6.678 | 5.442 | 1.227 |
| 50000 | 8 | 33.321 | 29.313 | 1.137 |
| 90000 | 1 | 6.643 | 5.367 | 1.238 |
| 90000 | 8 | 31.286 | 30.809 | 1.015 |
| 0 | 1 | 2.607 | 2.739 | 0.952 |
| 0 | 8 | 9.589 | 8.380 | 1.144 |

过滤C1两个候选轮均改善约1.14–1.28ms，不靠降低k/nprobe或减少eligible获得。**C8未统一改善**：10%可用p50退化约2.9%；无过滤控制项退化12.3%（该路径未改），候选A/B亦明显波动，说明并发服务/驱动背景限制归因。不能声称普遍并发收益。

部分C8 p95改善但未作稳态结论：10%可用基线163.96/165.73ms→候选62.52/80.12ms；50%可用178.25/174.67→154.66/105.69ms。90%可用基线A p95=529.24ms异常保留，不用该异常分母宣称稳定尾延迟改善。0可用C1候选退化约5%，保留。

原始[路由RPC JSON](ppu-ivf-full-filter-rpc-results-20261120.json)含全部4800计时样本与proof。有序tie与CPU fallback合约已单测；此RPC未覆盖真实文档/QA、百万、其他维度/k、多租户删除/动态表达式或生产并发。性能均为限定描述性结果，不是FlagFit完整RAG成绩。

临时进程退出、19540关闭，在线19620/PID3511159未变，数据保留。下一步若验收应用E2E，须在隔离应用/真实冻结query与质量门下测完整链路，不能将本轮C1省约1ms直接外推RAG秒级时延或部署默认。

### 真实RAG pilot通过与描述性计分

`real-rag-v3/`完成118600行导入/设备建索引，baseline/candidate各3条完整RAG；所有candidate ID/text hash、recall score容差、rerank score、最终top5、prompt hash/tokens、retrieval usage一致，3条答案也一致。6条全部length stop（256输出预算），仅功能一致性，不是QA门。

候选首条search=790.745ms，基线42.145ms；后两条基线17.845/17.527ms、候选12.903/13.131ms。首条异常保留，不拿后三点挑选宣称收益。pilot用作正确性，不计分。

通过后启动`real-rag-score/`：固定同collection/artifact和tile32，仅旧节点/跳过route节点不同；首20个固定query、C1、baseline-A/candidate-A/candidate-B/baseline-B，每臂先3个完整业务预热，正式共80请求。独立campaign lock持续持有，不重启31520/19620，不调用原业务集合。预注册DESCRIPTIVE_ONLY、稳态/QA未验，不扩C8或以追加样本碰绿。输出预算/k/top5不变，每请求比较prompt与rerank结果；正式结果完成后补。

## 8. 已授权真实业务快照与完整RAG小规模验收

用户明确允许调用31520的现有FlagFit模型服务（卡0/6/7），不修改/重启服务；隔离19540/卡1新建冻结业务数据集合，不访问19620业务集合。driver持有`/tmp/flagfit-campaign.lock`，锁不可用即停止；与其他计分互斥。

业务快照118600×1024，原始向量/实体摘要hash先核验；每个spool行检查row/主键/owner/text hash与FP32向量逐位一致，不重新embedding/归一化/截断文档。原业务owner可用118592行（排除8行），不是合成10%可用负载。新增集合保留原schema（pk512、metadata nullable、单shard）。同索引固定、两臂tile32，只比较host route跳过。

先3个冻结query的完整embedding→REST Search→rerank→prompt→生成pilot，召回20/top5/input8192/output256、相同模型与tokenizer；比较candidate IDs/text hashes/scores、rerank logits顺序和prompt hash/tokens。没有QA或稳态资格，pilot不作性能宣称；通过后再决定小规模描述性ABBA，不将3个query冒充100题质量验收。

首轮导入因测试schema将metadata设为非nullable，原spool metadata=null触发Milvus `json field metadata is illegal`，首批失败，无RAG计分。核对冻结schema后修复nullable及pk长度/shard数，在全新`real-rag-v2/`和新集合重跑，原失败集合/日志保留。driver finally清理本次临时PID，在线19620/31520不改。产物`/root/knowhere-ivf-rpc-20261120/real-rag-v2/`，当前结果待补。

第二轮已导入大部分数据，但在row118588处快照一致性门停止：全部spool审计发现8行text_sha256字段为空，而pk/owner顺序正确。这8行原文重新SHA256均与独立已验证entities快照一致；不是忽略数据不一致，新的导入显式验证原文后用快照指纹填入新集合，原spool/快照不改，修复记录保存为`real-snapshot-mismatches.json`。在新`real-rag-v3/`、新集合完整重跑，不沿用未完成集合计分。

### 完整RAG最终结果与交付边界

原`real-rag-score/`因会话中断，仅baseline-A/candidate-A各20条完成，candidate-B为14条、baseline-B未执行，明确拒收，不混入统计。新`real-rag-score-v2/`完整重跑20个冻结query的C1 ABBA，共80个正式请求（每臂每轮3条业务预热，另计12条）。同一collection/artifact、tile32，两库仅host full-filter routing不同。

全部80条transport/usage成功；每qid召回20条candidate IDs/text指纹、recall score容差、rerank score、top5、prompt hash/tokens一致，20个qid四轮答案均一致。独立三query proof产生21条FL_IVF_SCAN，全部device1/device_done1/full_scan0/route_ms0；正式四轮profile-off、无CPU fallback。证明轮不混入性能结果。

| 指标 | baseline-A | candidate-A | candidate-B | baseline-B |
|---|---:|---:|---:|---:|
| 平均E2E s | 6.6972 | 6.7208 | 6.6954 | 6.8068 |
| E2E p50 s | 6.7298 | 6.7193 | 6.7158 | 6.7264 |
| Search平均 ms | 18.265 | 49.898 | 22.927 | 40.592 |
| Search p50 ms | 18.073 | 13.440 | 13.196 | 17.866 |
| rerank平均 ms | 527.073 | 528.665 | 528.703 | 541.739 |
| 生成平均 s | 6.1260 | 6.1181 | 6.1193 | 6.2001 |

两臂各40条平均E2E为6.7520→6.7081s（约0.65%下降），但轮间方向不一致、独立稳态未验，**未证明稳定端到端加速**。Search两block p50均值17.969→13.318ms（约1.35倍），全样本均值29.429→36.413ms反而恶化：候选有189–297ms异常，baseline-B有472ms异常，全部保留，不删异常值重算胜出。生成约6.1s、rerank约0.53s，正常Search省约4.65ms不能直接兑现秒级E2E收益。

72/80条因256-token预算length stop，保持原策略、不截掉失败或延长输出掩盖截断；答案一致不等于自然完整答案QA。结论保持DESCRIPTIVE_ONLY，不扩C8、不宣称生产吞吐/尾延迟或部署默认。

全部临时server已退出，19540关闭；31520/PID3930017、19620/PID3511159保持，不修改在线模型/bundle。真实业务请求、文本、答案和逐qid衍生日志仅保留本地`/root/knowhere-ivf-rpc-20261120/real-rag-final-local-v2.json`及实验目录，不纳入GitHub。此处仅提交脱敏汇总和方法；合成向量RPC JSON另按第5–7节归档。

下一步优先定位检索异常与生成关键阶段，不继续扩大已低占比Search微调计分。现有跨query合批/BM128规格负结果保留，不重复碰绿。
