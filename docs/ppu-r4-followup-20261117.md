# R4 执行链纠偏、CSR 单次写入优化与重测指令（2026-11-17）

## 1. R4 实际结果：CPU fallback，而非成功 PPU 检索

输入 `/tmp/milvus-ppu-e2e-验证报告-r4-20261117.md`。本轮只读核查原始脚本/结果/日志，未修改原报告、未重启服务、未删除集合。

- `/root/milvus_run/milvus_stdout.log:3620–3621` 确有八卡 native HG adapter 注册；注册不代表检索 kernel 成功。
- 同日志 `3682–3687` 明确报 `KNOWHERE_FL_PPU_KERNEL_DIR is unset`，`3699–3704` 是100k BF六次 CPU fallback；IVF对应 `3943–3964`。百万 BF/IVF分别见 `8623–8682`、`9545–9599`，每类30次回退，符合五segment×一次预热及五次计时。
- 本轮日志没有 `device_done=1` / `FL_IVF_SCAN` 成功证据。失败以stderr/WARN呈现，ERROR为0不能证明PPU成功。
- `/root/start_milvus.sh:18–21` 未设置AOT根目录；当前进程环境也缺失。legacy `fl_kernels/brute_force_topk.ppuo`不是native runtime使用的manifest/hgbin。

**纠正要求：** 保留原始31.41/183.65ms等数值，但将FL列标注为“FL路径（本轮CPU fallback）”。删除“PPU功能四轮全绿”“PPU比CPU快1.30倍”“候选拷贝非瓶颈”“百万PPU扫描超线性”和3–4倍性能预测。R4无法支持这些执行链/归因结论。

BF Add/Deserialize已上传base并持有`dev_xb_`，Search上传query并复用它（`src/index/fl/fl_index_node.h`）。NQ100/D128每segment query只有51,200B，不是每次512MB base。训练失败同样可能CPU回退，不能把本轮build_s归因于PPU训练。

### 其他测试口径纠正

| 项目 | 可核验事实 / 明确要求 |
|---|---|
| Milvus SHA | 实际是`663f0d57407a489030b94e986ea60ad615b930d1`，不是563f0b5。原始结果的server版本、服务日志41/43行和build_milvus13c.log:547一致。 |
| Knowhere SHA | `70c96aa0bbdeac9b60b47b4ad054f5c86e9425ce`成立。 |
| IVF配置 | `/root/milvus_e2e.py:42–58,95,119`实际L2/nlist1024/nprobe16，不是已有独立sweep的128/4配置。 |
| 重复和延迟 | 一次预热、五次整体均值；`time.time()`且query.tolist在计时内。ms/q为batch/NQ摊销，不是单请求延迟；QPS是串行batch等效吞吐，不是并发吞吐。 |
| 真值 | BF最后一次有序ID一致；没有score对照，也没有保存逐次完整ID/score。不能写浮点距离逐元素/位级一致。 |
| 召回 | CPU IVF和FL IVF独立训练，recall约0.17–0.196；固定nprobe比较不等于matched-recall性能验收。 |
| segment | 100k单segment；百万五segment，行数92096/223872/224032/228064/231936。须保存测试前后segmentID/row数；不能借用其他轮布局/计时。 |
| 隔离/缓存 | 新rootPath和空集合成立，但localStorage/rocksmq仍为公共路径、compaction开启且有warmup。不能声称完全冷缓存、固定segment或不复用任何缓存。 |
| 构建 | 构建成功成立；clone证据在build_milvus13.log而不是13c同一行段。Go TMPDIR改到大分区是合理修复。 |

## 2. 本轮性能优化：无过滤 CSR 避免串行清零

基线70c96aa0中`vector<int64_t>::resize`先串行写零，随后限8线程拷贝同一片内存。改为请求私有`unique_ptr<int64_t[]>`默认初始化标量存储：先统计prefix offsets，再分配，按query不重叠区间写入所有元素，最后交给device或CPU scanner。

- 仍保留原候选数量、list/query顺序、probe comparator、拷贝并行阈值和线程上限。
- 过滤路径仍是单遍vector push_back，不重复/并行调用deferred predicate。
- 不新增跨请求缓存、不修改base生命周期、不降低recall、不修改kernel/训练/Add/序列化布局。
- 零候选不读取buffer；扫描失败后同一完整CSR交给CPU。full-probe设备失败也重建新CSR。
- profile仍报告真实candidate数量和有效分配payload；不是RSS或包含allocator开销的总量。

### 固定输入 ABBA 主机阶段测量

复用`tests/ppu/bench_fl_ivf_host.cc`，新增可选nlist参数，覆盖R4的1024。scan callback故意不计算距离，因此**不是PPU kernel或Milvus E2E成绩**。D128/NQ100/k10，seed42、固定交错list，预热2次、计时11次，baseline/candidate/candidate/baseline串行新进程。每格为两个进程各自p50的平均ms。

| OMP最大线程 | NB | nlist/nprobe | baseline | candidate | 延迟下降 |
|---:|---:|---:|---:|---:|---:|
| 8 | 100k | 128/4 | 0.951 | 0.869 | 8.6% |
| 8 | 1M | 128/1 | 1.312 | 1.162 | 11.4% |
| 8 | 1M | 128/4 | 2.225 | 1.031 | 53.6% |
| 8 | 1M | 128/16 | 10.092 | 2.594 | 74.3% |
| 1 | 1M | 128/16 | 14.077 | 11.033 | 21.6% |
| 8 | 1M | 1024/16 | 1.820 | 1.273 | 30.1% |

88线程/1024/16首轮9.411→15.358ms出现回退，不能隐去。进一步三轮ABBA、每进程31次计时为9.425→8.818、9.391→8.598、9.518→8.541ms。说明该大路由并行分支调度噪声很大，复测支持小幅改善，但不承诺88线程稳定加速。未固定CPU affinity/NUMA，不能把以上外推到生产并发。没有同时运行build/benchmark竞争资源。

原始JSON为`docs/ppu-r4-host-results-20261117.json`；本机编译/运行脚本及日志保留于`build/ppu-r4-followup/`。主机benchmark只隔离路由/分配/拷贝，正确性独立由完整测试验证。

## 3. 回归与测试脚本增强

- 完整库及knowhere_tests构建成功；`[fl]`27用例、3674断言通过。
- 大CSR线程一致性回归原有阈值两侧/1、2、8、88线程覆盖保留，新增大CSR scan失败CPU回退的ID/score断言。
- native PPU contract 2/2通过；它验证独立runtime，不冒充本轮node/Milvus E2E已成功。
- `git diff --check`通过；修改C++区间使用clang-format-18。pre-commit未安装，未执行hooks。
- `milvus_nprobe_sweep.py`新增显式`--probes`、每次ID/score/shape/duplicate/finite检查和保存、精确路径按ID匹配FLAT score（rtol/atol=1e-4）、计时前后segment一致性及准备集合关闭autocompaction。
- 脚本已通过py_compile及--help，**未连接服务执行新增E2E逻辑**。该脚本仍依赖外部日志证明native PPU成功，不能仅看PASS；uniform数据假定无边界ties，不作通用ties oracle。关闭compaction不保证不同索引collection布局相同，仍须查看JSON中的实际segments。

## 4. 给测试执行方的明确指令

### 4.1 先修复隔离服务启动环境

不要直接运行会pkill原服务的`/root/start_milvus.sh`，也不要运行会docker rm的start_deps脚本。先准备独立端口（如19540，确认未占用）、etcd/minio前缀及localStorage/rocksmq目录；不要改原19530实例。以下环境必须由**启动Milvus的进程继承**，在客户端shell export不能修复已运行PID：

```bash
export PPU_SDK_PATH=/usr/local/PPU_SDK
export LD_LIBRARY_PATH=/root/milvus/internal/core/output/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export KNOWHERE_FL_PPU_KERNEL_DIR=/tmp/knowhere-FL/build/ppu-implicit/final/kernels
export KNOWHERE_FL_PPU_WORKSPACE_MB=64
export KNOWHERE_FL_PPU_PROFILE=1
export OMP_NUM_THREADS=8
export TMPDIR=/root/gotmp
```

这里AOT根目录已存在；必须包含`d128_k1_l2_m0_s0`及`d128_k10_l2_m0_s0`子目录的manifest及其中所有hgbin。不要指向specialization子目录或legacy.ppuo目录。路径存在不保证正确执行；固定并保存manifest/hgbin哈希及Knowhere/Milvus实际SHA。

先运行独立contract（本轮已用此配置通过）：

```bash
LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-} \
  KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-kernels-tle" \
  /usr/bin/ctest --test-dir build/ppu-contract --output-on-failure
```

这是小shape contract，其bundle不能替代D128/k10服务bundle。服务测试前先做一批smoke，日志必须出现native runtime `PPU_PROFILE`，IVF每个segment/request须`device_done=1`，且本次窗口内**无unset、kernel失败或CPU fallback**。不满足即终止性能验收。8卡注册不意味着单segment使用8卡。

### 4.2 用增强脚本重测，不复用R4回退成绩

在正确启动并通过smoke的隔离服务上执行（以下会创建新的独立集合并保留，不覆盖已有同名集合；需执行方确认端口和前缀）：

```bash
python3 -I tests/ppu/milvus_nprobe_sweep.py \
  --port 19540 --nb 1000000 --nq 100 --nlist 1024 \
  --probes 1 4 16 64 256 512 1024 --repeats 31 \
  --prefix fl_r4_corrected_20261117 --revision '<actual-full-knowhere-sha>' \
  --output build/ppu-r4-followup/milvus-corrected-1m.json --prepare
```

100k另用`--nb 100000`。固定query、seed、segment布局、线程和CPU亲和；显式记录load时间、训练/Add/build边界。nprobe16用于复现R4，其他点用于recall≥0.90/0.95/0.99的matched-recall对照；达不到目标应明确报告，不插值宣称达标。full-probe须recall1且score在声明容差内。

profiling用于定位route/materialize/query staging/CSR H2D/score-select及RPC窗口；正式性能轮在新隔离进程中**unset KNOWHERE_FL_PPU_PROFILE**以关闭日志/事件开销（设0仍会启用）。同集合配对revision按ABBA重载，保存每次样本及结果，报告p50及分布，不把阶段重叠时间求和作RPC延迟。非--prepare模式复用集合时须核验索引配置和数据身份。

### 4.3 下一步真正的优化优先级

1. 先完成上述成功native PPU的R4配置重测，再决定1M瓶颈；现有fallback耗时不能指导kernel优化。
2. 本轮单次CSR写入先做真实配对node/RPC验收，尤其1024质心大路由、五segment争用、过滤及NUMA。
3. 若真实profile显示CSR H2D仍占主导，再设计设备常驻倒排表+probe IDs上传；必须独立验证预算、生命周期和回退，不能用地址缓存掩盖一致性问题。
4. BF base常驻已实现；仅在真实计时证明转置/分配等开销后优化对应阶段，不再重复实现“xb每次H2D消除”。

本轮没有新Milvus RPC加速数字，未验证建设峰值RSS、filtered性能矩阵、生产并发/IP服务/大k。没有提交或推送。
