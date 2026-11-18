# R6 分块路由：隔离 Milvus RPC 验收结果（2026-11-18）

## 范围与证据

用户明确授权临时19540实例、全新etcd/MinIO/localStorage命名空间、100k/1M测试集合。
原19530实例（PID1163356）保持运行；结束只停止本次临时进程，全部数据和日志保留。
无提交、推送、在线库替换或设备reset。实现与库级测量见 [R6后续报告](ppu-r6-followup-20261118.md)。

产物根目录：`/root/knowhere-r6-opt-20261118/rpc-1794985560164425293/`。
驱动、分析器及两版库：`/root/knowhere-r6-opt-20261118/{run_rpc.py,analyze_rpc.py,rpc-libraries/}`。

基线HEAD=`6b01a2bf48cf4e13ccef92bb1a38147a62e4ac00`；候选为本轮分块SIMD路由+高探针partition/sort。
仅重编IVF_FLAT节点translation unit，两版复用相同Milvus兼容对象、依赖和runtime。
库SHA256已核验，逐轮/proc/maps确认加载正确库。Milvus源码48b3d3bf，实际可执行文件SHA256、
节点头SHA256、bundle全部文件SHA256和编译命令见metadata.json及rpc-libraries/metadata.json。
不是重新全量编译两份Milvus的对照。

固定matrix+implicit bundle `build/ppu-implicit/final/kernels`、workspace64MiB、CPU affinity0–87、
OMP上限88、PASSIVE等待、PROC_BIND=true/PLACES=cores、GOMAXPROCS8、METRICS_PORT9092。
新命名空间：`r6audit_1794985560286822423`。

L2/d128/nq100/k10，无过滤，并发1。base seed42，query独立seed43。
proof-on与正式profile-off分离。正式按baseline-A/candidate-A/candidate-B/baseline-B串行执行，
每进程每索引/探针2次预热、30个计时样本；表格为同版两轮p50均值。
时间边界为pymilvus RPC（序列化、服务、设备结果回传），不含query列表转换与计时外oracle。

所有轮次通过逐样本shape、唯一ID、ID域、原始向量L2分数、重复结果稳定性、召回单调性和全探针精确性检查。
跨版本base/query hash、segment布局、有序IDs、每query命中数一致，分数在1e-4容差内一致。
100k FL_IVF_FLAT为1个segment，1M为5个segment；CPU/FL集合的segment布局不完全相同，不能当相同算法工作量。

## 实际 RPC 配对结果

| NB | nprobe | recall@10 | 基线 ms | 候选 ms | 延迟降低 |
|---:|---:|---:|---:|---:|---:|
|100k|4|0.065|5.237|5.197|0.8%|
|100k|16|0.149|5.424|5.331|1.7%|
|100k|64|0.365|6.373|6.240|2.1%|
|100k|256|0.724|10.279|10.086|1.9%|
|100k|512|0.912|13.065|13.662|**-4.6%**|
|100k|768|0.988|15.822|15.296|3.3%|
|100k|1024|1.0|6.093|6.114|-0.3%|
|1M|4|0.073|9.258|8.375|9.5%|
|1M|16|0.175|11.594|10.763|**7.2%**|
|1M|64|0.383|23.567|23.147|1.8%|
|1M|256|0.758|49.031|46.265|5.6%|
|1M|512|0.924|85.248|82.453|3.3%|
|1M|768|0.986|118.919|118.496|0.4%|
|1M|1024|1.0|17.657|18.027|**-2.1%**|

全探针走既有BF shortcut，跳过本轮路由修改，不能把该点的波动归因于SIMD。
BF未改动对照100k=6.063→6.098ms（-0.6%），1M=17.824→17.979ms（-0.9%）。
CPU IVF未改动点也有数个百分点漂移。小幅百分比不能当统计显著结论。
不将这里的CPU FLAT/IVF结果与R6报告直接拼接：环境、bundle、线程等待/布局和计时不同。

### 尾延迟与回退

- 1M/p16基线两轮p50=11.712/11.477ms，候选10.486/11.040ms；两轮p50均优于基线。
  p95基线30.714/30.233ms，候选11.391/41.595ms，**尾延迟没有一致改善**。
- 100k/p512基线两轮p50=13.085/13.046ms，候选13.867/13.458ms，两个候选轮均回退。
  p95基线14.670/14.396ms，候选34.981/14.968ms。
  这是实际回归，未通过删除异常样本或只挑单轮掩盖；不能宣布全场景验收通过。
- 1M/p512基线p50=84.533/85.964ms，候选81.787/83.118ms；p95有明显漂移，原始样本完整保留。
- 1M/p768均值基本持平，两轮候选与基线交错，不认定有效收益。

## 同召回目标

候选保持基线每个探针的结果和召回，属于同质量配对，不是靠减少nprobe换速度。

若限定仍使用普通IVF近似路径：100k目标0.90对应p512（候选回退4.6%），目标0.95对应p768（改善3.3%）；
1M对应p512（改善3.3%）和p768（基本持平）。

但若允许全部已测探针中的最快达标点，**两规模0.90/0.95均选择p1024全探针BF shortcut**：
100k约6.1ms，1M约18ms，召回1.0，比高探针普通IVF路径更快。
因此本次不能把p512/p768称为“最快matched-recall点”，也不能宣称本轮路由修改提高了最佳高质量搜索点。
逐轮自动选择结果见matched-recall.json。

## 执行证明与错误统计

proof轮396条PPU_PROFILE、210条FL_IVF_SCAN，device_done全部为1；fallback=0，severity ERROR=0。
每个计时FL_IVF请求关联到全部预期segment，实际nprobe与请求一致。
396条包括训练/添加、预热、搜索，不是396个RPC。
正式off四轮PPU_PROFILE=0、FL_IVF_SCAN=0、fallback=0、severity ERROR=0。
弃用警告是PyMilvus ORM API提示，不属于连接错误。
分析程序完成并通过，见evidence-summary.json、comparison.json、rpc-analysis.log。

host状态和ppu-smi快照在每规模完成后采集，集合已release；它们不是搜索期峰值或单索引常驻量。
metrics endpoint隔离在9092；所有新命名空间数据保留，未清理对象存储。

## 判定与下一步

本轮完成真实RPC验收，证明低探针1M有约7–10%的p50改善且结果不变，
同时发现100k/p512稳定回退和尾延迟不一致。不能称“所有配置加速”或“无性能回归”。

下一步优先：
1. 针对100k/p512，用相同库/同artifact做baseline与candidate的profile-on配对，拆解route、materialize、
   query staging、runtime alloc/H2D/launch/wait/D2H/free及多segment调度；不由一次kernel事件推断critical path。
2. 分离SIMD和partition/sort做消融，确认回归来自排序阈值、route team、内存first-touch还是服务调度；
   在回归点未解决前保留为工作区候选，不提交/推广。
3. 高探针普通IVF扫描成本显著，先验证CSR/设备workspace分配与上传瓶颈，再设计有界独占lease复用池。
   不以共享mutablebuffer加锁牺牲并发，也不未经SDK语义验证删除同步。
4. 新服务或数据写入延续相同隔离授权范围；原19530始终不动。全仓库UT、ASAN、过滤/IP生产并发、
   Train/Add构建峰值RSS和显存峰值仍未完成，既有库级检查边界见R6后续报告。
