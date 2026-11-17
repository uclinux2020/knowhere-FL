# PPU 优化按优先级实施与验收（2026-11-17）

## 顺序与范围

承接 [host/H2D定位](ppu-host-transfer-stages-20261117.md)，依次实施：1) 无过滤CSR精确分配；2) 全探针无过滤BF复用；3) 小工作量路由减少OpenMP启动；4) 固定单/多segment布局调度实验。前三项修改Knowhere执行策略，第四项新增实验脚本并完成真实服务验证，不在缺少queue/reduce独立计时的情况下贸然修改Milvus调度器。

本轮沿用653e72fd及未提交的阶段计时改动；Triton/TLE → FlagTree AOT → native HG → PPU，非CUDA-C/FlagGems。服务独立19540，原服务不停止/替换，测试结束后只停止本次实例。未自动提交或推送。

## 1. 无过滤CSR精确分配

对empty bitset先累计选中列表长度、生成prefix offsets，再一次resize，按query不重叠区间填充。不改变candidate顺序。大于等于1048576个候选ID且NQ>1时并行填充，较小CSR顺序填充。filtered路径保留单遍串行predicate，不重复/并行调用evaluator。

初始并行阈值131072导致百万/nprobe1的6.25MB小CSR物化约4.22ms，OpenMP启动成本抵消了收益；最后提高到1048576，物化降到0.637ms。百万全探针阶段一测得RPC240.634ms（此前552.455ms），但仍需800MB candidate H2D，因此继续第二步。

一次分配减少vector增长、容量膨胀及中间复制，未改变index Build/Add/Deserialize布局或最终存储。resize仍有初始化写入；filtered CSR仍可能扩容，不宣称所有过滤配置内存收益。

## 2. 全探针无过滤复用精确BF

仅当nprobe==nlist、bitset.empty、设备resident且brute_force_topk接口存在时，跳过host routing及CSR，使用同一dev_xb、query和请求k调用设备BF。使用实际k，不按NB截断结果形状。任何set-device/query staging/BF kernel失败均重建原candidate CSR并执行原CPU扫描；不会返回未初始化结果。

新增回归覆盖L2/IP、k1/10/70（NB67）、不足k padding、只剩一个有效向量的filter及设备BF故障。回归暴露旧Host adapter的短结果padding写在有效结果前、tie无ID总序，修复共享host top-k及同类PQ heap，以便CPU模拟adapter与设备/CPU节点语义一致。Host adapter不是PPU性能证据。

百万全探针不再物化100000000个ID，实际profile candidate_bytes=0、route/materialize/H2D=0，RPC第二步16.546ms、最终16.337ms；相对此前552.455ms限定样本约33.8倍。该收益是跳过host CSR与复用此前已存在的矩阵/隐式ID BF配置，不是kernel突然提升33倍。不支持的specialization依然回退，不保证任意bundle全探针都更快。

## 3. 小工作量host路由

路由策略保留每query相同dot累加和partial_sort comparator，只改变任务调度：nq>1且nq*nlist*dim大于约8388608个乘加项才启用OpenMP静态并行，否则顺序执行（判定以除法避免乘积溢出）。128维/NQ100/nlist128约1638400项，顺序路径route降到约0.71–0.91ms；此前约3.56–4.02ms。

没有更换浮点算法、训练逻辑或search参数。该阈值是本机实测策略，不是所有CPU/NUMA/维度/并发最优值；大路由矩阵分支的性能尚未做系统配对。

## 最终百万服务结果

相同持久化单segment、NB1M、128d、NQ100、L2/k10/nlist128、seed42。每项预热2次计时7次，CPU oracle在计时外。列为RPC p50 ms：

| nprobe | 定位基线 | 步骤1 | 步骤2 | 最终步骤1–3 | 最终recall@10 |
|---|---:|---:|---:|---:|---:|
| 1 | 14.892 | 20.931 | 23.669 | 7.321 | 0.052 |
| 4 | 28.444 | 26.841 | 25.623 | 19.020 | 0.197 |
| 16 | 64.264 | 47.591 | 45.623 | 35.698 | 0.462 |
| 128 | 552.455 | 240.634 | 16.546 | 16.337 | 1.0 |

最终与定位基线各nprobe结果ID一致，recall不下降，全探针精确。各阶段是串行不同进程/轮次，并非严格ABBA，仍可能受主机负载/缓存影响，不把微小差异当作显著收益。最终route=0.708/0.763/0.911/0ms；CSR物化=0.637/8.268/15.643/0ms。

## 4. 固定segment布局及调度实验

新增 `tests/ppu/milvus_segment_layout.py`，新建独立collection，设置 `collection.autocompaction.enabled=false`，分区隔离避免不同分区合并。对同seed数据分别创建1/4个分区，实测1/4个sealed segment，总NB100k一致；测试前后segmentID/row数不变。每次query都是NQ100，4segment在多卡执行，精确GT使用已有同数据FLAT。

| nprobe | 单segment RPC p50 | 四segment RPC p50 | 单/四node跨度 | 四node开始时间跨度 |
|---|---:|---:|---:|---:|
| 1 | 6.929ms | 8.297ms | 0.994/1.008ms | 0.064ms |
| 4 | 7.491ms | 8.523ms | 1.965/1.434ms | 0.046ms |
| 16 | 10.919ms | 9.718ms | 4.678/2.892ms | 0.096ms |
| 128 | 6.654ms | 6.852ms | 1.417/0.976ms | 0.085ms |

四segment summed node工作/span约3.04–3.62，证明实际重叠并行，不是四段串行。node结束后到RPC结束，中位数约3.53–4.64ms（单段2.90–3.95ms），含结果归并/服务/客户端反序列化等，不能全部叫调度开销。小探针四段稍慢，nprobe16并行扫描收益抵消额外开销。

不同segment独立训练质心，近似recall不同：单段0.034/0.139/0.403，四段0.052/0.180/0.452；不将近似延迟直接归因为同recall调度收益。全探针均recall1，且总NB/query完全一致，可作更公平的限定对照。没有修改errgroup/服务scheduler或结果reduce，保留正确并行机制。固定布局实验运行于初版step3阈值；最终只调整小CSR填充阈值，其多segment性能尚未重测，不混用为最终成绩。

## 验证

- 最终完整库/Catch2二进制构建通过，26个FL用例、3448断言全部通过。
- 原生PPU contract 2/2通过，双流、CSR/PQ等回归覆盖保留。
- 真实Milvus每步sweep完成；最终逐queryrecall非下降、全探针recall1，无CPU fallback。
- 固定布局1/4segment实验通过，实测布局前后不变。
- 分析脚本将full_scan的零candidate BF记录与node关联；H2D等时间为嵌套指标，不重复求和。

原始产物位于 `build/ppu-csr-results/step{1,2,3}_1m.json`、`final_steps_1m.json`、`final_steps_analysis_1m.json`、`step4_layout_100k.json`、`step4_analysis_100k.json`，服务日志在 `build/milvus-e2e-followup-run/`。未测建设峰值RSS、filtered性能矩阵、IP服务RPC、大k真机、生产并发、NUMA亲和与matched-recall近似单/多segment配对。新建测试collection保留于独立命名空间，原服务不受影响。
