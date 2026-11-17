# Host 候选物化、H2D 与 segment 调度定位（2026-11-17）

## 方法及边界

在653e72fd上增加 opt-in `KNOWHERE_FL_PPU_PROFILE` 分阶段计时，不改变候选/算法/工作区策略。

- Node：host质心路由、串行CSR物化、query分配/H2D、完整runtime调用、node开始wall timestamp与elapsed。
- Runtime：arena分配、CSR offset重基与同步H2D、launch到stream sync、score/selection设备事件、结果D2H、batch数。
- Client：保存每次实际RPC开始/结束的wall时间，以请求窗口关联node区间。
- `tests/ppu/analyze_milvus_stages.py` 用区间跨度衡量多node延迟，不把并行工作时间求和当作RPC延迟。runtime用base地址及候选数量关联，排除BF/训练地址复用；仅适用于同机、隔离、顺序RPC，不是生产并发trace方案。

复用此前独立19540服务、同数据collection、128d/L2/k10/nlist128、NQ100、seed42、每项预热2次计时7次。Triton/TLE → FlagTree AOT → native HG → PPU，64MiB预算。已有服务未停止或替换。本次测试服务结束后已停止。

计时有嵌套：scan包含H2D/设备等待/D2H，launch_wait包含设备score/selection；不能重复相加。各列是独立样本中位数，列之和也不一定等于总中位数。runtime_ms在workspace析构之前结束，node scan_ms包含返回时清理。route_ms包含OpenMP调度、质心打分和partial_sort，不是纯计算量。H2D时间包含offset重基和HG同步调用开销，不等同于DMA事件时间。

## 真实服务测量（ms）

| NB | nprobe | RPC | route | CSR物化 | query staging | 完整scan | CSR H2D | score+select设备事件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100k | 1 | 12.080 | 4.849 | 0.106 | 0.030 | 0.256 | 0.089 | 0.053 |
| 100k | 4 | 14.483 | 3.992 | 0.382 | 0.045 | 0.626 | 0.212 | 0.117 |
| 100k | 16 | 15.765 | 4.120 | 3.800 | 0.044 | 2.240 | 0.807 | 0.315 |
| 100k | 128 | 44.940 | 3.910 | 25.834 | 0.042 | 8.731 | 5.852 | 2.328 |
| 1M | 1 | 14.893 | 4.020 | 1.828 | 0.041 | 1.063 | 0.498 | 0.243 |
| 1M | 4 | 28.444 | 3.558 | 15.930 | 0.039 | 3.104 | 1.862 | 0.816 |
| 1M | 16 | 64.265 | 3.874 | 42.599 | 0.046 | 11.888 | 7.602 | 3.767 |
| 1M | 128 | 552.457 | 5.528 | 430.225 | 0.143 | 108.279 | 76.117 | 31.594 |

全探针百万：物化约占RPC的77.9%，H2D约13.8%，设备事件约5.7%；query staging很小。Node内跨度543.494ms，node之前约2.778ms、之后约5.960ms。node外区间包含客户端/服务RPC、队列、计划/过滤、结果归并/序列化等，不称为已独立测出的调度时间。

小探针：route约4ms，而score/selection小于0.3ms，优化kernel不是首要方向。100k/4存在部分样本node总时间明显高于各细分阶段，说明尚有set_device、query free、主机抢占及调度等未覆盖开销；不能将差额全部归因于某一个操作。

## 物化和传输字节

百万每query全探针需要1000000个候选ID，NQ100形成100000000个int64，CSR候选有效数据800MB（十进制）。实测 `host_capacity_bytes=1370628928`，vector capacity大于有效数据，增长/重新分配复制与串行insert构成主要优化目标。这个数是capacity，不是峰值RSS。

百万探针1/4/16，候选ID总字节约6.25MB/25.00MB/99.99MB。query只约51.2KB；结果约12KB。原始base已经常驻设备，不能把候选ID的800MB H2D错当作重复上传512MB raw base。

源码：`src/index/fl/fl_ivf_flat_node.h::BuildCandidateLists`先OpenMP并行路由，随后串行循环query/list，对单一vector反复insert，无预计算CSR尺寸/精确reserve。runtime每batch重新上传CSR，与score预算共同限制batch；未跨请求缓存候选。这些解释与本轮计时一致。

## 多 segment：本轮发现的关键环境变化

此前百万集合有5个segment；本轮持久化集合已由后台compaction合并为一个NB=1000000的segment。所有计时请求只有一个FL node调用，start spread=0，不能宣称测出了5segment调度损耗。

Milvus当前源码 `internal/querynodev2/segments/search.go:123-135`：单segment直接execute；多segment通过errgroup创建goroutine并等待。多segment并非在这层强制串行，但下游队列/CPU线程池、HG context与共享总线仍可能产生争用。没有本轮多segment区间证据，不能量化或推断其占比。

**单segment仍有552ms RPC延迟，故全探针大延迟不能主要归因于多segment调度。** 本轮真实主因是host CSR物化，其次候选H2D。

## 优先级建议（未实施算法优化）

1. 无过滤：先计算每query候选长度和prefix sum，一次精确分配CSR，再按query不重叠区间填充；验证reserve/并行写入的配对收益。涉及filtered/deferred bitset时不能随意并行或重复调用evaluator。
2. 全探针无过滤：考虑用精确BF已有接口跳过CSRID物化与上传；需要相同metric、padding、tie、过滤语义和bundle覆盖，不能只将全探针误当成近似搜索。
3. 小探针：隔离OpenMP路由启动成本，比较顺序/限线程/批量质心距离；route_ms目前包含调度等待，需固定线程数配对。
4. 从根本上减少ID H2D需设备倒排列表常驻并上传probe IDs，而不是只减少launch；新增布局必须核查预算、Add/Deserialize生命周期、并发与回退。
5. 多segment专项：固定segment布局或禁止测试collection compaction，再做同NB单/多segment配对，记录实际node并发区间、service队列和reduce时间。不能用本轮单segment数字替代。

## 验证及产物

- 原生PPU contract测试2/2通过。
- 完整库/测试二进制构建成功，全部FL测试25用例/2435断言通过。
- 真实100k/1M RPC sweep通过，逐query recall不下降，全探针recall=1；无CPU fallback。
- 分析脚本语法验证通过；重叠区间夹具验证sum=8ms而span=5ms，不混淆并行工作和延迟。首次分析曾被base地址复用污染，加入candidate_count约束后重新生成结果；上表使用修正后的结果。
- 原始JSON/日志：`build/ppu-csr-results/milvus_stages_{100k,1m}.json`、`milvus_stage_analysis_{100k,1m}.json`、`build/milvus-e2e-followup-run/stage-profile-service.log`。

不宣称性能优化已完成；本轮是观测与定位，不是提交新的加速实现。未执行固定segment多卡配对、CPU原生扫描计时、生产并发/过滤/IP/大k矩阵。所有新增改动未提交/推送。
