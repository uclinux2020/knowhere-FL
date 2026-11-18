# R6 长样本尾延迟诊断（2026-11-18）

本轮同第二版runtime库、同固定索引/query/segment，cache0/512MiB，scan workspace64MiB。
授权19540隔离实例及存储范围不变，原19530不动；所有临时进程已停止，数据保留。
产物 `/root/knowhere-r6-opt-20261118/tail-diagnostic-1794995590309399893/`；
驱动父目录`run_tail_diagnostic.py`，逐请求前后CPU cgroup、CPU/IO/memory PSI、process stat/schedstat/io快照在results.json。

proof-on每点30样本；正式off0/512/512/0，每点200样本，均3次预热，1M/nprobe512/768。
所有请求有序ID和分数与原始baseline一致，segment布局不变；未删除慢样本。
读计数器位于RPC计时外，但采样可能影响负载，因此与未采样轮不可直接拼接性能百分比。

| 轮次 | nprobe512 p50/p95 ms | nprobe768 p50/p95 ms |
|---|---:|---:|
|cache-off A|89.719 / 102.059|119.291 / 819.307|
|cache-on A|86.005 / 96.667|123.936 / 139.869|
|cache-on B|87.596 / 136.049|124.770 / 164.635|
|cache-off B|88.223 / 98.799|130.860 / 147.761|

1600个off计时请求中28个超过300ms；cache-off同样出现约900ms慢请求，
因此不能把800ms级尾延迟视为缓存开启特有现象。
本轮每个请求CPU cgroup nr_throttled增量均为0，未观察到cgroup带宽限流；
当前quota=-1。cgroup wait_sum增量中位数：普通请求约0.000347秒，>300ms请求约0.736秒。
该计数覆盖共享容器里的其他进程/线程，并不是单Milvus线程调度时间，
**仅支持慢请求与容器调度等待增加相关，不能证明CPU调度是唯一原因或缓存无责任**。
需进一步按全部worker线程采样/设备trace区分host竞争、driver阻塞和服务队列。

之前一次慢请求窗口的只读服务日志有collection release/reload/staging，但主要是workload切换边界，
不能把这些事件直接认定为每个慢请求原因。没有修改服务或设备策略来消除背景负载。
没有证明饱和服务尾延迟改善，池仍默认关闭。

本轮日志和样本已保留；完整相位/PSI关联分析尚未完成，不在此次提交中宣称已定位所有瓶颈。
此前完整构建、29个FL用例/7493断言、原生池并发/CSR检查结果见缓存第二版报告。
