# R6 workspace缓存第二版验收（2026-11-18）

## 状态与实现

第二版补上初版小形状占满容量的缺口：请求无法best-fit命中且预算不足时，
仅在idle容量足够的情况下淘汰空闲条目；淘汰期间实际buffer仍计入retained预算，
完成context归属释放后才预留新请求容量。driver调用位于pool mutex之外，不动inflight租赁。

新增`GetWorkspaceCacheStats()`和`TrimWorkspaceCache()`，提供limit/retained/idle bytes、idle entries、
hits/misses/evictions。retained-idle包含active/retiring预留量，不等于仅执行中的kernel显存。
Trim仅清理调用时的idle条目，正在执行的租赁仍可随后归还，因此不是shutdown/quiesce API。
profile行增加retained/idle/evictions快照；不是显存查询接口，且多线程时快照可受其他请求影响。

缓存仍默认关闭，环境`KNOWHERE_FL_PPU_WORKSPACE_CACHE_MB`第一次使用时读取；本轮开启值512MiB，
不是单请求workspace上限（仍64MiB），也不是进程/单卡全部设备分配上限。
普通超预算分配、索引/query/packed-base内存均不计入池容量。
无提交/推送、不替换在线库，不启用生产配置。

## 真机正确性检查

扩展`tests/ppu/test_ppu_workspace.cc`：
- 5个同时租赁的buffer不alias，超池预算租赁仍走普通分配；失败未完成lease丢弃。
- 大形状淘汰小idle条目后重新入池；统计evictions增加。
- trim释放idle、保留active租赁，active完成后仍可归还；最终trim后retained为0。
- context隔离、显式stream轮换、变化query的L2/IP独立真值。
- 3个host线程各自绑定context、独立query/stream，执行5次真实PPU kernel链并检查全部结果。
  这证明该测试下并发安全，不证明设备实际overlap或饱和服务吞吐。

原生测试通过，见`pool-v2-test.log`；随后增加profiling统计字段，最终完整Knowhere构建通过，
FL回归29用例/7493断言通过（`pool-v2-full-build.log`、`pool-v2-fl-tests.log`）；
格式/diff检查通过。未运行ASAN/TSAN、全仓库UT、真实driver故障注入或长期压力。

## 固定artifact RPC对照

产物：`/root/knowhere-r6-opt-20261118/pool-v2-rpc-1794994898476542688/`。
脚本父目录`run_pool_v2_rpc.py`、`analyze_pool_v2.py`；独立库`pool-v2-rpc-library/`，SHA256在metadata.json。
同一第二版runtime库+相同SIMD-only节点object，只切换cache0/512，其他条件固定：
已授权19540隔离实例、原集合/索引产物、相同query/segment，L2/d128/nq100/k10、无过滤、并发1、
OMP88/PASSIVE/CPU affinity0–87/GOMAXPROCS8、相同matrix+implicit bundle、64MiB扫描workspace。
proof与off计时分离；off顺序0/512/512/0，每进程每点3预热+60测量。
每样本有序ID和L2分数与最初baseline一致，segment布局核验通过。

| NB/nprobe | cache-off p50均值 ms | cache512 p50均值 ms | 延迟降低 |
|---|---:|---:|---:|
|100k/512|13.054|12.528|4.0%|
|1M/16|10.585|10.404|1.7%|
|1M/512|82.134|80.846|1.6%|
|1M/768|129.507|128.111|1.1%|

1M/p16 cache两轮p50=10.747/10.062ms，off10.532/10.638ms，轮次交错；
1M/p512 cache两轮81.119/80.573ms，off82.687/81.582ms，均值略优但cache第二轮p95=780.131ms；
1M/p768 cache第二轮p95=821.833ms；不能将小幅p50提升包装成低尾延迟收益。
100k/p512 cache两轮12.337/12.720ms，off13.300/12.807ms，p50改善更一致；
p95 cache15.613/13.843ms，off14.639/14.226ms，尾延迟仍非一致改善。
全部异常样本保留，没有删点。与初版不能直接拼接百分比，因为实验时间/系统负载不同。

## 设备证据与命中

192条PPU_PROFILE、fallback=0、日志severity ERROR=0；正式off没有profiling。
按base_ptr调用序号与节点匹配、再按start_us落入客户端窗口筛选，检查candidate_count及segment数。
这是concurrency1场景的关联方法，不能推广同segment并发请求。

本轮所有证明工作负载cache-hit中位数均为1，包括初版未命中的1M/p768。
1M/p768 alloc中位数0.822→0.000627ms、release0.0497→0.00169ms。
说明idle淘汰恢复该形状的复用，**没有消除候选上传/调度成本**；RPC整体均值仅改善1.1%。
proof全程最大retained=527,696,384B，小于512MiB上限536,870,912B；累计7次淘汰。
该计数包含多context池预留量，是内部统计，不是设备实际显存峰值。

候选H2D证明数据在两轮差异较大；多segment相位中位数不能相加当RPCcritical path，
profile-on相位也不能直接和profile-off请求墙钟相减。
日志和原始samples见results.json/analysis.json、每进程log和memory.csv。
显存快照在集合release后采集，不能当作请求peak或独占索引内存。

全部临时进程已停止，原19530 PID1163356继续运行，测试数据保留。

## 判定

第二版完成了有界淘汰、主动trim及并发kernel正确性验证，证明命中时alloc/free开销明显减少。
**仍保持默认关闭、未提交的实验候选**：RPC收益小、尾延迟偶发严重回退，未满足生产无回归门槛。
下一步优先确认H2D/服务调度和背景竞争，做更长同artifact压力及driver故障回收验证；
不要仅以缓存命中率1就推广池，也不要继续无止境增大budget。
