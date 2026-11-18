# R6 workspace复用原型与RPC结果（2026-11-18）

> 本文记录初版实验。当前工作区已补空闲淘汰、清理和统计接口；第二版结果见
> [缓存第二版验收](ppu-r6-workspace-cache-v2-20261118.md)，初版成绩不作为第二版成绩。

## 状态

新增默认关闭的workspace租赁原型，`KNOWHERE_FL_PPU_WORKSPACE_CACHE_MB`未设置/0时关闭，
可设置1–1024MiB；在第一次arena请求初始化时读取，后续修改环境不重配。
**保留为工作区实验，不启用生产默认，不宣称完成生产级池。**
扫描workspace上限仍为64MiB，缓存预算不同于单请求扫描预算。

实现位于`src/common/fl/adapter/ppu/fl_ppu_runtime.cc`：
- process-wide预算包含池内idle与inflight；超预算走普通request-owned分配，故它不是进程总显存上限。
- idle条目按context匹配、best-fit租出，请求独占；driver alloc/free不在池mutex中执行。
- 仅完整成功launch/D2H后并再次同步的arena可缓存；失败lease销毁并释放预算。
- idle条目清除stream指针，防止请求随后销毁显式stream留下悬挂引用。
- cache hit和实际lease容量记录在PPU_PROFILE；增加workspace_release_ms，runtime_ms现在包含显式arena回收。
  与旧runtime_ms边界不同，不能未经修正直接跨版本相减。
- 不缓存query/候选内容、结果或packed base；query仍每请求alloc/H2D/free。

重要限制：**尚无idle淘汰/扩容替换策略**。多形状、小buffer可占满预算，后续较大请求退回普通分配，
即使idle总量足够也不会自动释放小entry；下面1M/p768证明轮已出现该情况。
池state按既有ContextState方式保留至进程退出，没有主动清理/管理API，不用于要求卸载释放资源的生产插件场景。
真实driver失败下stream完成保障沿用Buffer原有析构；本轮只模拟未完成lease丢弃，不声称注入真实设备故障。

## 验证

新增原生`test_ppu_workspace.cc`检查：
- 未完成lease丢弃；best-fit命中保持真实容量；5个同时inflight请求不alias，1MiB预算仅4个256KiB条目入池。
- 不同context不复用；超预算请求普通分配；显式stream轮换、变化query、L2/IP结果无陈旧数据。
- 实際kernel路径6次调用，改变query后独立已知真值验证ID/分数。
- 既有CSR契约测试在cache开启下验证IVF/PQ、L2/IP、空/不够k候选、多batch和partial tail，两种stream。

独立原生测试构建`/root/knowhere-r6-opt-20261118/pool-contract/`成功；新租赁测试及CSR测试通过。
第一次新测试使用错误bundle路径而失败，随后使用`build/ppu-kernels`中d5/k3 bundle通过，
`pool-test.log`保留失败，`pool-test-fixed.log`、`pool-csr-test.log`为成功证据。
完整Knowhere/UT构建及FL **29用例/7493断言通过**（host正确性，不替代原生证据）。
格式、git diff --check通过。没有执行全仓库UT、ASAN/TSAN、故障注入或饱和多请求kernel并发。

## RPC对照

产物 `/root/knowhere-r6-opt-20261118/pool-rpc-1794993762532608219/`。
脚本父目录`run_pool_rpc.py`、`analyze_pool.py`；库`pool-rpc-library/`。
同一个新runtime库、同一个已验证SIMD-only节点object；只改变cache环境0/512MiB。
沿用授权19540隔离实例和固定索引产物/queries/segment布局，扫描workspace64MiB。
其余88线程/PASSIVE/CPU affinity0–87/GOMAXPROCS8/bundle固定。
proof-on0/512分开，正式off按0/512/512/0串行执行，每进程每点3预热+60样本。
每样本ID/分数与最初baseline一致；segment布局检查通过。
192条PPU_PROFILE，fallback=0、日志severity ERROR=0。正式off无profiling。
临时进程均停止，原19530 PID1163356持续运行，数据保留；无提交/推送或生产配置修改。

| NB/nprobe | cache关闭p50均值 ms | cache512p50均值 ms | 均值降低 |
|---|---:|---:|---:|
|100k/512|13.472|12.637|6.2%|
|1M/16|10.988|10.058|8.5%|
|1M/512|85.877|84.085|2.1%|
|1M/768|127.899|119.064|6.9%|

同库cache开关，而非旧库对新库的完整配对，因此不能称cache关闭代码完全没有新增instrumentation边界变化。

尾延迟仍不一致：100k/p512 cache两轮p95=21.425/13.965ms，off17.771/15.458ms；
1M/p16 cache11.361/14.493ms，off12.761/12.176ms。
1M高探针off包含779/840ms级p95，背景漂移严重，不删除这些样本、不据此泛化大倍数尾延迟收益。
1M/p512 p50轮次交错，也不能以2.1%均值认定统计显著。

## 相位与池饱和

按base_ptr调用序号关联runtime到节点，再按start_us落入RPC窗口筛选；校验candidate_count及每请求segment数。
仅适用于本次concurrency1，不能泛化同segment并发。相位来自profile-on，不能直接与off RPC相减。

100k/p512：alloc中位数0.314→0.000185ms，release0.0363→0.00174ms，命中中位数1；
1M/p16：alloc0.154→0.000296ms，release0.0258→0.00103ms，命中中位数1。
**复用确实消除了命中请求的设备alloc/free成本**，并不减少候选上传量或batch数。
1M/p512命中中位数1，但candidate H2D中位数仍有大幅波动。
1M/p768命中中位数0：缓存被前面形状占用而未淘汰，alloc仍0.628ms（off0.748ms），
故其6.9% RPC均值变化不能全归为直接cache-hit收益，可能含前序状态/竞争和系统漂移。

显存快照在collection release之后采集，池仍保留device arena，文件`*-memory.csv`；
不是请求peak、单进程增量或纯索引驻留量。尚未追加retained/idle/inflight实时计数和容量管理接口。

## 后续门槛

下一步应先补idle淘汰、可控清理和retained/inflight accounting的可测API，再验证多context、
真实并发kernel调用、故障回收，以及不同query/候选形状下cache命中。
性能继续用同库on/off paired RPC，不将旧组合SIMD+partition结果当成当前实现。
在这些门槛完成之前，本原型保持默认关闭、未提交，不推广成生产无回归结论。
