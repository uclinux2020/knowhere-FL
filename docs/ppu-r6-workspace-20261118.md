# R6 workspace预算对照（2026-11-18）

## 判定

**保持默认64MiB，不推广128/256MiB。** 大workspace减少批次数，但100k高探针稳定回退；
1M高探针仅有小幅、漂移明显的p50收益，尾延迟不一致。没有修改生产配置、runtime代码或在线19530服务。
当前保留代码仍为仅质心分块SIMD，见[消融报告](ppu-r6-ablation-20261118.md)。

## 实验

产物 `/root/knowhere-r6-opt-20261118/workspace-1794989944219656970/`；
脚本同父目录`run_workspace.py`、`analyze_workspace.py`。
复用授权临时19540实例、之前相同集合和索引产物，不重新建库；全部临时进程已停止，数据保留。
同一SIMD-only库、bundle、query、segment布局，唯一设置变化为
`KNOWHERE_FL_PPU_WORKSPACE_MB=64/128/256`，实际分配不必等于上限。
其余固定88线程/PASSIVE/CPU0–87/GOMAXPROCS8/metrics9092，同前轮。
L2/d128/nq100/k10、无过滤、并发1。

3个profile-on进程各点3次预热+3次测量；正式profile-off顺序64/128/256/256/128/64，
每进程每点3预热+60测量。逐样本ID/分数与原始baseline相同，segment布局核验一致。
证明288条PPU_PROFILE、fallback=0、日志severity ERROR=0。
原服务PID1163356保持运行。无提交/推送。

## RPC结果（同预算两轮p50均值，ms）

| NB/nprobe | 64MiB | 128MiB | 256MiB |
|---|---:|---:|---:|
|100k/512|12.765|13.455|13.766|
|1M/16|10.826|10.928|10.755|
|1M/512|87.386|84.285|84.738|
|1M/768|125.396|121.481|119.949|

100k/p512的128/256MiB相对64分别回退5.4%/7.8%，两轮大预算均较慢。
1M/p512均值改善3.5%/3.0%，但64两轮83.369/91.403ms存在明显漂移；
128第二轮p95=249.418ms，64两轮p95=94.043/104.131ms。
1M/p768的256均值改善4.3%，但两轮124.619/115.279ms差异明显；
128两轮p95=165.377/181.270ms，高于64的130.082/138.960ms。
1M/p16原本单批，预算增加没有稳定p50收益。

## 相位与内存

按节点start_us落入客户端窗口关联计时请求，每个请求检查预期segment数量。
CGO节点日志与runtime stderr可能跨行重排，不能取简单“上一条事件”：
分析器按base_ptr的调用序号分别配对，并检查事件/节点总数及candidate_count相等。
并发1使每segment跨请求调用顺序一致；这一方法不能泛化到同segment多请求并发。
相位汇总只使用计时证明样本：100k每点3个节点调用、1M每点15个。
最初按指针聚合混入其他probe样本，已弃用并用上述关联重生成analysis.json，不引用旧汇总。

- 100k/p512批数2→1→1；实际arena峰值65,691,392→109,479,936→109,479,936B。
  candidate H2D中位数3.313→3.200→3.272ms，workspace alloc0.326→0.599→0.531ms。
  更少批次未带来RPC收益。
- 1M/p512不同segment批数[2,4]→[1,2]→[1]；实际最大arena约63.6→127.2→234.6MiB。
  各segment/请求混合的H2D中位数18.834→10.863→10.497ms；不是集合critical path。
- 1M/p768批数[2,6]→[1,3]→[1,2]；最大arena约63.2→127.7→255.5MiB。
  workspace alloc混合中位数1.883→2.601→7.445ms，增大预算也增大分配代价。
- 1M/p16各segment均单批，最大arena7,899,904B，三个预算相同；不因上限升高直接分配256MiB。

以上来自profile-on，不能直接与profile-off RPC相减；多卡segment并发不能求相位和充当请求墙钟。
ppu-smi快照在各轮集合release后采集，不是请求峰值/单索引驻留量。
workspace预算是每请求上限，不是每卡/进程并发上限；增大预算会放大inflight显存压力。
未测显存峰值或饱和生产并发。

## 下一步

不再用简单增大workspace作为默认优化。若继续runtime改造，应优先建立有界、按context/device分桶的
inflight独占workspace租赁与回收验证，测alloc/free residual、cache hit和retained/inflight bytes；
同时确认候选H2D成本及SDK异步传输语义。不能直接去掉同步或放一个跨请求共享buffer。
在该功能有独立多线程/故障回收测试和配对真机证据前，不引入生产默认池。
