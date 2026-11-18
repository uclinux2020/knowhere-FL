# R6 路由消融与最终保留实现（2026-11-18）

## 结论

完成原始baseline、组合候选、仅SIMD、仅partition/sort四种真实Milvus RPC消融。
**当前工作区撤回partition/sort，只保留分块质心SIMD缓存；所有探针恢复partial_sort。**
这不是证明partition/sort必然有缺陷，而是在实测组合回归、排序收益不稳定情况下缩小优化范围。

上一轮组合候选结果见[RPC报告](ppu-r6-rpc-results-20261118.md)，不得当成当前仅SIMD候选的成绩。

## 实验与证据

产物：`/root/knowhere-r6-opt-20261118/ablation-1794989041101744421/`。
脚本：同父目录`build_ablation.py`、`run_ablation.py`；消融库在`ablation-libraries/`。
沿用已授权19540临时实例及上一轮命名空间、同一集合/索引产物、query seed43、原始有序ID/分数。
不重新建库、不改19530服务，完成后全部临时进程停止，数据保留。

其余环境同前轮：88线程上限、PASSIVE、0–87 CPU affinity、GOMAXPROCS8、64MiB workspace、
相同matrix+implicit bundle与Milvus可执行文件。两种消融库只替换IVF_FLAT节点TU，保存库/头SHA256。

4个独立profile-on进程，各测100k/p16和p512（3预热+3样本），共48条PPU_PROFILE；
fallback=0，日志severity ERROR=0。
正式off顺序baseline/combined/simd/partition/partition/simd/combined/baseline，
每进程每点3预热+60计时样本。每样本计时外检查有序ID和L2分数与前轮baseline一致，segment布局也检查一致。
不将profile-on相位与off RPC墙钟直接相减。

## Profile-off结果（同版两轮p50均值）

| 工作负载 | baseline ms | combined ms | SIMD-only ms | partition-only ms |
|---|---:|---:|---:|---:|
|100k/p16|5.602|5.371|5.375|5.493|
|100k/p512|12.930|13.217|12.967|12.783|
|1M/p16|11.664|10.825|10.879|11.486|

- SIMD-only在1M/p16两轮p50=10.975/10.783ms，对应baseline11.671/11.658ms，均值改善**6.7%**。
  p95 SIMD12.387/14.554ms，baseline14.156/13.000ms；尾延迟仍未一致改善。
- 100k/p16 SIMD两轮5.198/5.552ms，baseline5.719/5.484ms；均值改善4.0%，但轮次交错和漂移存在。
- 100k/p512组合两轮13.162/13.272ms，相对baseline12.907/12.952ms均回退，均值回退2.2%。
  与前轮组合回退4.6%方向一致。
- 100k/p512 SIMD两轮12.823/13.112ms，均值回退0.3%，视为基本持平，**不是证明零回归**。
  partition-only均值改善1.1%，不足以证明该策略有稳定普遍收益。

仅SIMD保留低探针主要收益，避免保留组合版本明确复现的p512回退。不同轮顺序/背景负载不完全隔离，
不能把结果严格解释为两种算法存在必然的非线性相互作用或认定统计显著性。

## 相位观测与限制

100k/p16实际节点route中位数baseline0.430ms、combined0.223ms、SIMD0.219ms、partition0.442ms；
scan约0.27–0.28ms，低探针route收益来自SIMD，不来自partition（该探针仍走partial_sort）。

100k/p512节点route：baseline2.001、combined1.809、SIMD2.040、partition1.964ms；
materialize分别1.633/1.218/0.785/0.999ms，scan约4.97–5.06ms。
profile-on与off墙钟方向不完全一致，只有3个证明样本，不能据此定位RPC回退到某个具体相位。
仍需请求级服务调度/merge及高候选量传输与分配验证，不以kernel时间差倒推critical path。

## 最终检查与未验证项

保留节点与实测SIMD-only头除注释/空白外一致（自动对比通过）。完整库重建和全部FL回归通过，
结果在`simd-final-build.log`、`simd-final-tests.log`。格式/diff检查通过。
已有expanded CSR tests继续覆盖L2/IP、dim1/17/128、nlist64/65、过滤、线程变化和失败回退；
去除“排序dispatch边界”注释，因为最终实现不再有该dispatch。

未重跑仅SIMD全探针矩阵/所有高探针1M RPC；不能把之前组合版全部结果移作当前结果。
未完成生产并发、ASAN、全仓库UT、Train/Add wall与构建峰值RSS实测，缓存额外512KiB/nlist1024/d128仍为布局计算。
未提交/推送，不替换在线库。

下一步建议优先拆解高候选量runtime H2D/分配/回收和服务调度，再决定workspace有界租赁池或候选上传优化。
当前仅SIMD是范围较小、低探针收益经独立消融复现的工作区候选，不宣称全部场景无回归。
