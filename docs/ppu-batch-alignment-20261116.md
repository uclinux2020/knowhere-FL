# 下一阶段：预算受限query批次对齐

## 结果

本轮保持同一Triton（结合TLE）→FlagTree→原生HG→PPU路径，优化runtime的budget-limited query batch：当batch小于总query数且能够容纳一个query tile时，将batch向下对齐到queries_per_block，避免每个中间批次重复执行大量masked query lanes。最终partial batch及单query仍正确处理，预算不增加。不改算子精度、候选数量，不缓存查询结果，不引入PyTorch/CUDA-C生产算子。

| 工作负载 | 旧query请求 p50 ms | 对齐后 p50 ms | 改善 |
|---|---:|---:|---:|
| 100k/nq200，默认64MiB | 1.7269 | 1.6396 | 1.05x |
| 500k/nq200，默认64MiB | 7.1698 | 7.1438 | 接近噪声 |
| 1M/nq200，默认64MiB | 约14.42 | 14.385 | 接近噪声 |
| 1M/nq200，显式256MiB逆序配对 | 14.7872 | 13.2404 | 1.12x |

默认64MiB百万batch原来已是16，恰好整tile，因此没有显著改善；256MiB预算原batch约66，向下对齐到64可减少masked运算。该收益不能冒充默认预算性能，也尚未达到历史CUDA-C端到端10x目标。

环境及计时：device0 ZW810E/HGGC13.0，GNU13.3，CPU0–7/8线程，seed42、dim128/k10/L2、nq200，2次warmup、15次计时；query-request包含query allocation/H2D/free、Search工作区、每请求base转置、计算和result D2H，原始索引驻留，非完整RPC服务E2E。基线为e9fdeee8保存binary；两者使用同一b64/w2/select2048/QPB16/transposed/implicit kernel bundle。没有并行运行性能任务。

## 不采纳的实验

1. 每请求转置同时计算base norm，避免每个matrix tile重复norm：数值通过，但百万query请求由14.416ms退化至48.095ms。HG事件显示score阶段约41.77ms，select约3.54ms，退化在score而非只有packing开销。未做指令级profiler确认具体编译原因，撤销此实验代码与开关。
2. 32-query tile：64MiB为26.308ms，128MiB为16.832ms，256MiB为20.988ms，均未优于原16-query配置，不采纳。
3. packing tile64/128行、64维：约14.40–14.44ms，未带来明显收益，撤销额外参数及runtime分支。

原始成功与退化样本、事件日志均保留在[JSON](ppu-batch-alignment-results-20261116.json)，不只选择最佳实验数据。

## 验证

- 最终只保留runtime批次对齐的6行功能变更；所有norm/packing实验源文件已移除，不更改默认编译配置。
- 默认流128维contract：L2/IP、ties、尾query、PQ/IVF/assign、codebook更新、大偏置L2通过。
- 1MiB预算、nb10000、nq201和nq1实测：所有ordered ID/score mismatch=0，same-representation recall=1。
- 原生C++构建及git diff --check通过；不声明完整Knowhere/Milvus通过，既有Folly ABI问题仍未解决。

复现沿用docs/ppu-transpose-20261116.md的bundle，设置KNOWHERE_PPU_QUERY_E2E=1和显式KNOWHERE_FL_PPU_WORKSPACE_MB=256，对百万/nq200工作负载运行knowhere_ppu_bench。64MiB为默认预算，不隐式改成256MiB。

未测生产RPC全链路、多卡并发、所有shape、真实数据near-tie质量、冷cache、构建/Add峰值RSS和显存峰值。代码及分析尚未提交推送。下一阶段依然需要降低距离计算及中间score物化成本；这轮有限的批次效率收益不能代表10x目标已实现。
