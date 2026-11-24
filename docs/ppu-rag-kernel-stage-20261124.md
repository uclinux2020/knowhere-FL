# RAG kernel 阶段性测量（2026-11-24）

## 交付边界
本阶段与 FlagFit 的 embedding→检索→reranker 分析联动。Knowhere源码基线为
`733c70d2`；**未修改检索库、距离kernel、top-k实现、索引参数或生产bundle**。
本提交是组件测量、证据与后续阻塞的交付，不是新优化上线。

FlagFit 本轮实现了默认关闭的批量CLS gather：减少embedding池化前的全token回传，
保持原有CPU FP32归一化顺序。文档embedding整批吞吐改善，但C8分档延迟门未过，
因此不能把该结果写成完整RAG性能提升。详见联动报告与工具：
- [FlagFit 三段报告](https://github.com/uclinux2020/flagfit/blob/main/docs/analysis/20261124-encoder-milvus-kernel-e2e.md)
- [真实向量原生探针](https://github.com/uclinux2020/flagfit/blob/main/tools/bench-rag/recipes/rag_kernel_knowhere_native.py)
- [薄C ABI桥](https://github.com/uclinux2020/flagfit/blob/main/tools/bench-rag/recipes/rag_kernel_knowhere_probe.cc)
- [分段解析器](https://github.com/uclinux2020/flagfit/blob/main/tools/bench-rag/recipes/analyze_knowhere_native_proof.py)

## k20原生PPU组件：已测

数据：冻结业务向量118600×1024、FP32/IP，保留全部行，以原owner条件生成118592个
候选（排除8行）。从100条heldout中取前20条，保持查询行序。调用现有PPU
`Search`，候选CSR已在host生成；**不包含Milvus表达式计算、质心路由、segment、RPC或embedding**。

环境：物理卡1（PPU-ZW810E），自由频；单campaign锁串行；每k先3次预热，
正式20 query×3轮=60样本，profile-on。未独立认证稳态，不能用此窗口报生产QPS。
原生库来自历史 `route-library/libknowhere.so`，bundle来自历史tile32候选；不是本轮
从HEAD重建。C ABI桥只转发既有API，不实现另一个距离算子或CPU回退。

| 分段 | 60样本中位 ms |
|---|---:|
| 距离扫描 score_ms | 0.305600 |
| top-k select_ms | 0.044040 |
| 候选H2D | 0.090765 |
| 结果D2H | 0.016782 |
| workspace申请 | 0.001699 |
| workspace释放 | 0.002177 |
| 原生runtime | 0.485403 |

外侧query H2D＋Search＋同步平均0.512689ms、最大0.555785ms；
它与内部runtime不是同一计时边界。计时外的FP64精确IP＋global-row tie break
参考校验全部通过：每次有序ID完全一致，分数在atol=rtol=1e-4内。
这是当前固定候选上的精确计算验证，不是重新执行Milvus ANN召回验收。

**结论**：该k20原生扫描与选择的成本不到1ms，不能将历史十几毫秒的服务search
父段全归因于这两个kernel。剩余差额可能涉及多segment、路由、host和RPC，
本轮没有服务内证据，不归因。没有基线/候选kernel A/B，因此没有Knowhere加速比。

## 未完成与停止条件

- k50/k100：已知bundle目录缺对应规格。私有AOT脚本已在FlagFit准备，但执行
  本仓 `python/fl_kernels/compile_ppu.py` 被跨仓代码权限检查拦下；没有换工具绕过，
  没有执行编译或高k测试。此次提交授权不当作此前被拒绝执行的自动解禁。
- Milvus隔离：默认配置支持embedded etcd＋local storage，但现有二进制metrics
  listener绑定wildcard。隔离recipe要求private loopback netns；预检原文为
  `unshare: unshare failed: Operation not permitted`。没有启动服务，没有修改已有集合，
  没有退回共享etcd/对象存储或host网络。
- 未测：k50/100、RPC、ANN建索引/构建内存/落盘大小、增长索引Add、全过滤/少于k、
  多segment、多租户表达式/C8检索。不能用当前60点代替这些覆盖。

## 原始证据与复现条件

本仓 [raw/20261124-rag-k20](raw/20261124-rag-k20/) 保存逐请求JSONL、独立warmup、
FP64参考top-k、profile完整日志、分段对齐JSON、数据/bridge/bundle指纹与阻塞记录。
向量正文、模型、库与二进制不入库；本地路径仅是复现索引，不保证其他机器存在。

复现需本地业务快照、PPU SDK、历史库/bundle、显式设备和全局campaign锁；参考
FlagFit脚本的参数，不直接执行旧线上服务recipe。k20 proof命令形状：

```bash
python -I tools/bench-rag/recipes/rag_kernel_knowhere_native.py \
  --out <new-private-directory> --bridge <built-C-ABI-bridge.so> \
  --bundle <validated-bundle-root> --queries 20 --repeats 3 --profile --device 1
```

本次交付只核验文档/原始记录一致性，不重跑设备或执行被阻止的AOT步骤。
