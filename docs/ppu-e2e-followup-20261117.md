# Milvus PPU E2E 问题后续优化（2026-11-17）

## 基线与边界

输入报告：`/tmp/milvus-ppu-e2e-验证报告-20261117.md`，其代码基线为 `823671fe`，使用历史 CUDA-C/shim 路径。当前基线为 `1983a8dc`，实际执行链为自实现 Triton/Triton-TLE → FlagTree AOT → native HG → PPU，不是 FlagGems 或历史 CUDA-C 算子。

本轮没有重跑 Milvus 服务，下面的时间不能当作 Milvus RPC E2E。也不能用历史报告的时间直接计算本轮代码改动的加速比。

原报告本身有需复核的指标：100k IVF 的 0.38ms/batch、NQ=100 应对应约263158 QPS，而表中写11930.5；同一行的29.6倍只由11.24/0.38得出。因此应从原始脚本和日志确认实际批次口径，不将该数字作为验收门槛。

## 已知问题逐项处理

### 1. BF 常驻设备与批量 launch

当前 `FlBruteForceIndexNode::Add/Deserialize` 已调用 `TryStageToDevice`，Search 使用 `dev_xb_`，每请求仅上传 query；runtime 已按二维 grid、query tile 和多层 selection 批量调度。本轮不再重复添加设备地址缓存，也不据历史时间差推断当前仍重复上传 xb。

当前常驻索引、d128/k10/L2/NQ100，矩阵/隐式ID配置、64MiB预算的请求边界复测：100k为1.426ms，1M为11.589ms，均与独立直接FP32 oracle有序ID一致、score mismatch=0。计时含query分配/H2D/free、runtime arena、launch/sync、结果D2H；不含首次index staging、输入生成、AOT及服务开销。这是当前能力复测，不是本轮CSR修改带来的BF收益。

### 2. IVF nprobe 非单调

修复三处缺陷：

- 质心 `partial_sort` 只比较score，等分时nprobe变化可选出非嵌套集合；增加centroid ID升序作为tie-break。
- CPU fallback heap等分时无ID总序，结果可能与设备不同；统一为score优先、global ID升序。
- 少于k个候选时，原CPU出堆从k-1开始写，导致padding在有效结果之前；改为有效排序前缀加末尾padding。

新增Catch2确定性序列化夹具（15个列表，逐个nprobe=1..15，L2/IP、并列/非并列向量）检查候选集合嵌套、逐探针recall不下降、全探针精确以及padding。原生PPU增加2051个逆序候选、多tile/多merge层、等分、空/短列表、全候选精确回归。没有证据表明当前设备top-k合并仍有缺陷，不能将host路由问题错归为kernel合并错误。

### 3. 百万 IVF 的扫描量与工作区

原实现按全请求最大候选宽度分配score批次，却一次上传整个NQ的CSR，candidate空间未计入工作区限制。

现在：

- CSR仅保留最大query batch所需的candidate空间，offset在每批减去起始CSR位置后上传；
- CSR候选、offset及arena对齐开销纳入IVF/PQ工作区预算；
- 每批按实际最大候选宽度启动score/selection，整批空候选直接填padding；
- `KNOWHERE_FL_PPU_PROFILE=1` 输出 `candidate_count`、`candidate_width`、`workspace_bytes`、`candidate_device_bytes`、`scored_slots`，便于区分真实扫描与矩形padding。

均匀候选矩阵下，每query扫描10000/100000，而不是100k/1M全表；但这只是独立runtime输入的验证，不能代替Milvus实际nprobe展开统计。

### 4. Status 兼容

本轮保持 `fl_device_error=35` 不变，避免改变已发布错误码/ABI。当前 `StatusCategoryOf/Status2String/ToSegcoreErrorCode` 已覆盖它；Milvus旧版自己的无default exhaustive switch仍需补case。删除或改成其他枚举的别名会丢失错误语义，且不能保证兼容已有消费者，不作为无验证的优化。

待重新集成Milvus时，确认该版本是否直接使用Knowhere映射；否则仅在Milvus映射补server-side错误case。此次未修改、提交或发布外部Milvus补丁。

## 配对真机结果

同机ZW810E device0、gcc13、128维、k10、L2、NQ100、seed42、concurrency1、CPU oracle固定8线程。每进程先运行oracle和CPU计时，再串行PPU预热2次；oracle不在PPU计时区。比较HEAD `1983a8dc` runtime与本轮runtime，均使用同一AOT目录 `build/ppu-implicit/final/kernels`。旧runtime由 `git show HEAD:...` 独立构建；无CPU fallback。

| 工作负载 | 预算 | 旧runtime p50 | 本轮 p50 | 解读 |
|---|---|---|---|---|
| NB100k，均匀每query10k候选 | 64MiB | 1.011ms | 1.007ms | 持平，5次样本 |
| NB1M，均匀每query100k候选 | 64MiB | 9.248ms | 9.268ms | 约0.2%差异，持平，5次样本 |
| NB1M，前75query各1k、后25query各100k候选 | 4MiB | 6.203ms | 5.507ms | 延迟下降11.2%，约1.13倍，7次样本 |

全部有序ID/score mismatch=0、同候选oracle recall=1。这不是对原始全表GT的IVF recall。

不均衡场景新arena=4024576B（小于4MiB），candidate设备有效空间=1600024B；原整个CSR单独为20600808B（按2575000个int64候选加101个offset计算，非峰值显存实测）。`scored_slots=2674000`，较原全请求矩形100×100000=10000000减少73.26%；实际有效候选2575000，不声称全部padding已消除。

旧runtime7次区间6.178–6.245ms；新runtime5.480–5.999ms。均匀候选不会因分批CSR获得显著速度收益，主要改善设备内存边界；减少batch大小也可能增加launch/H2D次数，故不宣称全输入普遍加速。

原始JSON/日志与构建输出保存在 `build/ppu-csr-results/`（非版本化构建产物）。

## 实际验证

- 独立 `tests/ppu` CMake编译通过。
- native PPU d5/k3基础及嵌套IVF回归通过，默认流/显式流均通过。
- TLE同一路径配置的基础及CSR回归通过。
- 新CSR专项：1MiB、83queries、5003vectors，连续空/短/长列表、partial tail、非零CSR起点；IVF/PQ × L2/IP × 两种流全部通过。
- CSR专项观察 `workspace_bytes=1005312`，小于1MiB；空batch被跳过，`scored_slots`等于该夹具有效候选数70067。
- 完整库 `libknowhere.so` 和修改后的UT源文件编译通过，但完整UT及专项Catch2链接失败：`undefined reference to folly::f14::detail::F14LinkCheck<(F14IntrinsicsMode)2>::check()`。未添加假符号、禁用link check或使用CPU fallback掩盖；Catch2新增测试尚未执行成功。
- 修改行经clang-format-18格式化，`git diff --check`通过。

## 下一轮验收建议

先修复现有Folly编译/依赖intrinsics mode不一致，执行 `[fl]` Catch2和新增nprobe夹具；随后重新构建并隔离启动Milvus测试实例，不影响已有服务。在真实service调用记录index staging次数、CPU fallback次数、每query实际候选数、kernel路径和workspace，再运行相同query/filter的nprobe sweep（1、4、16、nlist），对照精确GT。

本轮未验证完整Milvus E2E、88核CPU对照、生产多卡并发、不同dim/大k及筛选组合，也未测量build peak RSS或改变索引最终存储布局。CSR streaming不减少host候选物化；query重排、设备倒排表常驻以及跨请求合并均未实施。
