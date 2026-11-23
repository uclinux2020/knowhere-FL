# knowhere-FL

**knowhere-FL is a fork of [Knowhere](https://github.com/zilliztech/knowhere) that introduces a backend-neutral device abstraction layer (FL) for accelerating vector search on diverse AI chips, built on top of [FlagOS](https://github.com/flagos-ai), a unified open-source AI system software stack.**

基于 [zilliztech/knowhere](https://github.com/zilliztech/knowhere) 的 fork，新增后端中立的 **FL（FlagOS）设备抽象层**与 4 个 FL_* 索引节点，用于在国产 GPU / PPU 上加速 Milvus 向量检索。

- **设备抽象层**：`src/common/fl/`（FlDeviceOps 函数表 + 显存池 + RAII 缓冲 + 多卡路由 + host / ppu 适配器）
- **索引节点**：`src/index/fl/`（FL_BRUTE_FORCE / FL_IVF_FLAT / FL_IVF_PQ / FL_CAGRA）
- **PPU 实现**：`src/common/fl/adapter/ppu/`（Triton / Triton-TLE → FlagTree AOT → 原生 HG）；AOT 编译器 `python/fl_kernels/compile_ppu.py`
- **测试**：`tests/ut/test_fl_index.cc`（Host adapter 单测）、`tests/ppu/`（真机合约与微基准）
- **构建开关**：`WITH_FL`（设备后端）、`WITH_FL_PPU`（Zhenwu/ZW810E 芯片适配器）

## 架构总览

knowhere-FL 沿用 FlagOS 生态 fork 项目的插件化叙事（参照 [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL) 的 `plugin/<vendor>/` 设计）：**上游 Knowhere 的检索接口与索引名不变**，只通过一个后端中立的设备抽象层接入多元 AI 芯片。

| 层 | 位置 | 说明 |
|---|---|---|
| 设备抽象层（后端中立） | `src/common/fl/` | `FlDeviceOps` C-ABI 函数表 + 显存池 + RAII 缓冲 + 多卡路由 |
| 芯片适配器（对应 `plugin/<vendor>/`） | `src/common/fl/adapter/` | `host_adapter`（CPU 参考实现）/ `ppu_adapter`（Zhenwu/ZW810E） |
| 索引节点 | `src/index/fl/` | `FL_BRUTE_FORCE` / `FL_IVF_FLAT` / `FL_IVF_PQ` / `FL_CAGRA` |
| AOT 编译工具 | `python/fl_kernels/compile_ppu.py` | Triton/Triton-TLE 源码 → FlagTree PPU backend → `.hgbin` + `manifest.json` |
| 微基准 / 测试 | `python/fl_kernels/`、`tests/ut/`、`tests/ppu/` | 算子微基准、Host 单测与真机合约 |

接入新芯片只需实现一个 `FlDeviceOps` 适配器并注册——与 Megatron-LM-FL 的 `override_registry` 机制同构，主代码不感知具体芯片。

> PPU 只有**一条统一真机路径**：Triton（TLE 为其异步加载扩展，不是第二套后端）→ FlagTree PPU backend → `.hgbin`/`manifest.json` → 原生 HG context/memory/module/launch。运行期不链接 PyTorch/ATen、CUDA-C kernel、CUDA runtime 或 `libppu_runtime.so` 兼容层。`ppu/` 下的 CUDA-C 融合算子与 shim 是历史验证产物，`WITH_FL_PPU` 构建不编译、不链接它们，其加速数字不归因于当前 FlagOS Triton 路径。
>
> `docs/ppu-*.md` 为逐轮实验记录（含原始数据与边界），属于历史证据，不是本文档的当前状态承诺。

## 当前状态（2026-11-23）

| 领域 | 状态 |
|---|---|
| 设备抽象层 | 已完成：C-ABI 函数表、`FlMemoryPool`（`TryReserve` 原子预留）、RAII 缓冲、多卡 router、Host/PPU 适配器 |
| 索引节点 | 4 节点均可用：`GPU_BRUTE_FORCE`/`GPU_IVF_FLAT`/`GPU_IVF_PQ`（用户面名）+ `FL_CAGRA`；接口合规（Serialize 版本头、`GetVectorByIds`、`HasRawData`、资源估算） |
| PPU 真机路径 | 已完成：Triton/Triton-TLE AOT → 原生 HG 执行；适配器在首次 `GetDeviceOps/DeviceCount/DeviceAvailable` 时 `call_once` 懒注册，无需 Milvus 额外初始化 |
| 设备算子 | `brute_force_topk`（L2/IP）、`ivf_flat_scan`、`ivf_pq_scan`（ADC，单字节码，非残差布局）、`kmeans_assign`（top-1 路由，质心更新留 CPU）；`graph_search` 未实现，CAGRA 仍走 CPU 遍历 |
| 训练/编码 | IVF_FLAT Train 的设备 assign 已接线；PQ 的 coarse/subspace Train 与 Add 编码仍为 CPU，设备化是独立后续任务 |
| 构建 | `WITH_FL=ON` 与 `WITH_GPU/WITH_CUVS/WITH_CARDINAL` 互斥、与 `WITH_LIGHT` 互斥；`WITH_FL_PPU=ON` 需要 `PPU_SDK_PATH` |
| 测试 | FL 单测 36 用例 / 8392 断言通过（Host adapter）；`tests/ppu/` 覆盖 runtime、注册、workspace、CSR、split tile、assign 矩阵 oracle 等真机合约 |

### 可选的 PPU 编译配置

均通过 `compile_ppu.py` 离线编译，产物按 `dim/k/metric/PQ layout` 匹配；默认配置不变，未启用时走直接距离与 64MiB 预算：

| 配置 | 作用 | 状态 |
|---|---|---|
| `--tle` | TLE 异步向量加载 | 统一路径内的加载实现，不作为独立后端 |
| `--matrix --queries-per-block N` | IEEE FP32 矩阵距离 | 可选；128 维/k10 限定验收通过 |
| `--pq-lut` | per-query LUT + ADC | 可选；PQ 候选改善约 2.1–2.35x |
| `--implicit-ids` | 取消完整 score 表的重复 int64 ID | 可选；BF 限定验收通过 |
| `--transpose-base` | 维度优先 base 布局 | 可选；额外显存独立限额 |
| `--compact-score` | 局部 top-1 压缩（仅矩阵 k1） | 可选；k10 融合退化，未启用 |
| `--ivf-block` | IVF/PQ score tile 与 dense tile 解耦 | 实验配置；split-tile 输出 schema2，旧 runtime 明确拒绝 |

### 性能证据边界

- **原生请求局部**：BF 128 维/k10 相对旧 PPU 实现 100k/1M 改善 79.7x/54.7x；1024 维/IP/k20 同编译链 tile128→32 的 native 请求改善 5.3–8.3x。均为限定配置，不等同 Milvus RPC 或应用 E2E 倍数。
- **隔离 Milvus RPC**：1024/IP/k20 过滤负载 C1 下 tile32 与“全探针过滤跳过质心路由”各改善约 1.1–1.3ms；C8 结果混合、尾延迟未改善，不能声称普遍并发收益。
- **构建**：k1 assign 矩阵 bundle 为 opt-in；1M 索引创建完成时间约 2.18x 方向性改善，但 nlist128/1M 的轨迹并非逐 query 无损，未默认启用。
- **真实业务 RAG 小规模**：20 个冻结 query 的 E2E 差异约 0.65% 且轮间方向不一致（生成约 6.1s 主导），仅 DESCRIPTIVE_ONLY；未证明稳定端到端加速。
- **未验证**：生产并发、多卡请求内分片、大 k、其他维度、真实过滤表达式成本、构建峰值显存/RSS。

## Milvus 集成指南

> 目标：在国产 GPU（首发 PPU/ZW810E）上以 `GPU_BRUTE_FORCE` / `GPU_IVF_FLAT` / `GPU_IVF_PQ`
> 原索引名运行 Milvus，用户面零改动。

### 0. 前置条件

| 项 | 要求 |
|---|---|
| Milvus 版本 | v2.4+（knowhere 以 conan 包形式被 internal/core 消费） |
| Milvus 二进制 | 需带 PPU native HG 显存 provider 的构建；官方 stock 二进制的 Load 会报 `CUDA not supported: failed to retrieve GPU memory info`，不能因算子已接线就认为 `GPU_*` 加载准入已完成 |
| knowhere-FL 构建 | 本仓库 `cmake -DWITH_FL=ON [-DWITH_FL_PPU=ON -DPPU_SDK_PATH=...]` |
| 芯片适配器 | PPU：`WITH_FL_PPU=ON` + PPU 运行时 SDK；其他芯片：实现 `FlDeviceOps` 并注册 |
| PPU 算子产物 | `KNOWHERE_FL_PPU_KERNEL_DIR` 指向离线 AOT 产物根，且 specialization 与 `dim/k/metric/PQ layout` 匹配 |
| 无卡验证 | 不装任何适配器即为 Host 回退模式（功能正确、性能为 CPU 档） |

### 1. 集成路径（二选一）

#### 路径 A：conan 本地包覆盖（推荐，贴近生产）

```bash
# 1) 在 knowhere-FL 仓库下打本地 conan 包（版本号与 Milvus conanfile 中声明的一致，
#    例如 knowhere/2.x.x@milvus/dev，以目标 Milvus 版本的 conanfile.py 为准）。
#    conanfile.py 已内置 with_fl / with_fl_ppu 选项。
cd knowhere
conan create . --version=<milvus_expect_version> --user=milvus --channel=dev \
    -o with_fl=True                     # CPU 回退模式（无卡可跑）
# 或 PPU 真机：
PPU_SDK_PATH=/opt/ppu_sdk conan create . --version=<ver> --user=milvus --channel=dev \
    -o with_fl=True -o with_fl_ppu=True

# 2) Milvus 构建将自动从本地 conan 缓存取包，无需改动 Milvus 源码
cd milvus && make
```

#### 路径 B：源码级覆盖（开发联调用）

Milvus internal/core 支持以本地 knowhere 源码树替代 conan 包。将本仓库
`knowhere/` 目录软链或拷贝到 Milvus 的 knowhere 第三方源码路径，并在
`internal/core/CMakeLists.txt` 的 knowhere 段加 `-DWITH_FL=ON` 传入。
联调期改动 knowhere 无需重新打 conan 包。

### 2. 芯片适配器注册（关键一步）

knowhere 提供 `RegisterDeviceOps` 挂载点，适配器应在索引使用前完成注册：

- **PPU**：`WITH_FL_PPU=ON` 构建时，首次调用 `GetDeviceOps()`、
  `DeviceCount()` 或 `DeviceAvailable()` 会通过 `std::call_once` 懒注册原生 HG 适配器，
  不再要求 Milvus 手动调用 `RegisterPpuAdapter()`。还需配置
  `KNOWHERE_FL_PPU_KERNEL_DIR` 并离线编译相应 Triton/Triton-TLE specialization；
  [PPU 算子实现与验证](docs/ppu-triton.md) 给出了真实 SDK、编译和测试命令。
  产物缺失或不兼容会明确报错并触发调用侧 CPU 回退，不能算作 PPU 加速。
- **无适配器**：索引仍可建/查，全部走 CPU 回退路径，性能等同 CPU 索引。
- **Host 适配器**（`RegisterHostAdapter()`）：仅用于 CI/UT，不要在生产注册。

### 3. Milvus 侧配置

milvus.yaml 中与 FL 相关的键（复用官方 GPU 语义）：

```yaml
gpu:
  initMemSize: 0        # FL 显存池初始预留（字节）；0 = 首次使用时按可用显存 1/2
  maxMemSize: 0         # 显存池上限；0 = 全部可用显存。超过即触发 OOM 预检回退 CPU
```

建索引参数与官方 GPU 索引完全一致：

```json
{"index_type": "GPU_IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
{"index_type": "GPU_IVF_PQ",   "metric_type": "L2", "params": {"nlist": 1024, "m": 16, "nbits": 8}}
{"index_type": "GPU_BRUTE_FORCE", "metric_type": "IP", "params": {}}
```

约束（与官方一致 + FL 特有）：
- GPU_IVF_FLAT / GPU_BRUTE_FORCE 支持 L2 / IP；GPU_IVF_PQ 仅 L2（朴素点积
  ADC 无法分解 IP 距离，Train 直接拒绝并给出指引，勿配 IP）
- COSINE 请先归一化后用 IP（建索引时会收到明确报错提示）
- 不支持 range search / grouping search
- IVF_PQ 仅 nbits=8 且 dim 需整除 m；不存原始向量（HasRawData=false，
  影响 query 时 output_fields 直接回向量——需要该能力请用 GPU_IVF_FLAT）
- bitset 过滤语义：GPU_BRUTE_FORCE / GPU_IVF_FLAT / GPU_IVF_PQ 均为精确
  过滤（满 k；BRUTE_FORCE 设备路径内部 over-fetch 保证）；FL_CAGRA 为
  后置过滤，极端过滤比例下可能返回少于 k 条有效结果（与官方 GPU_CAGRA
  直接拒绝过滤查询相比是降级而非报错）
- PPU 需要可选的编译配置（matrix/LUT/tile 等）时，bundle 必须与调用参数匹配；
  未启用可选配置即使用默认 direct 距离与 64MiB 预算

### 4. 验收清单

| # | 检查项 | 通过标准 |
|---|---|---|
| 1 | Milvus Standalone 启动，querynode 日志出现 `FL PPU native HG adapter registered` | 适配器注册成功 |
| 2 | GPU_IVF_FLAT 建 100 万向量索引 | 成功且显存占用 ≤ 数据量 ×1.2 |
| 3 | 批量搜索（nq=1000, k=10） | recall@10 vs CPU faiss ≥ 0.95，QPS ≥ CPU HNSW 3× |
| 4 | 显存池 cap 为 1 字节重启 | 自动 CPU 回退，结果正确，无崩溃 |
| 5 | 拔卡/禁用适配器重启 | 功能正确（CPU 档性能），无 fl_device_error 崩溃 |
| 6 | 序列化（compaction/加载）往返 | recall 不变，格式版本头校验生效 |

### 5. 已知限制

- 单 segment 绑定单卡（router 轮询分派跨 segment 摊多卡；segment 内不分片）
- PQ 为非残差实现，召回略低于 faiss 残差 PQ；PQ 训练与编码仍在 CPU
- 训练采样策略未实现（全量训练）；千万级以上建议先抽样训练再全量 Add
- CAGRA 为 CPU 构图（>20 万行明确拒绝并指引 IVF_PQ）、设备 `graph_search` 未实现；仍走 CPU 遍历
- PPU runtime 私有 workspace 尚未纳入 `FlMemoryPool` 限额
- k1 assign 矩阵 bundle、split IVF tile、全探针过滤跳过路由均为 opt-in 实验配置，未默认部署
- 矩阵 k1 训练产物与 direct 不保证逐 query 一致（1M 低 probe 召回有下降），采用前需明确质量准则

## 构建方式

```bash
# 完整库（含 PPU 适配器与 UT）
PPU_SDK_PATH=/usr/local/PPU_SDK make WITH_FL=True WITH_FL_PPU=True WITH_UT=True WITH_DISKANN=False

# 或直接使用 CMake
mkdir -p build && cd build
cmake .. -DWITH_FL=ON -DCMAKE_BUILD_TYPE=Release          # 与 WITH_GPU 互斥
# PPU 真机适配器（可选）：
# cmake .. -DWITH_FL=ON -DWITH_FL_PPU=ON -DPPU_SDK_PATH=/opt/ppu_sdk
make -j
```

UT 默认使用 Host adapter，无需 PPU 即可验证设备路径语义：

```bash
make test                                        # 运行全部 UT（需先 WITH_UT=True 构建）
build/Release/tests/ut/knowhere_tests "[fl]"     # 仅 FL 用例（Catch2 tag）
```

PPU 真机合约与 AOT 产物见 [PPU 实现与验证](docs/ppu-triton.md)：

```bash
cmake -S tests/ppu -B build/ppu-contract \
  -DPPU_SDK_PATH=/usr/local/PPU_SDK \
  -DNLOHMANN_JSON_INCLUDE_DIR=/path/to/json/include -DCMAKE_BUILD_TYPE=Release
cmake --build build/ppu-contract -j2
```

## 待办与后续方向

| 优先级 | 事项 | 状态 / 下一步 |
|---|---|---|
| P1 | k1 assign 矩阵 bundle 默认化 | 已完成 100k/1M 构建与隔离 RPC 验收；需固定 segment/真实数据、确定质量准则 |
| P1 | PQ 设备端 Train / 编码 | 未实现；需 dsub/k1 pack、传输成本与同召回 oracle 专项验证 |
| P2 | PPU `graph_search` 与设备构图 | 未实现；CAGRA 仍 CPU 构图 + CPU 遍历 |
| P2 | SQ8 / SQ4 节点 | 无 FL 节点；属产品能力新增，需配置/格式/白名单/ADC 整体交付 |
| P2 | IVF tile / 过滤路由优化推广 | 仅验证 1024/IP/k20；需其他维度、大 k、生产并发与真实表达式成本 |
| P2 | 生产并发、多卡请求内分片、构建峰值显存 | 未验证；不以低并发微基准外推生产吞吐/尾延迟 |
| P3 | 训练采样、workspace 纳入显存池预算 | 未实现 |

## 设计决策速查

- D1 FlagOS 全系算子库优先（FlagTree 是编译链，不等于 FlagGems；按实际执行链归因）
- D2 独立后端、编译期与 CUDA 互斥、用户面沿用 GPU_* 名称
- D3 与 FlagFit/vLLM-plugin-FL 统一算子版本基线
- D4 COSINE = 归一化 + IP（同官方 GPU 约束）
- D5 OOM 前置校验 + CPU 回退（`TryReserve` 原子预留）

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/ppu-triton.md](docs/ppu-triton.md) | PPU Triton/Triton-TLE 实现、AOT 编译与真机验证 |
| [docs/ppu-*.md](docs/) | 逐轮优化与实验报告（含原始 JSON、边界与失败记录） |
| [ppu/README.md](ppu/README.md) | 历史 CUDA-C 融合算子与 shim（保留对照，不参与 `WITH_FL_PPU` 构建） |
