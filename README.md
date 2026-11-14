# knowhere-FL

**knowhere-FL is a fork of [Knowhere](https://github.com/zilliztech/knowhere) that introduces a backend-neutral device abstraction layer (FL) for accelerating vector search on diverse AI chips, built on top of [FlagOS](https://github.com/flagos-ai), a unified open-source AI system software stack.**

基于 [zilliztech/knowhere](https://github.com/zilliztech/knowhere) 的 fork，新增后端中立的 **FL（FlagOS）设备抽象层**与 4 个 FL_* 索引节点，用于在国产 GPU / PPU 上加速 Milvus 向量检索。

- **设备抽象层**：`src/common/fl/`（FlDeviceOps 函数表 + 显存池 + RAII 缓冲 + 多卡路由 + host / ppu 适配器）
- **索引节点**：`src/index/fl/`（FL_BRUTE_FORCE / FL_IVF_FLAT / FL_IVF_PQ / FL_CAGRA）
- **微基准**：`python/fl_kernels/`；**单测**：`tests/ut/test_fl_index.cc`
- **构建开关**：`WITH_FL`（设备后端）、`WITH_FL_PPU`（Zhenwu/ZW810E 芯片适配器）

## 架构总览

knowhere-FL 沿用 FlagOS 生态 fork 项目的插件化叙事（参照 [Megatron-LM-FL](https://github.com/flagos-ai/Megatron-LM-FL) 的 `plugin/<vendor>/` 设计）：**上游 Knowhere 的检索接口与索引名不变**，只通过一个后端中立的设备抽象层接入多元 AI 芯片。

| 层 | 位置 | 说明 |
|---|---|---|
| 设备抽象层（后端中立） | `src/common/fl/` | `FlDeviceOps` C-ABI 函数表 + 显存池 + RAII 缓冲 + 多卡路由 |
| 芯片适配器（对应 `plugin/<vendor>/`） | `src/common/fl/adapter/` | `host_adapter`（CPU 参考实现）/ `ppu_adapter`（Zhenwu/ZW810E） |
| 索引节点 | `src/index/fl/` | `FL_BRUTE_FORCE` / `FL_IVF_FLAT` / `FL_IVF_PQ` / `FL_CAGRA` |
| 微基准 / 单测 | `python/fl_kernels/`、`tests/ut/` | 算子微基准 + 全链路单测 |

接入新芯片只需实现一个 `FlDeviceOps` 适配器并注册——与 Megatron-LM-FL 的 `override_registry` 机制同构，主代码不感知具体芯片。

> 下文正文为 FL 实施工作日志 + Milvus 集成指南。

---

# knowhere-FL 实施工作区与 Milvus 集成指南

> 对应方案文档：《基于FlagOS的Milvus国产GPU加速技术方案.md》v2.1
> 基线：zilliztech/knowhere @ 68b6319（2026-11-13 浅克隆）
> 代码：本仓库 `main` 分支（基于上游基线的 FL 提交序列）
> 日期：2026-11-14 ｜ 状态：4 索引（含 FL_CAGRA）+ 接口合规 + 多卡分派 + 资源估算 + Milvus 集成件已落地，两轮走读修复完成，待远程 PPU 环境联调

---

# 第一部分：实施工作日志

## 第七轮落地（2026-11-14，三路并行走读后的集中修复，commit fb7d629）

走读覆盖全部 30 个改动文件（设备层 / 索引节点 / 构建与测试三路并行），确认 2 个 P0 编译级问题、3 个 P1 正确性问题并全部修复；同时完成 P2 健壮性与 P3 规范性批量治理。

### 1. P0 编译级修复（远程 Linux 全量编译前必须落地）
- **`FlDeviceBuffer` 缺移动赋值**：用户声明了移动构造+析构后移动赋值不再隐式生成，4 个节点的 `dev_xb_ = FlDeviceBuffer(...)` 必然编译失败 → 补齐移动赋值，并顺带修复两个关联缺陷：reset() 对瞬态缓冲无条件 TrackFree 腐蚀池计数（新增 `tracked_` 标志）、跨适配器重注册后经全局表释放显存的 ABI 风险（构造时捕获 `free_fn_`）
- **CAGRA Search 结果缓冲所有权**：`GenResultDataSet(nq,k,vector.data(),...)` 悬垂+双重释放 → 改 new[] 缓冲（与 GetVectorByIds 同模式）
- **expected.h**：`fl_device_error=35` 已加入枚举但 `StatusCategoryOf` 的 `-Wswitch=error` switch 未覆盖 → 全构建编译错误；补 case + `Status2String` 分支
- **fl_ivf_pq_node.h**：`ComputeNorms` 前孤悬 `int64_t` 残留 token；**fl_cagra_node.h**：`Status::format_error` 不存在 → `invalid_binary_set`，`Status::inner_error` 弃用别名 → `knowhere_inner_error`
- **CMake**：`WITH_LIGHT + WITH_FL` 组合此前静默通过但必然链接失败 → 新增 FATAL_ERROR（置于 WITH_LIGHT 声明之后）

### 2. P1 正确性修复
- **BRUTE_FORCE bitset 语义**：设备路径原后置过滤在"被过滤向量落入 top-k"时少返回有效结果（与 CPU 前置过滤路径结果不一致）→ 改为 over-fetch（`fetch_k=min(nb, k+filtered)`，延迟计数未知时按最坏情况）+ 主机端紧凑化，**设备/CPU 两路径结果逐位一致**；UT 同步加严为"屏蔽 rank-0 后整体顺位、满 k 返回"
- **IVF_PQ IP metric 系统性错排**：朴素点积 ADC LUT 无法分解 IP 距离（基向量范数被量化丢弃）→ Train/Deserialize 明确拒绝 IP（L2-only），config 校验、头注释、host adapter 契约注释同步
- **显存池 TOCTOU**：`Ensure()`（只查不订）与 `TrackAlloc()` 分离存在并发超限窗口 → `TryReserve()` 检查+记账原子化，6 处 staging 调用点全部迁移，失败路径一律 `TrackFree` 释放预留

### 3. P2 健壮性
- 4 个节点 Deserialize 全面加固：`ReadChecked` 有界读（MemoryIOReader 静默短读此前会产生未初始化尾部+字段错位）、`FlScalarDimsSane` 拒绝腐败头部引发的巨型分配、IVF 成员 id 越界校验、CAGRA CSR offsets 单调性/终点校验
- `fl_kmeans_cpu.h`：nb≤0/nlist≤0 越界种子守卫 + nlist>nb 防御性钳制
- `fl_device.cc`：环境变量 strtoll ERANGE/溢出钳制、`RegisterDeviceOps(nullptr)` 注销路径不再副作用重置池配置、`DeviceAvailable()` 每次 Add 打日志改一次性、`g_ops` 改 atomic 发布
- CAGRA Serialize 空索引拒绝（与其余三节点及 Deserialize 校验对齐）

### 4. P3 规范与性能
- 13 个头文件 include guard 统一 `KNOWHERE_FL_` 前缀；fl_format.h 补 `<iomanip>` 显式依赖；PPU 适配器头注释 CMake 开关名修正
- `fl_graph_search_cpu.h`：tie-break 确定化（等距按 id，std::sort 不稳定）+ 空图种子越界守卫
- host adapter：brute_force L2 全基向量范数一次预计算（热循环只留点积）；ivf_pq_scan 由"全量 vector + partial_sort"改有界堆（与节点扫描模式一致）
- python 微基准：`torch.cuda.synchronize` → `torch.get_device_module`（PPU/昇腾/海光非 CUDA 后端可移植性）；kmeans CPU update 由 Python 逐行循环改向量化 index_add（消除基准自身失真）；ivf_scan bench 增加 JIT 预热

### 5. bitset 语义最终口径（对用户面承诺）
| 索引 | bitset 语义 | 说明 |
|---|---|---|
| BRUTE_FORCE | 精确（满 k） | 设备路径 over-fetch + 主机紧凑，与 CPU 前置过滤逐位一致 |
| IVF_FLAT / IVF_PQ | 精确（探针列表内前置过滤） | 设备/CPU 候选集相同，结果一致 |
| FL_CAGRA | 后置过滤（可能 <k 有效） | 官方 GPU_CAGRA 直接拒绝过滤查询；FL 降级为过滤而非报错，ANN 索引可接受 |

## 第六轮落地（2026-11-13 深夜）

### 1. StaticEstimateLoadResource（4 个节点）
Milvus load 前的资源预算接口：BRUTE_FORCE 按文件 1:1、IVF_FLAT 按 nb×dim×4、IVF_PQ 按 nb×m(uint8)、CAGRA 按向量+CSR 图估算。

### 2. FL_CAGRA 图索引 v1（Phase 2.3）
- `FlDeviceOps` 新增 `graph_search` 算子；语义金标准抽为共享头 `common/fl/fl_graph_search_cpu.h`（host 适配器与索引节点共用，设备 kernel 必须复现其语义）
- CPU 暴力 kNN 构图（O(nb²)，>20 万行明确拒绝并指引 IVF_PQ，待设备构图里程碑放开）；构图确定性（固定行为，序列化往返结果逐位一致）
- 遍历走设备 graph_search 或共享 CPU 遍历——**CPU 遍历即方案 sanction 的 adapt_for_cpu 模式**
- bitset 后置过滤（官方 GPU_CAGRA 直接不支持，FL 降级为过滤而非报错）；保留原始向量（HasRawData=true）
- **刻意只注册 FL_CAGRA、不抢注 GPU_CAGRA**——v1 尚未达成官方语义与性能承诺

### 3. 显存池环境变量
`KNOWHERE_FL_POOL_INIT_MB` / `KNOWHERE_FL_POOL_MAX_MB`，适配器注册时自动加载（非法值告警忽略、max<init 自动钳制）。

### 4. docs/ADAPTER_PORTING.md（Phase 3 铺垫）
新芯片适配器移植指南：L0~L4 分级落地策略、设备原语/算子契约、CMake 接线模式、6 项验收清单、常见坑（HIDDEN_PARAM/原子性能/流语义/fp32-only）。

### 5. UT 关键修复 + 新增（累计约 23 用例）
- **关键修复**：发现 `Index::Search` 的 bitset 参数无默认值，此前全部两参调用会在远程编译失败——17 处改为与官方一致的 `nullptr` 三参形式
- 新增：坏 magic/高版本序列化拒绝、4 节点静态资源估算、CAGRA 高召回（frontier 256，recall≥0.9）、CAGRA 序列化往返逐位一致、超行数构建拒绝、bitset 后置过滤

## 第五轮落地（2026-11-13 深夜）

### 1. IndexNode 接口合规审计（远程编译排雷）
- **审计发现并补齐关键缺口**：三个节点缺纯虚函数 `GetVectorByIds` / `HasRawData`（不实现则类无法实例化，远程编译必失败）
  - BRUTE_FORCE / IVF_FLAT：从 `host_xb_` 返回原始向量（走 `CompactOutToIn` id 映射），`HasRawData=true`
  - IVF_PQ：仅存量化码，`HasRawData=false`，`GetVectorByIds` 返回 not_implemented 并给出指引
- 同步补注册宏可选静态方法 `StaticHasRawData`（`KNOWHERE_REGISTER_STATIC` 的 static_assert 只强制 `StaticCreateConfig`，HasRawData 为 if-constexpr 可选——Milvus load 前用它判断原始数据可用性）

### 2. 序列化格式版本头（前向兼容）
- 新增 `src/index/fl/fl_format.h`：`{magic="FLID", format_version=1}` 头，三个索引 Serialize 写入、Deserialize 校验
- 后续格式演进（残差 PQ / fp16 存储 / 分片布局）升版本号并保留旧版本读取分支；旧读新报明确错误而非静默错读

### 3. 多卡分派层 FlDeviceRouter（Phase 2 骨架）
- `src/common/fl/fl_device_router.h`：轮询 Acquire() 分派设备 id；索引 staging 时绑定单卡（`device_id_` 成员），Search 用同一卡
- 清除三个节点全部 5 处硬编码 `set_device(0)`
- 预留演进路径注释：shard-per-device 复制、FlagCX scatter/gather、负载感知分派

### 4. Milvus 集成接插件
- `docs/MILVUS_INTEGRATION.md`：conan 本地包覆盖 / 源码级覆盖两条集成路径、适配器注册时机、milvus.yaml 配置、建索引 JSON 示例、**Phase 1.6 六条验收清单**、v1 已知限制（内容已并入本文档第二部分）
- `conanfile.py` 新增 `with_fl` / `with_fl_ppu` 选项并透传 CMake（`WITH_FL` / `WITH_FL_PPU`），exports_sources 已覆盖新增源码

### 5. UT 新增 3 个用例
- 多卡 router 轮询行为（无适配器/单卡均返回 0）
- BRUTE_FORCE GetVectorByIds 逐位对照原始向量
- IVF_PQ HasRawData=false + GetVectorByIds 返回 not_implemented

## 第四轮落地（2026-11-13 晚）

### 1. CMake 接入 PPU 适配器编译开关
- 新增 `WITH_FL_PPU` 选项（依赖 WITH_FL）；ON 时定义 `KNOWHERE_FL_WITH_PPU`，通过 `PPU_SDK_PATH`（env 或 -D）定位 PPU 运行时 SDK 并链接 `ppu_runtime`；未配置 SDK 时 FATAL_ERROR 防误配
- 不加该选项时 PPU 适配器源码编译为空 TU，保持跨平台可构建

### 2. FlDeviceOps 新增 `ivf_pq_scan` ADC 算子
- 接口语义：设备侧对每查询建 LUT（对 codebook reshape 后走 mm 组合），再扫描候选集累加 LUT 项；codes 为 nb×m uint8（v1 仅支持 nbits=8）
- Host 适配器完整参考实现（LUT + ADC 扫描）；PPU 适配器挂 `ivf_pq_scan.ppuo` AOT 挂载点（暂返回 -1 走 CPU 回退）

### 3. GPU_IVF_PQ-FL 索引（Phase 2.1 v1）
- `fl_kmeans_cpu.h`：抽出共享 CPU Lloyd kmeans 辅助（PQ 各子空间 codebook 训练复用，确定性 seed）
- `fl_ivf_pq_config.h`：nlist/nprobe/m/nbits 对齐官方语义；v1 约束 nbits=8、dim 整除 m；COSINE 按 D4 拒绝
- `fl_ivf_pq_node.h`：CPU 训练（粗聚类 + m 个子空间 kmeans）→ 编码（nb×m uint8）→ 设备 staging（OOM 预检）→ 设备 ADC / CPU ADC 双路径扫描；非残差 PQ（v1 简化，已注释）；完整序列化
- `fl_ivf_pq.cc`：双名注册 `GPU_IVF_PQ` / `FL_IVF_PQ`；上游接线（INDEX_FL_IVFPQ 常量、索引表条目）
- 修复：Add 中并行编码与倒排表构建分离（避免 push_back 数据竞争）

### 4. UT 与微基准
- 新增 3 个 PQ 用例：设备 ADC 与 CPU 回退结果逐位一致（D5 验证）+ 召回下界 ≥0.4（待远程校准）、序列化往返确定性、非法参数拒绝（nbits≠8 / dim 不整除 m / COSINE）
- `python/fl_kernels/ivf_scan_composition.py`：IVF_FLAT 候选扫描（index_select+mm+topk）与 IVF_PQ ADC（bmm LUT + gather+sum+topk）的纯 FlagGems 组合微基准，输出 QPS 与"是否需要 ivf_scan 融合算子"的裁决依据

## 第三轮落地（2026-11-13 晚）

### 1. FlDeviceOps 新增 `kmeans_assign` 算子（设备端质心路由）
- `src/common/fl/fl_device.h`：新增 `kmeans_assign` 入口——纯 mm+argmin 组合、**无原子操作**，注释明确质心*更新*（atomic index_add）暂不作设备算子，待微基准确认
- Host 适配器实现（OpenMP，作为设备 kernel 的语义参考）；PPU 适配器骨架同步新增 `ppu_kmeans_assign`（预留 `kmeans_assign.ppuo` AOT 产物挂载点，暂返回 -1 走 CPU 回退）

### 2. IVF_FLAT-FL Train/Add 接入设备路由
- 配置新增 `use_device_assign`（默认 true，train 阶段生效）
- Train：Lloyd 迭代中 **xb 全程只上传一次**，质心每轮重传（nlist×dim 极小）；assignment 走设备，质心更新留 CPU（方案 5.2 混合模式正式落地）；设备失败自动降级 CPU 路由并释放暂存显存
- Add：一次性设备路由（含显存池 TrackAlloc/TrackFree 完整记账）
- 新增 UT：device-assign 与 CPU 路由双路径 recall 一致性（nprobe=nlist 均须 recall=1.0）

### 3. Phase 1.2 kmeans 组合微基准脚本
- `python/fl_kernels/kmeans_composition.py`：完整 Lloyd 迭代的纯 FlagGems 组合实现
  - assignment：`mm`+`sum`+`argmin`（无原子）
  - update：`index_add`（**原子性能审计对象**）+ `clamp`+`div`
  - 输出 hybrid（设备路由+CPU 更新，即当前 C++ v1 行为）vs full-device 的每阶段 P50 对照、index_add 原子裁决、inertia 聚类质量对照 faiss.Kmeans（±5% 验收线）
  - 结论直接决定 C++ 侧是否把质心更新也下沉到设备

## 第二轮落地（2026-11-13 晚）

### 1. FlDeviceOps 扩展 + Host 适配器（无卡可测全链路）
- `FlDeviceOps` 新增 `ivf_flat_scan`（候选集扫描 kernel 入口，cand_ids/cand_offsets 前缀数组接口）
- `src/common/fl/fl_host_adapter.{h,cc}`：host 内存模拟设备 + OpenMP 实现全部算子，三大用途——UT 无卡验证设备路径、CPU-only CI、芯片适配器作者的语义参考实现
- `RegisterHostAdapter()` / `UnregisterHostAdapter()`

### 2. PPU 适配器骨架
- `src/common/fl/adapter/ppu/fl_ppu_adapter.cc`：实现 `FlDeviceOps` 的 PPU 版本，预留 FlagFit launch 封装集成点（hgMemAlloc/hgMemcpyH2D/hgAotKernelLoad/hgAotLaunchTopK 等 extern "C" 接口）；`KNOWHERE_FL_WITH_PPU` 宏保护，无 SDK 时编译为空 TU；`ivf_flat_scan` 暂返回 -1 走 CPU 回退（Phase 1.3 待 ivf_scan kernel）

### 3. IVF_FLAT-FL 索引（Phase 1.2/1.3 v1）
- `src/index/fl/fl_ivf_flat_config.h`：nlist/nprobe 参数对齐官方语义，COSINE 按 D4 拒绝
- `src/index/fl/fl_ivf_flat_node.h`：
  - Train：CPU Lloyd kmeans（25 轮，OpenMP，空簇重播种）——方案 5.2 备案路径，设备 kmeans 待 PPU 原子微基准
  - Search：质心路由（host，nq×nlist）→ 候选合并（前缀数组）→ 设备 `ivf_flat_scan` 或 CPU 回退扫描；bitset 前置（候选过滤，第七轮起不再后置）
  - 完整序列化（header + 质心 + 原始向量 + 倒排表）
- `src/index/fl/fl_ivf_flat.cc`：双名注册 `GPU_IVF_FLAT` / `FL_IVF_FLAT`

### 4. 单元测试（Catch2，自动 glob 收集）
- `tests/ut/test_fl_index.cc`（`#ifdef KNOWHERE_WITH_FL` 保护，无卡可跑）：
  - Host 适配器注册/反注册
  - BRUTE_FORCE：CPU 回退路径 recall@10=1.0（对照精确 KNN）、设备路径（host 模拟）recall@10=1.0
  - OOM 预检（1 字节上限）→ 回退 CPU 结果仍精确（D5 验证）
  - 序列化/反序列化往返
  - COSINE 拒绝（D4 验证）
  - IVF_FLAT：nprobe=nlist 时精确、nprobe<nlist 时 recall≥0.8

## 第一轮落地（2026-11-13 下午）

### 1. 构建系统（CMake）
- `knowhere/CMakeLists.txt`
  - 新增 `WITH_FL` 编译开关（默认 OFF）
  - 与 `WITH_GPU/WITH_CUVS/WITH_CARDINAL` 互斥（设计 D2，FATAL_ERROR 防护）
  - `WITH_FL=ON` 时定义 `KNOWHERE_WITH_FL`；OFF 时从构建中剔除 `src/common/fl/*.cc`、`src/index/fl/*.cc`

### 2. 设备抽象层（方案第 4 章）
- `src/common/fl/fl_device.h` / `fl_device.cc`
  - `FlDeviceOps` C-ABI 函数表：device_count / set_device / malloc / free / memcpy(H2D/D2H/D2D) / stream / `brute_force_topk` kernel 入口
  - `RegisterDeviceOps()` 注册点——**PPU 适配器挂载处**（对接 FlagFit 已验证的 hgLaunchKernelEx/HIDDEN_PARAM/context 绑定链路）
  - `FlMemoryPool`：对齐官方 `initMemSize/maxMemSize` 语义，新增 OOM 前置校验 `Ensure()`（设计 D5，修复官方路径缺陷；第七轮起改为原子 `TryReserve()`）
  - `FlDeviceBuffer` RAII 设备内存封装

### 3. GPU_BRUTE_FORCE-FL 索引（方案 Phase 1.1）
- `src/index/fl/fl_brute_force_config.h`：metric 校验 L2/IP；COSINE 按设计 D4 拒绝并提示归一化+IP
- `src/index/fl/fl_index_node.h`：`FlBruteForceIndexNode<fp32>`
  - 设备路径：数据入库即 staging 到设备显存（带 OOM 预检，失败回退 CPU）；Search 走 `brute_force_topk` kernel；bitset 经 over-fetch 精确过滤（第七轮起）
  - CPU 回退：OpenMP 暴力搜索（仅用于正确性 bring-up 与降级，非性能路径）
  - Serialize/Deserialize：header(dim/nb/metric) + 原始向量
  - RangeSearch 显式不支持（与官方 GPU 索引一致）
- `src/index/fl/fl_brute_force.cc`：双名注册 `GPU_BRUTE_FORCE`（用户面，CUDA 路径互斥排除）+ `FL_BRUTE_FORCE`（调试别名），并发度随设备数伸缩

### 4. 上游接线
- `include/knowhere/comp/index_param.h`：新增 `INDEX_FL_BRUTEFORCE = "FL_BRUTE_FORCE"`
- `include/knowhere/index/index_table.h`：新增 `{INDEX_FL_BRUTEFORCE, VECTOR_FLOAT}` 合法项
- `include/knowhere/expected.h`：Status 枚举尾部追加 `fl_device_error = 35`（保持上游合并友好；第七轮补齐 StatusCategoryOf/Status2String 覆盖）

### 5. Phase 0 微基准脚本（FlagOS 算子库优先的实证）
- `python/fl_kernels/distance_topk.py`
  - 距离计算纯 FlagGems 组合：`mm` + 平方和归约（L2 的 GEMM 化分解）/ `mm`（IP），top-k 直接 `torch.topk`
  - 对照 faiss CPU 暴力搜索（召回金标准），输出 recall@k、P50、QPS 到 JSON
  - 在远程 PPU 环境运行，产出算子支持度清单数据

---

# 第二部分：Milvus 集成指南

> 目标：在国产 GPU（首发 PPU/ZW810E）上以 `GPU_BRUTE_FORCE` / `GPU_IVF_FLAT` / `GPU_IVF_PQ`
> 原索引名运行 Milvus，用户面零改动。（对应方案 Phase 1.5/1.6）

## 0. 前置条件

| 项 | 要求 |
|---|---|
| Milvus 版本 | v2.4+（knowhere 以 conan 包形式被 internal/core 消费） |
| knowhere-FL 构建 | 本仓库 `cmake -DWITH_FL=ON [-DWITH_FL_PPU=ON -DPPU_SDK_PATH=...]` |
| 芯片适配器 | PPU：`WITH_FL_PPU=ON` + PPU 运行时 SDK；其他芯片：实现 `FlDeviceOps` 并注册 |
| 无卡验证 | 不装任何适配器即为 Host 回退模式（功能正确、性能为 CPU 档） |

## 1. 集成路径（二选一）

### 路径 A：conan 本地包覆盖（推荐，贴近生产）

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

### 路径 B：源码级覆盖（开发联调用）

Milvus internal/core 支持以本地 knowhere 源码树替代 conan 包。将本仓库
`knowhere/` 目录软链或拷贝到 Milvus 的 knowhere 第三方源码路径，并在
`internal/core/CMakeLists.txt` 的 knowhere 段加 `-DWITH_FL=ON` 传入。
联调期改动 knowhere 无需重新打 conan 包。

## 2. 芯片适配器注册（关键一步）

knowhere 只提供 `RegisterDeviceOps` 挂载点，**适配器必须在 Milvus 进程启动早期注册**：

- **PPU**：`WITH_FL_PPU=ON` 构建时，`fl_ppu_adapter.cc` 中的
  `RegisterPpuAdapter()` 需被调用一次。推荐在 knowhere 的 FL 初始化函数
  （`src/common/fl/fl_device.cc` 中暴露的惰性注册）或 Milvus segcore 初始化处调用。
- **无适配器**：索引仍可建/查，全部走 CPU 回退路径（D5），性能等同 CPU 索引。
- **Host 适配器**（`RegisterHostAdapter()`）：仅用于 CI/UT，不要在生产注册。

## 3. Milvus 侧配置

milvus.yaml 中与 FL 相关的键（复用官方 GPU 语义）：

```yaml
gpu:
  initMemSize: 0        # FL 显存池初始预留（字节）；0 = 首次使用时按可用显存 1/2
  maxMemSize: 0         # 显存池上限；0 = 全部可用显存。超过即触发 OOM 预检回退 CPU（D5）
```

建索引参数与官方 GPU 索引完全一致：

```json
{"index_type": "GPU_IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
{"index_type": "GPU_IVF_PQ",   "metric_type": "L2", "params": {"nlist": 1024, "m": 16, "nbits": 8}}
{"index_type": "GPU_BRUTE_FORCE", "metric_type": "IP", "params": {}}
```

约束（与官方一致 + FL 特有）：
- GPU_IVF_FLAT / GPU_BRUTE_FORCE 支持 L2 / IP；GPU_IVF_PQ 仅 L2（朴素点积
  ADC 无法分解 IP 距离，v1 起 Train 直接拒绝并给出指引，勿配 IP）
- COSINE 请先归一化后用 IP（D4，建索引时会收到明确报错提示）
- 不支持 range search / grouping search
- IVF_PQ v1 仅 nbits=8 且 dim 需整除 m；不存原始向量（HasRawData=false，
  影响 query 时 output_fields 直接回向量——需要该能力请用 GPU_IVF_FLAT）
- bitset 过滤语义：GPU_BRUTE_FORCE / GPU_IVF_FLAT / GPU_IVF_PQ 均为精确
  过滤（满 k；BRUTE_FORCE 设备路径内部 over-fetch 保证）；FL_CAGRA 为
  后置过滤，极端过滤比例下可能返回少于 k 条有效结果（与官方 GPU_CAGRA
  直接拒绝过滤查询相比是降级而非报错）

## 4. 验收清单（Phase 1.6 出口标准）

| # | 检查项 | 通过标准 |
|---|---|---|
| 1 | Milvus Standalone 启动，querynode 日志出现 `FL PPU adapter registered` | 适配器注册成功 |
| 2 | GPU_IVF_FLAT 建 100 万向量索引 | 成功且显存占用 ≤ 数据量 ×1.2 |
| 3 | 批量搜索（nq=1000, k=10） | recall@10 vs CPU faiss ≥ 0.95，QPS ≥ CPU HNSW 3× |
| 4 | 显存池 cap 为 1 字节重启 | 自动 CPU 回退，结果正确，无崩溃（D5） |
| 5 | 拔卡/禁用适配器重启 | 功能正确（CPU 档性能），无 fl_device_error 崩溃 |
| 6 | 序列化（compaction/加载）往返 | recall 不变，格式版本头校验生效 |

## 5. 已知限制（v1）

- 单 segment 绑定单卡（FlDeviceRouter 轮询分派跨 segment 摊多卡；segment 内不分片）
- PQ 为非残差实现，召回略低于 faiss 残差 PQ（后续版本升级，格式版本头已预留）
- 训练采样策略未实现（全量训练）；千万级以上建议先抽样训练再全量 Add

---

# 第三部分：构建、待办与设计决策

## 构建方式（远程 Linux/PPU 环境）

```bash
cd knowhere
mkdir build && cd build
cmake .. -DWITH_FL=ON -DCMAKE_BUILD_TYPE=Release   # 与 WITH_GPU 互斥
# PPU 真机适配器（可选）：
# cmake .. -DWITH_FL=ON -DWITH_FL_PPU=ON -DPPU_SDK_PATH=/opt/ppu_sdk
make -j
# UT（CPU 回退路径先行，无需 PPU）
```

## 待办（按方案计划）

| 优先级 | 事项 | 对应方案条目 | 状态 |
|---|---|---|---|
| P0 | 远程 PPU 环境跑 `distance_topk.py`：验证 FlagGems `mm/topk/norm` 在 FlagTree PPU 后端的可用性、精度、微基准 | Phase 0.1 / M0 | 脚本就绪，待执行 |
| P0 | PPU 适配器实现 `FlDeviceOps`（复用 FlagFit launch 封装） | Phase 0.3 | ✅ 骨架已写，待远程填 AOT kernel 加载 |
| P1 | Linux 全量编译 `WITH_FL=ON` + 跑通 UT（第七轮已清除全部已知编译阻断） | Phase 0.2 | UT 就绪，待远程编译 |
| P1 | kmeans 组合算子（`mm`+`argmin`+`index_add`）微基准，实测 PPU 原子性能 | Phase 1.2 | 脚本就绪（kmeans_composition.py），待远程执行；C++ 混合模式（设备路由+CPU 更新）已落地 |
| P1 | IVF_FLAT-FL 节点 | Phase 1.3 | ✅ v1 已落地（设备 kmeans_assign 路由 + 双路径扫描 + use_device_assign 开关） |
| P2 | `ivf_scan` 融合算子（KernelGen 生成基线→人工优化→贡献 FlagGems） | Phase 1.3 | 微基准脚本就绪（ivf_scan_composition.py），待远程实测裁决 |
| P2 | IVF_PQ-FL 索引 | Phase 2.1 | ✅ v1 已落地（CPU 训练 + 设备/CPU 双路径 ADC，nbits=8，L2-only） |
| P2 | FL_CAGRA 图索引 | Phase 2.3 | ✅ v1 已落地（CPU 构图 + 设备/CPU 遍历 + adapt_for_cpu 模式）；设备构图/graph_search kernel 待远程 |
| P2 | Milvus Standalone 整机集成 + 基准报告 | Phase 1.5/1.6 | 集成指南见第二部分，待远程 |

## 设计决策速查

- D1 FlagOS 全系算子库优先（五层选型纪律）
- D2 独立后端、编译期与 CUDA 互斥、用户面沿用 GPU_* 名称
- D3 与 FlagFit/vLLM-plugin-FL 统一算子版本基线
- D4 COSINE = 归一化 + IP（同官方 GPU 约束）
- D5 OOM 前置校验 + CPU 回退（修复官方缺陷；第七轮起预留原子化）
