# R2 执行链纠偏与 IVF 候选拷贝线程优化（2026-11-17）

## 1. 先纠正性能基线

输入报告为 `/tmp/milvus-ppu-e2e-验证报告-r2-20261117.md`。本轮核对其本地原始结果和日志，未修改该报告、未重启原 Milvus 服务。

R2 的数值有原始出处，但**不属于成功执行的 PPU 搜索**：

- `/root/milvus_run/milvus_stdout.log:3257–3258` 记录 native HG adapter 和八卡注册。
- 同日志 `3320–3325` 明确报 `KNOWHERE_FL_PPU_KERNEL_DIR is unset`；`3333–3338` 记录 BF `falling back to CPU`，`3565–3570` 记录 IVF `CPU fallback`。百万测试也有相同回退。
- 注册成功、没有 ERROR 级日志、返回正确 ID 都不能替代 kernel 执行证据。R2 脚本没有逐位核对浮点 score。
- CWD 中的 legacy `fl_kernels/brute_force_topk.ppuo` 不被当前 native Triton/TLE runtime 使用。当前 runtime 必须通过 `KNOWHERE_FL_PPU_KERNEL_DIR` 加载 specialization 的 manifest 和 `.hgbin`。
- BF 在 Add/Deserialize 时已将 xb 上传到 `dev_xb_`；Search 只上传 query。NQ100/DIM128 的 query 为 51.2KB，而不是每次上传 512MB base。R2 的“80%+ xb H2D”没有测量支持。
- 实际基线 SHA 是 `afa3db777a0a0d6270a0685f5cf8e646124f7ffe`，与 R2 报告所写完整 SHA 不一致。

因此不能用 R2 的 BF 288.49ms 计算本轮 PPU 加速倍数，也不能据此宣称 main 回归55%。当前真实成功执行链及服务阶段定位参见 [host/H2D定位](ppu-host-transfer-stages-20261117.md) 和 [已有按优先级优化](ppu-ordered-optimizations-20261117.md)。xb 常驻化和全探针无过滤跳过 CSR 已存在，不重复引入地址/collection 缓存。

## 2. 本轮实施

只修改无过滤 IVF_FLAT 的大 CSR 拷贝调度：

- 保留一次 resize、prefix offsets、query 不重叠范围及原有成员顺序。
- 保留1048576候选 ID 的并行阈值。
- 每次拷贝的 team 大小限制为 `min(nq, omp_get_max_threads(), 8)`；单线程时不启动并行 team。
- 不修改全局 OpenMP 配置、质心打分、probe comparator、nprobe、距离 kernel 或过滤路径。过滤 predicate 仍保持原单遍串行语义。

原因：此前真实百万/nprobe4、16 的 CSR 物化仍占明显时间。内存拷贝受带宽限制，88线程 team 的启动/竞争成本大于额外带宽收益。本轮先用固定输入隔离该阶段，再保留有配对证据的小改动；不贸然更换 Milvus 调度器或引入跨请求缓存。

8线程是该88核主机的限定策略，不宣称所有CPU/NUMA/并发配置都最优。

## 3. 固定输入 ABBA 测量

新增 `tests/ppu/bench_fl_ivf_host.cc`，通过真实 node 的 Deserialize/Search 和 host adapter 驱动路由/CSR，scan callback **故意不执行距离计算**。因此下表不是 PPU kernel 或 Milvus RPC 性能。

条件：128维，NQ100，nlist128，L2/k10，seed42，均匀交错倒排列表；固定序列化布局免除训练差异。每个进程预热2次、计时11次；同一 workload 按 baseline/candidate/candidate/baseline 串行运行，分别新建进程；基线 node header 为 `afa3db77`，candidate 为本轮工作树，公共链接库相同。没有并行跑 benchmark。未固定CPU affinity，后台主机负载和NUMA仍可能影响变化。

表中每格为两个进程各自 batch p50 的算术平均（ms），不是将样本混合后的 p50：

| OpenMP最大线程 | NB | nprobe | baseline | candidate | 说明 |
|---:|---:|---:|---:|---:|---|
| 88 | 100k | 4 | 0.941 | 0.963 | 小输入未改变调度，约2.3%波动，不宣称改善 |
| 88 | 1M | 1 | 1.331 | 1.325 | 低于并行阈值，基本持平 |
| 88 | 1M | 4 | 3.118 | 2.018 | 延迟下降35.3%，约1.54x |
| 88 | 1M | 16 | 15.347 | 10.530 | 延迟下降31.4%，约1.46x |
| 8 | 1M | 4 | 2.090 | 2.050 | 已限定线程，变化小 |
| 8 | 1M | 16 | 10.232 | 10.192 | 基本持平 |
| 1 | 1M | 4 | 3.151 | 3.189 | 约1.2%波动 |
| 1 | 1M | 16 | 14.495 | 14.385 | 基本持平 |

第一次 ABBA 也测到88线程设置下nprobe4/16改善，但nprobe16基线后半轮漂移更大；最终重新编译、保存全部原始样本的结果采用上表。原始产物为 `build/ppu-r2-followup/host_abba_final.json`，含stdout原始11次计时和stderr；初次结果为 `host_abba.json`。固定基线header和编译脚本也在该目录。

Host benchmark包含host query staging、结果分配和无计算callback；base初次host staging、输入生成、反序列化不在搜索计时内。它不返回有效KNN，不用于测recall。正确性由下面的完整回归独立验证。

## 4. 回归验证

- 完整库和 `knowhere_tests` 构建成功。
- `git diff --check` 通过；新增C++及修改区间按clang-format-18格式化，保留无关原有格式。`pre-commit` 未安装，未执行其hooks。
- `[fl]`：27个用例、3473条断言全部通过。
- 新增大CSR回归：固定倒排布局，nprobe5/6位于并行阈值两侧；OpenMP最大线程1/2/8/88下候选ID、offset及list/query顺序一致，候选数准确。原有L2/IP、过滤、ties、padding、全探针设备失败CPU回退测试保留。
- 原生PPU contract：2/2通过，包括L2/IP、IVF、PQ、CSR预算及流回归；runtime/kernel本轮未改动。
- 新增host benchmark源码并通过本机匹配依赖手动编译；尝试加入可选CMake目标时，运行遭遇静态/动态gRPC重复flags，故未保留该目标，不把它称为可直接运行的通用CMake集成。

实际遇到的环境失败也记录：初次测试动态加载glog/gflags不匹配；将Milvus依赖目录放在最前又误加载了旧libknowhere，产生两个旧语义失败。最终把**本次构建目录置于LD_LIBRARY_PATH首位**，然后加载匹配Milvus依赖，27用例全通过。没有伪造ABI符号或关闭检查。`/usr/local/bin/ctest` 的Python wrapper缺cmake模块，改用系统 `/usr/bin/ctest` 后2/2通过。

复现当前库回归：

```bash
/usr/bin/cmake --build build/full-validation/Release --target knowhere_tests -j8
LD_LIBRARY_PATH="$PWD/build/full-validation/Release:/root/milvus/internal/core/output/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}" \
  OMP_NUM_THREADS=8 build/full-validation/Release/tests/ut/knowhere_tests '[fl]'
LD_LIBRARY_PATH="/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}" \
  KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-kernels-tle" \
  /usr/bin/ctest --test-dir build/ppu-contract --output-on-failure
```

Host benchmark需完整依赖图和匹配动态库，不属于独立HG contract。本机编译命令由 `build/ppu-r2-followup/build_host_bench.py` 从现有compile_commands取flags，并只链接所需共享依赖，避免全量测试链接列表引入重复静态gRPC。该脚本含本机Conan缓存路径，不是可移植构建指南：

```bash
python3 -I build/ppu-r2-followup/build_host_bench.py
LD_LIBRARY_PATH="$PWD/build/full-validation/Release:/root/milvus/internal/core/output/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}" \
  OMP_NUM_THREADS=88 build/ppu-r2-followup/host_candidate 1000000 4 11
```

## 5. 验收边界与下一步

本轮未重启/替换19530原服务，未创建或删除collection，未提交或推送。没有新的Milvus RPC配对结果，不把host35%改善称为E2E35%改善；没有用R2 CPU回退耗时当基线。

未测：生产并发、NUMA亲和、非均匀list物化性能、更大nlist的路由性能、过滤性能矩阵、构建峰值RSS。此改动不改变训练/Add/存储布局、最终大小、候选数量或工作区大小；未测构建指标，不能宣称构建收益。

下一轮应先在隔离服务正确设置当前AOT根目录，并对每次计时窗口核查 `device_done=1`、native runtime事件及无CPU fallback。固定segment布局，配对nprobe1/4/16/full，保存候选数、route/materialize/H2D/scan和RPC原始样本。BF另需节点级成功/失败及query staging计时，不能只检查注册和ERROR数量。跨请求常驻倒排列表、probe-only上传的收益和生命周期需独立设计/验证，不属于本轮已完成项。
