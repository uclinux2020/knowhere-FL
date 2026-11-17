# Folly 链接修复与 Milvus 实际 nprobe sweep（2026-11-17）

## 结论

- Folly F14 链接阻塞已解决，完整 Knowhere/Catch2 二进制完成构建；`[fl]` 25用例、2435断言全部通过，`[ivf-nprobe]`专项也通过。
- Milvus真实RPC在100k/1M、128维、NQ100、k10、nlist128上完成1/4/16/128探针扫描；稳定集合下两版有序结果全部相同，逐query recall无下降，全探针recall=1。
- 真实FL扫描未退化为无条件全表扫描，所有观测 `device_done=1`，无CPU fallback日志。
- **不确认普遍端到端加速**：稳定配对下大部分结果接近持平，部分请求回退；第一轮数据存在后台compaction/重建和系统扰动，不能将其明显加速当作验收成绩。

## Folly 根因及修复

原Knowhere完整构建依赖 `/root/.conan2/p/b/folly764e4e4d01f84/p` 只导出F14IntrinsicsMode=1（Simd），而全局 `-msse4.2` 的Knowhere与预编译milvus-common需要mode=2（SimdAndCrc）。曾尝试移除全局SSE4.2；Knowhere自身mode一致后，milvus-common仍要求mode2，故撤回，不改变项目ISA基线。

当前缓存recipe已包含SSE4.2修复，但旧二进制仍被复用；`conan install --build='folly/*' --no-remote` 因缺失liburing/2.8无法完整解析。经用户授权，使用旧Folly包的**同源源码和同依赖生成器**在仓库build下新建构建树，显式 `-msse4.2`，安装到 `build/folly-crc-install`，不覆盖缓存。`nm -DC`确认真实库导出mode2，未添加假符号或关闭ABI检查。

复制现有CMakeDeps生成器到 `build/folly-crc-conan-overlay`，只将folly package root指向新安装目录，以 `-Dfolly_DIR=$PWD/build/folly-crc-conan-overlay`配置完整库。运行时还需优先使用Conan gflags库，系统同SONAME的gflags与当前glog符号不兼容。

复现关键命令（路径是本次现场环境）：

```bash
cmake -S /root/.conan2/p/b/folly764e4e4d01f84/b/src -B build/folly-crc \
  -DCMAKE_TOOLCHAIN_FILE=/root/.conan2/p/b/folly764e4e4d01f84/b/build/Release/generators/conan_toolchain.cmake \
  -DCMAKE_PREFIX_PATH=/root/.conan2/p/b/folly764e4e4d01f84/b/build/Release/generators \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-m64 -msse4.2' \
  -DCMAKE_C_FLAGS='-m64 -msse4.2' -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/folly-crc-install"
cmake --build build/folly-crc -j6
cmake --install build/folly-crc
PPU_SDK_PATH=/usr/local/PPU_SDK cmake -S . -B build/full-validation/Release \
  -Dfolly_DIR="$PWD/build/folly-crc-conan-overlay"
PPU_SDK_PATH=/usr/local/PPU_SDK cmake --build build/full-validation/Release -j8
LD_LIBRARY_PATH=/root/.conan2/p/b/gflag50f7aa0dff15d/p/lib:$PWD/build/folly-crc-install/lib:$PWD/build/full-validation/Release:/usr/local/PPU_SDK/lib \
  build/full-validation/Release/tests/ut/knowhere_tests '[fl]'
```

链接完成后揭示现有FL测试夹具问题：native PPU懒注册使Host adapter注册失败；两处bitset错误地按每ID一个byte设置而非packed bits。修正测试显式注册隔离、初始化顺序和位图编码后 `[fl]` 全通过。Host adapter测试不是PPU性能证据。

## 实际服务与配对方法

不停止已有PID520187的Milvus实例，不替换其库。新实例使用19540端口、独立etcd/minio root、独立localStorage/rocksmq路径。复用同一个Milvus Go/core可执行文件，动态加载两版Knowhere：基线1983a8dc、候选fbcca97e加`FL_IVF_SCAN`观测；两版均使用 `/root/milvus/cmake_build/conan` 的相同依赖、相同编译选项与DiskANN开关。该Milvus依赖本已有CRC模式Folly，不使用Knowhere UT的隔离Folly。

AOT固定 `build/ppu-implicit/final/kernels`，Triton/Triton-TLE → FlagTree → native HG → PPU；64MiB预算。不是历史CUDA-C/shim，也不是FlagGems算子。

脚本 `tests/ppu/milvus_nprobe_sweep.py`：seed42，100queries×128d，L2/k10，四个独立collection，nlist128，探针1/4/16/128，预热2次、计时7次。query.tolist在计时外；计时含pymilvus RPC序列化、服务执行、D2H、返回反序列化。FLAT作为相同数据的精确GT。初次prepare创建collection后保留，不覆盖或删除，以便跨revision reload。

第一轮candidate创建索引后测量，随后baseline reload期间后台compaction导致百万集合的近似结果和CPU控制项变化，第一轮配对废弃。追加baseline-stable→candidate-stable两轮，所有query/探针有序ID完全一致，作为当前限定结果；未禁止后台compaction或固定CPU亲和，仍有系统变动风险。未做足够长时间重复实验来证明统计显著性。

## 稳定配对RPC p50（ms/batch）

| NB | 路径 | nprobe | 旧runtime | 候选runtime | 旧/新 |
|---|---|---:|---:|---:|---:|
| 100k | CPU FLAT | — | 14.816 | 17.265 | 0.858 |
| 100k | CPU IVF | 16 | 6.331 | 6.321 | 1.002 |
| 100k | FL BF | — | 6.503 | 6.226 | 1.044 |
| 100k | FL IVF | 1 | 9.965 | 14.051 | 0.709 |
| 100k | FL IVF | 4 | 14.460 | 14.608 | 0.990 |
| 100k | FL IVF | 16 | 16.337 | 15.610 | 1.047 |
| 100k | FL IVF | 128 | 52.299 | 41.134 | 1.271 |
| 1M | CPU FLAT | — | 131.544 | 140.342 | 0.937 |
| 1M | CPU IVF | 16 | 25.673 | 27.385 | 0.937 |
| 1M | FL BF | — | 17.260 | 16.652 | 1.037 |
| 1M | FL IVF | 1 | 17.321 | 13.138 | 1.318 |
| 1M | FL IVF | 4 | 28.138 | 27.334 | 1.029 |
| 1M | FL IVF | 16 | 66.382 | 71.364 | 0.930 |
| 1M | FL IVF | 128 | 561.780 | 544.142 | 1.032 |

稳定FL IVF recall：100k=0.034/0.139/0.403/1.0，1M=0.052/0.197/0.462/1.0；两版一致，无逐query下降。CPU IVF使用独立训练，不假设候选/recall与FL完全相同。候选BF与同轮CPU FLAT的p50比为2.77×/8.43×，该结果包含此前矩阵/隐式ID优化，**不能归因于本轮CSR改动**。

## 实际候选扫描

初轮完整观测样本（每query均值，NQ100）：

| NB | nprobe1 | nprobe4 | nprobe16 | 全探针 |
|---|---:|---:|---:|---:|
| 100k | 783.63 | 3127.21 | 12495.24 | 100000 |
| 1M | 7818.23 | 31276.11 | 125026.20 | 1000000 |

百万实际为5个segment：100000、220000、220000、224032、235968，分配到多张卡；候选数必须按所有segment累加，不能将单次runtime nb当作集合NB。局部nlist均128，扫描比例接近nprobe/nlist；只有nprobe128按设计全扫描。所有FL日志device_done=1，无CPU fallback；CPU Faiss实际扫描计数未加探针，不能称已完成CPU/PPU候选计数一一对照。

CSR预算改动主要降低私有显存，真实Milvus的接近均匀倒排列表没有复现独立runtime不均衡场景的11%普遍速度收益。全探针下候选ID在host物化和H2D仍很重，多segment、多卡及host路由开销与裸kernel边界不同。

## 产物与剩余工作

JSON、samples、每query hit/ordered IDs、服务日志、构建日志保存 `build/ppu-csr-results/` 和 `build/milvus-e2e-followup-run/`。本次新启动的两版测试实例已停止，原服务保留运行；新测试collection保留在独立命名空间便于复现。

未执行全仓库所有Catch2用例，只完成全二进制构建及全部FL用例。未测IP服务E2E、过滤、大k、固定88核对照、生产并发或建设峰值显存/RSS；未自动提交/推送改动。下一轮应冻结segment/compaction，固定CPU并发与卡路由，增加host candidate构建计时，再做ABBA重复配对；优先减轻host CSR物化和全探针数据传输，而不是继续假设退化为全表扫描或只调top-k kernel。
