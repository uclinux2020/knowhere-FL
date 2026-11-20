# k1 矩阵分配与完整构建实验（2026-11-20）

## 范围

使用当前候选节点/同一完整库二进制，仅替换独立k1 AOT bundle。没有更改线上Milvus、没有覆盖既有bundle，不更改默认配置。统一链路为Triton/TLE → FlagTree hgbin → native HG → PPU，未使用Torch/ATen/CUDA-C。

新root `/root/knowhere-k1-matrix-20261120/kernels`，d128/k1/L2/m0，matrix、implicit IDs、warps2、QPB16、block32、select2048、TLE。对照为 `build/ppu-implicit/final/kernels` 的k1 direct/QPB4/implicit配置，不是之前无implicit的direct bundle，故不能跨实验直接比较倍数。

## 完整 Train + Add 配对

同seed42随机输入，CPU affinity0–7、OMP8/PASSIVE/PROC_BIND、物理卡1、workspace64MiB/cache0；每shape direct/matrix/matrix/direct，各block3个fresh process，每版6样本。无profiling性能轮；每次25次train assign+1次Add assign，失败中止不接受CPU fallback。输入生成与Serialize不在Train+Add计时内。

下表每版为两个block p50均值，ms：

| NB | nlist | direct Train+Add | matrix Train+Add | direct/matrix |
|---:|---:|---:|---:|---:|
| 100k | 128 | 289.778 | 138.148 | 2.10 |
| 1M | 128 | 3300.960 | 1500.820 | 2.20 |
| 100k | 1024 | 1579.620 | 309.139 | 5.11 |

1M/nlist128 Train assign回调中位数1870.225→369.178ms；callback包含runtime launch/wait/D2H，不是纯kernel计时。各shape仅6样本/版，不作p95或统计显著性宣称，也不是create-index RPC/flagfit全链路成绩。

## 数值与质量边界

新增 `tests/ppu/test_ppu_assign_matrix.cc`，可显式传入物理device编号，在当前native runtime上验证L2最近质心ID与double直接距离oracle一致：nlist31/128/1024，nq1/17/257，随机、完全平分、10000 common-offset，共27组合；实际设备1通过，无CPU fallback。

新增独立CMake目标 `knowhere_ppu_assign_matrix_test`，配置/构建成功。不自动加入默认CTest，因为必须由执行者显式选择设备与匹配的k1 bundle。

**训练产物并非全部逐位一致**：每个variant内重复产物稳定；100k/1M nlist128的direct与matrix序列化hash不同，100k/nlist1024两者hash相同。矩阵浮点累加方式可能改变近似平分assignment并沿Lloyd迭代传播；随机固定质心oracle通过不等同整轮训练、最终recall无损。

因此当前收益只称“构建速度候选”，**未通过同召回/质量验收，不能默认推广或宣布无损加速**。下一步需要对两种新训练artifact分别检查实际nprobe候选、recall@10和分数、全探针精确性，至少质量目标0.90/0.95；不能拿上一轮固定artifact的RPC正确性证明新bundle训练质量。

独立proof-on 100k/nlist1024已保存设备事件；与性能轮分开。初次oracle链接因系统gflags不匹配失败；独立runtime编译先尝试C++17时遇std::osyncstream编译错误，改项目要求的C++20后编译执行成功，未伪造符号或改依赖。

## 新训练产物的实际召回验证

扩展 `bench_fl_ivf_add.cc quality`：独立seed43的100个query、L2/k10，对原始向量用double直接距离计算精确GT（计时外）。每种bundle重新完成Train/Add，再扫探针，2预热+7样本；每次检查native callback执行、唯一/合法ID、double oracle分数与全探针精确性。两root的k10 hgbin完全相同，仅k1训练bundle不同；没有CPU fallback。

| NB / nlist | nprobe | direct recall | matrix recall | matrix逐query命中减少数/100 |
|---|---:|---:|---:|---:|
| 100k /128 | 16 | 0.383 | 0.407 | 22 |
| 100k /128 | 64 | 0.833 | 0.832 | 26 |
| 100k /128 | 96 | 0.958 | 0.963 | 8 |
| 1M /128 | 16 | 0.417 | 0.425 | 12 |
| 1M /128 | 64 | 0.848 | 0.849 | 12 |
| 1M /128 | 96 | 0.964 | 0.964 | 3 |
| 100k /1024 | 16 | 0.134 | 0.134 | 0 |
| 100k /1024 | 64 | 0.351 | 0.351 | 0 |
| 100k /1024 | 512 | 0.892 | 0.892 | 0 |
| 100k /1024 | 768 | 0.981 | 0.981 | 0 |

所有shape全探针recall1，两版GT一致。100k/nlist1024训练artifact hash一致且各probe有序结果一致；nlist128训练artifact改变，**并非逐query无损**，100k/probe64整体recall也减少0.001（0.1个百分点）。不能将整体平均持平/提高称为所有query不下降。

质量目标0.90/0.95：两种nlist128 artifact均在probe96达到≥0.95；nlist1024均在probe768达到≥0.95。不存在这一固定输入上的质量目标失败，但这不是普遍质量保证，未声明或强加生产容忍阈值。

查询性能未明显改善：1M/nlist128/probe96 p50为87.542→87.835ms，两种artifact同k10配置；全探针11.527→11.558ms/recall1，反而快于高probe。每点7样本，不支持tail-latency结论，不是RPC。按0.90/0.95目标在已测网格选最快达标点，两种bundle均选全探针快路径，不能据此声称新k1改变Search速度。

**推进建议**：k1矩阵bundle保留opt-in。100k/nlist1024是本轮同时满足artifact/召回一致、构建约5.11倍加速的限定配置；可优先纳入隔离Milvus重建配对，再补1M/nlist1024。nlist128若要求每query严格不下降则不合格，应保留direct或先获得明确质量容忍准则。IP、过滤、大k、其他维度/真实数据、多segment、生产并发仍未覆盖。

产物 `quality-<nb>-<nlist>-<variant>.json/log`、run_quality.py、quality-build日志。质量测试初次脚本内联shell引号导致语法错误，未执行构建；改独立Python脚本后编译与测试全部完成。未将失败当设备或性能样本。

## 百万 / nlist1024 补测与服务重建验收

补测每种bundle各1次新Train/Add和100个相同query（每probe2预热+7检查样本），仅方向/质量验证，不作稳定加速倍数验收：

| 项目 | direct | matrix |
|---|---:|---:|
| Train ms | 15227.700 | 3198.240 |
| Add ms | 743.419 | 268.357 |
| Train+Add ms | 15971.200 | 3466.590 |
| recall/probe16 | 0.165 | 0.164 |
| recall/probe64 | 0.389 | 0.378 |
| recall/probe512 | 0.916 | 0.921 |
| recall/probe768 | 0.981 | 0.982 |
| recall/全探针 | 1.000 | 1.000 |

构建单次约4.61倍改善，但两版artifact hash不同。probe64整体召回下降1.1个百分点、32/100个query减少命中；probe512/768整体略提高，仍有14/7个query减少命中。故100k/nlist1024的artifact一致结论不能推广到1M；若要求严格逐query无下降，1M也不合格。两版都达到≥0.90/0.95质量目标，全探针精确性和每次native callback/分数检查通过。

服务级构建验收已准备并在已授权隔离范围内运行：`run_milvus_build_pair.py`，临时19540/metrics9093、物理卡1、CPU0–7/OMP8，同candidate library，仅direct/matrix k1 root不同、k10相同；direct-A/matrix-A/matrix-B/direct-B，每轮新prefix创建100k/1M的CPU FLAT与FL IVF集合。不删除/覆写任何旧集合，不替换在线库，结束只停本次PID，数据保留。

每轮新建artifact，不假设segment边界相同，也不要求不同artifact有序ID相同。记录实际segment布局、recall、每query命中和create_index_complete_s（含排队/RPC/状态轮询，不能称纯Train/Add），每probe7个Search样本，仅用于质量/趋势验证。正式构建轮profile-off；已有独立矩阵proof提供执行证据，不将不同边界的事件当成新RPC直接证据。未默认部署；1M低probe质量下降必须在最终结论保留。

当前新服务实验 `/root/knowhere-k1-build-rpc-1795139198716896414/` 的direct-A/matrix-A均完成：100k create-index完成7.793→6.025s；1M 20.066→8.574s。仅第一对，不作最终ABBA结论。100k各probe recall一致；1M probe64 recall0.402→0.386、probe512 0.918→0.909、probe768 0.988→0.980，质量下降需保留。新collections的segment划分不固定，不能把变化全部归因于矩阵数值。候选B/基线B尚在执行。

服务产物指针 `/root/knowhere-ppu-build-opt-20261119/k1-build-rpc-path.txt`，日志 `/root/knowhere-k1-matrix-20261120/milvus-build-pair.log`。空间预检/root约50GiB可用，本轮创建约8.8M条输入（含CPU GT），不清理历史数据腾空间。

## 服务级重建ABBA最终结果

四轮direct-A/matrix-A/matrix-B/direct-B均完成，所有输入/query hash、完整工作负载、每样本分数/ID检查和全探针GT校验通过。每版每规模2次独立新建collection；7次查询样本不等于7次构建。实际服务顶层ERROR=0、fallback提及=0、profile事件=0，证明正式轮profiling未启用；设备执行证据来自此前独立proof，不宣称本轮新增RPC硬件trace。

| NB | direct create-index完成均值s | matrix均值s | direct/matrix |
|---:|---:|---:|---:|
| 100k | 7.747 | 6.291 | 1.23 |
| 1M | 19.940 | 9.133 | 2.18 |

该指标包含RPC/队列/训练/添加/持久化/约0.5s状态轮询，不是纯Train/Add。CPU FLAT控制项100k direct均值1.622s、matrix1.261s，1M direct1.009s、matrix1.510s，说明轮询粒度/服务背景有变动；每版仅两次构建，不声称统计显著或普遍固定倍数。百万在两个matrix轮均为8.57–9.69s，相应direct为19.81–20.07s，是本次服务级改善方向证据。

100k所有探针四轮召回完全相同（probe512=0.912，768=0.988，全探针1）。1M四轮：

| nprobe | direct-A | matrix-A | matrix-B | direct-B |
|---:|---:|---:|---:|---:|
| 64 | 0.402 | 0.386 | 0.378 | 0.391 |
| 512 | 0.918 | 0.909 | 0.914 | 0.937 |
| 768 | 0.988 | 0.980 | 0.989 | 0.990 |
| 1024 | 1.000 | 1.000 | 1.000 | 1.000 |

matrix百万probe64两轮低于两个direct轮；其他点也存在下降。新collection实际segment布局不同，direct自身也变化，因此不能分离数值差异与分段影响，更不能称百万召回无损。四轮均在probe512达到≥0.90、probe768达到≥0.95；按网格中最快达标点选取可能为全探针BF快路径，不代表必须采用同probe。未授予默认部署或生产召回容忍阈值，matrix仍opt-in。

**最终判定**：兑现了本轮条件下约2.18倍百万索引创建完成时间改善，达到“百万建索引约2倍”方向目标；不确认flagfit应用全链路查询加速、不确认逐query无损，也不将构建加速泛化为生产吞吐/尾延迟。下一步若继续，应固定同一segment输入分别训练并测真实业务数据，决定可接受的质量准则后再考虑默认配置；或为严格一致需求验证专用direct-distance top1融合，避免矩阵累加顺序改变。

临时进程全部退出，在线Milvus PID2434541仍运行，数据和旧证据保留。结果 `/root/knowhere-k1-build-rpc-1795139198716896414/build-pair-summary.json`，包含每轮构建、recall/p50/p95、segment和质量目标达标点；未提交或推送。

## 产物与复现

目录 `/root/knowhere-k1-matrix-20261120/`：compile.log、manifest/hgbin、run_pair.py、results.json、各fresh-process日志、proof.log、oracle测试和CMake构建。

```bash
PPU_SDK=/usr/local/PPU_SDK FLAGTREE_BACKEND=ppu \
  /root/flagos22-venv/bin/python python/fl_kernels/compile_ppu.py \
  --dim 128 --k 1 --tle --matrix --implicit-ids --num-warps 2 \
  --queries-per-block 16 --block 32 --select-block 2048 \
  --output <NEW_ROOT>/d128_k1_l2_m0_s0

UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD \
  LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib KNOWHERE_FL_PPU_KERNEL_DIR=<NEW_ROOT> \
  /root/knowhere-k1-matrix-20261120/test-build/knowhere_ppu_assign_matrix_test 1
```

当前root只含L2/d128/k1，不可直接拿去替换包含k10检索的Milvus完整root；实际服务需独立新root中的所有必需specialization及指纹。测试完成资源释放，未替换在线库、未提交或推送。
