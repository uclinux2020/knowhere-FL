# IEEE FP32 矩阵距离优化及超越历史 CUDA-C 的限定验收

## 结果与适用范围

本轮在统一的 `Triton（结合Triton-TLE）→ FlagTree AOT → 原生 HG → ZW810E` 路径实现可选矩阵距离配置。**在 dim=128、k=10、L2、nq=200、10万/50万/100万向量、显式1024MiB工作区预算的实测中，两轮配对测试均快于历史 CUDA-C 产物。** 不能扩展为任意数据、任意参数、任意预算或完整Milvus端到端都更快。

没有引入PyTorch或CUDA-C生产算子。历史CUDA-C只作为比较用产物。FlagOS是多算子库生态；本次是自实现Triton/TLE算子经FlagTree编译，不称为FlagGems算子。

| nb | CUDA-C p50 ms，两轮 | 本轮 p50 ms，两轮 | 配对加速比 |
|---|---:|---:|---:|
| 100k | 6.666 / 6.699 | 3.820 / 3.818 | 1.745 / 1.755 |
| 500k | 27.662 / 27.631 | 17.797 / 17.819 | 1.554 / 1.551 |
| 1M | 52.135 / 52.252 | 33.411 / 33.446 | 1.560 / 1.562 |

每次2次预热、15次计时；第二轮逆转执行顺序；CPU固定0–7、8线程，seed42，device0；CPU oracle在计时外。全部2000个结果槽核对，ordered ID/score mismatch=0，same-representation recall=1。最大绝对分数误差约2.29e-5；并非位级相同浮点累加。IEEE FP32输入，不用TF32/FP16降低精度换速度。

原始JSON包含每个样本：[ppu-matrix-results-20261116.json](ppu-matrix-results-20261116.json)。设备仍为Driver1.3.2-d7f5a2、HGGC13.0、GNU13.3、FlagTree PPU Triton3.6。历史产物 `/root/topk.cubin` SHA256：`370ca8109a94df959b3656defe1fcc9a169d48660ec21ef9d629f038328ebe9f`。当前源码基于`1d889664`加本轮未提交变更。

## 为什么更快

- 旧配置每program直接距离只复用4条query；新配置16条query共享base tile，以`tl.dot(..., input_precision='ieee')`计算矩阵cross，再由norm组合得到L2。
- 候选tile调至64，实测优于128/256。32-query tile与较大candidate tile均有退化，没有默认采用。
- 同stream多级top-k仍保留，最终结果D2H包括在Search计时；不是只比较kernel时间。
- query加载实际使用TLE扩展；base大tile仍用普通Triton加载，避免共享存储膨胀。

未使用设备profiler证明逐指令原因；上述tile收益以配置消融及真实计时为证据。

## 数值稳定性和保守启用

矩阵L2使用`||q||²+||b||²-2q·b`，对大公共偏置可能发生消减误差。本轮新增大偏置10000附近的真机回归，并在首query anchor绝对值超过1024时同时平移query和base，保持数学L2不变；IP不平移。当前PPU编译器对跨branch的async tensor有dominance错误，改为分支内加载解决，未修改第三方SDK。

**这不保证任意浮点分布下与直接差分距离完全相同。** 首query anchor不能覆盖所有query/base的极端分布，near-tie仍可能受累加顺序影响；所以矩阵实现保持`--matrix`显式启用，不替换默认直接FP32距离。已验证随机seed42、另一个seed7、ties、尾query mask、大公共偏置回归。完整128维contract test还验证L2/IP、IVF、PQ、assign、empty/fewer-than-k与非法CSR；矩阵只用于brute-force/assign的设备距离分支，不把PQ成绩归因于它。

## 工作区取舍必须披露

默认runtime预算仍64MiB，不偷偷改为1GiB。百万规模、nq200、本轮配置：

- 64MiB独立复测：77.537ms，**仍慢于CUDA-C约52ms**。
- 256MiB独立复测：38.837ms，快于CUDA-C，但不是本轮两轮配对正式验收的预算。
- 1024MiB两轮配对：约33.4ms，正式表格使用该预算。

预算为score/compact工作区目标，candidate CSR另计；不是显存峰值采样。base输入约512MB另外驻留，预算不足容纳一query时明确失败。当前workspace未接入全局FlMemoryPool，因此不承诺生产多请求的总显存限额。

## 竞争样本和失败实验

早期配对期间另启动过本会话的PPU预算测试，造成500k第二轮28.999ms落后于CUDA-C27.765ms；验收脚本正确返回失败。这组失败样本已保存在原始JSON，不能删除失败后声称无波动。停止本会话其他PPU测试后重新独立执行，正反序全部配对通过。设备2–5有既存作业，未操作；没有独占整机保证。

还试验了排序key融合、矩阵局部top-k融合及tf32x3多乘积；排序编译长时间未完成后停止，局部融合跨规模不稳定、tf32x3未改善。未将失败实验并入生产代码，最终只保留IEEE矩阵配置。

## 编译及复现

从仓库根目录，使用已安装FlagTree PPU Python与原生SDK：

```bash
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK FLAGTREE_BACKEND=ppu
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}

python python/fl_kernels/compile_ppu.py --tle --matrix \
  --queries-per-block 16 --block 64 --dim 128 --k 10 \
  --output build/ppu-matrix/kernels/d128_k10_l2_m0_s0
export KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-matrix/kernels"
export KNOWHERE_FL_PPU_WORKSPACE_MB=1024

python python/fl_kernels/compare_ppu_cuda.py --output build/ppu-matrix/acceptance \
  --executable build/ppu-contract/knowhere_ppu_bench --legacy-artifact /root/topk.cubin
```

`compare_ppu_cuda.py`重新编译完整矩阵bundle，按100k/500k/1M正反序配对，核对正确性，任意配对未领先则非零退出。不要同时运行其他性能任务。比较边界：resident base/query，包含workspace alloc/free、launch/sync、result D2H；排除初始base/query H2D、AOT、数据生成及CPU oracle。

主机基准现记录workspace、kernel目录、seed；支持`KNOWHERE_PPU_BENCH_SEED`用于正确性扩展。目录名称含torch的已有Python环境不代表运行导入Torch；编译与比较脚本阻止导入，验收`torch_imported=false`。

## 未完成项

完整Knowhere/Milvus仍受既有Folly/milvus-common F14 ABI问题阻塞；没有端到端服务性能、真实生产向量near-tie质量、大k、多维、多卡并发、冷缓存与RSS/显存峰值实测。生产默认仍为直接差分距离配置。此次是目标工作负载下、显式预算及配置的超越，不是无限制性能保证。代码及报告尚未提交推送。
