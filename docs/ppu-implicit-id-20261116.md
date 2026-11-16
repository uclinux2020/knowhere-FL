# PPU 隐式 ID 与请求端到端优化：2026-11-16

## 结论

本轮目标是端到端超越历史 CUDA-C 10倍，**实测尚未达到**。已实施并验证有效优化：不再写入/读取brute-force距离表中的重复全局ID，降低工作区与带宽开销；调优warp、score tile及选择tile。保持统一Triton（结合TLE）→FlagTree→原生HG→PPU，不引入PyTorch，不新增CUDA-C生产算子。

同device0、dim128/k10/L2/nq200/seed42、CPU0–7/8线程，预热2次、每项15次请求计时：

| nb | CUDA-C查询请求 ms | 优化查询请求 ms | 加速比 |
|---|---:|---:|---:|
| 100k | 6.678 | 2.541 | 2.63x |
| 500k | 27.722 | 10.846 | 2.56x |
| 1M | 52.240 | 21.404 | 2.44x |

使用**默认64MiB工作区预算**，不靠增加到1GiB取得成绩。查询请求边界包含query设备分配、query H2D、Search内部工作区/launch/sync、结果D2H、query释放；index已驻留，AOT及服务RPC/调度不在此边界。没有完整Milvus端到端验证，不把这里称为服务全链路。

更宽的“每次请求重新分配并上传全量base/query”边界（百万向量，7次样本）：CUDA-C96.061ms，新实现62.952ms，**1.53x**。base约512MB上传成本共同限制加速，不能把resident Search比例当成一次性全输入端到端比例。

所有测量的ordered ID/score mismatch=0，same-representation recall=1；FP32累加顺序并非位级一致。原始完整样本及失败调优配置见 [JSON](ppu-implicit-id-results-20261116.json)。本轮基于`634e9b06`加未提交修改；设备Driver1.3.2-d7f5a2/HGGC13.0、GNU13.3、FlagTree PPU Triton3.6沿用前轮。

## 优化点

brute-force完整距离表中每条query的ID都是0..nb-1，重复物化没有必要。新增`--implicit-ids`：

- matrix/direct batched distance只写float score，不写每candidate的int64 ID。
- 第一层`select_implicit`直接由列位置生成全局ID；后续compact top-k仍携带实际ID，正确处理并列分数。
- 每query完整score表从12*nb字节降为4*nb字节；compact两级缓冲仍按12字节/槽计入预算。
- runtime按真实新布局计算batch容量；旧manifest与IVF/PQ不使用隐式ID，候选重排和PQ数据语义保持不变。
- 不缓存数据结果、不降低精度、不削减扫描候选，不以缓存返回制造加速。

百万/nq200，64MiB预算旧矩阵配置查询请求77.507ms；仅隐式ID（w4/b64/select1024）的resident Search31.364ms；w2降至25.057ms；select2048降至23.383ms；score block32最终查询请求21.404ms。该配置相对旧查询请求约3.62x。

调优消融也保留：w8为40.531ms，score block128查询请求49.309ms，均不采用。最优测量配置是num_warps2、score block32、select block2048、16-query IEEE FP32 matrix tile。参数仍显式设置，不将一个shape的最优参数强制应用所有索引。

## 精度、边界与验证

- 128维L2/IP、ties、尾query mask、大公共偏置、IVF/PQ、codebook更新、assign、empty/fewer-than-k、invalid CSR真机contract通过。
- 直接距离隐式ID配置与matrix+implicit最终组合分别测试；PQ仍走原有显式IDs/LUT，未改变其表示。
- `test_retrieval_triton.py --implicit-ids`为非PQ bundle生成隐式ID算子，PQ分支不加该选项。
- 不完整隐式IDmanifest缺少第一层selector或batched distance会明确拒绝，而不是读取未初始化ID缓冲。
- Python语法、C++独立编译、`git diff --check`通过；完整Knowhere/Milvus仍受既有Folly ABI问题阻塞。

## 复现

```bash
export PPU_SDK=/usr/local/PPU_SDK PPU_HOME=/usr/local/PPU_SDK FLAGTREE_BACKEND=ppu
export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD
export LD_LIBRARY_PATH=/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
python python/fl_kernels/compile_ppu.py --tle --matrix --implicit-ids --num-warps 2 \
  --queries-per-block 16 --block 32 --select-block 2048 --dim 128 --k 10 \
  --output build/ppu-implicit/kernels/d128_k10_l2_m0_s0
export KNOWHERE_FL_PPU_KERNEL_DIR="$PWD/build/ppu-implicit/kernels"
export KNOWHERE_FL_PPU_WORKSPACE_MB=64
export KNOWHERE_PPU_QUERY_E2E=1
build/ppu-contract/knowhere_ppu_bench build/ppu-implicit/request.json \
  bf 1000000 200 128 10 L2 1 random 15 8 0
```

比较历史产物时单独进程设置`KNOWHERE_PPU_LEGACY_CUBIN=/root/topk.cubin`。全输入模式改用`KNOWHERE_PPU_FULL_INPUT_E2E=1`。两个计时模式不包含数据生成、AOT、冷模块加载及RPC；报告JSON明确记录所用边界，不改变既有resident计时含义。

## 为什么未到10x及下一步

百万查询请求需低于约5.22ms才能达到10x，当前21.4ms还需约4.1x改善；全输入请求需低于约9.61ms，而本次设备上传已造成数十ms成本，不能只优化kernel实现这个边界的10x。后续应结合真实服务index驻留生命周期，测量query H2D、设备计算、top-k、D2H各阶段，优先研究低物化分层选择、批量持久tile及受预算约束的工作区复用，而非混淆端到端定义。

未测：生产服务E2E、真实数据near-tie质量、所有dim/k/filter、冷缓存/冷AOT、并发多卡、index构建/Add/serialization耗时和峰值RSS/显存。JSON的RSS仍为基准进程生命周期高水位，不是建索引峰值。未承诺10x；代码和分析尚未提交推送。
