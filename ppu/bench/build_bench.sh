#!/bin/bash
# 构建 fl_bench：与 build_smoke.sh 同环境 + OpenMP + pthread
cd /root
source /root/knowhere-FL/Release/generators/conanbuild.sh
RL=""
for d in /root/.conan2/p/b/*/p/lib; do
  [ -d "$d" ] && RL="$RL -Wl,-rpath-link,$d -L$d"
done
rm -f /root/fl_bench
g++ -O2 -std=c++20 -fopenmp -DGLOG_USE_GLOG_EXPORT /root/fl_bench.cc -o /root/fl_bench \
  -I/root/knowhere-FL/include \
  -L/root/knowhere-FL/Release -lknowhere \
  -Wl,--allow-shlib-undefined \
  $RL $LDFLAGS -lpthread 2> /root/link_bench.log
echo "LINK_EXIT=$?"
head -8 /root/link_bench.log
ls -la /root/fl_bench 2>/dev/null
