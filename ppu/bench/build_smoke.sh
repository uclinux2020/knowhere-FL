#!/bin/bash
# 链接 fl_smoke v3：allow-shlib-undefined 放行 F14 ABI 差异
cd /root
source /root/knowhere-FL/Release/generators/conanbuild.sh
RL=""
for d in /root/.conan2/p/b/*/p/lib; do
  [ -d "$d" ] && RL="$RL -Wl,-rpath-link,$d -L$d"
done
rm -f /root/fl_smoke
g++ -O2 -std=c++20 -DGLOG_USE_GLOG_EXPORT /root/fl_smoke.cc -o /root/fl_smoke \
  -I/root/knowhere-FL/include \
  -L/root/knowhere-FL/Release -lknowhere \
  -Wl,--allow-shlib-undefined \
  $RL $LDFLAGS 2> /root/link4.log
echo "LINK_EXIT=$?"
head -8 /root/link4.log
ls -la /root/fl_smoke 2>/dev/null
file /root/fl_smoke 2>/dev/null
