#!/bin/bash
rm -rf /var/tmp/foofs/bench
mkdir /var/tmp/foofs/bench
gcc -o benchmark -O2 benchmark.c && ./benchmark /var/tmp/foofs/bench > foofs.txt

rm -rf ./bench
mkdir ./bench
./benchmark ./bench > nfs.txt
