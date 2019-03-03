#!/bin/bash -vxe

if [ "$1" = "barfs" ]; then
    mode=barfs
elif [ "$1" = "nfs" ]; then
    mode=nfs
else
    mode=all
fi

barfs_bench() {
    echo "benchmark barfs"
    rm -rf /tmp/barfs/bench
    mkdir /tmp/barfs/bench
    make benchmark
    ./benchmark /tmp/barfs/bench > ../res/barfs.txt    
}

nfs_bench() {
    echo "benchmark NFS"
    # We assume NFS shared directory is mounted on /home/zeyuanhu/lab2_nfs
    ./benchmark /home/zeyuanhu/lab2_nfs > ../res/nfs.txt
}

if [ $mode = barfs ]; then
    barfs_bench
elif [ $mode = nfs ]; then
    nfs_bench
else
    barfs_bench
    nfs_bench
fi
