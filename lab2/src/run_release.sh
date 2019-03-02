#!/bin/bash -vxe

rm -rf /tmp/barfs_cache
mkdir /tmp/barfs_cache
if mount | grep /tmp/barfs > /dev/null; then
    fusermount -u /tmp/barfs
fi
rm -rf /tmp/barfs
mkdir /tmp/barfs
ssh zeyuanhu@192.168.1.120 rm -rf /tmp/barfs
ssh zeyuanhu@192.168.1.120 mkdir /tmp/barfs
make barfs
# arguments:
# 1. remote_user
# 2. remote_host
# 3. remote_path
# 4. local_cache_path
# 5. local_path (mount point; required by fuse_main())
./barfs zeyuanhu 192.168.1.120 /tmp/barfs /tmp/foofs_cache /tmp/barfs -o default_permissions,auto_unmount

# For debug purpose:
# you can add -d -s options to ./barfs like below
# ./barfs zeyuanhu 192.168.1.120 /tmp/barfs /tmp/foofs_cache /tmp/barfs -o default_permissions,auto_unmount -d -s
# More info: https://www.cs.hmc.edu/~geoff/classes/hmc.cs135.201001/homework/fuse/fuse_doc.html#compiling
