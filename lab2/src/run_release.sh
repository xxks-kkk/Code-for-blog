#!/bin/bash
rm -rf /dev/shm/foofs_cache
mkdir /dev/shm/foofs_cache
fusermount -u /var/tmp/foofs
rm -rf /var/tmp/foofs
mkdir /var/tmp/foofs
ssh zjia@mike-n-ike.cs.utexas.edu rm -rf /var/tmp/foofs
ssh zjia@mike-n-ike.cs.utexas.edu mkdir /var/tmp/foofs
gcc -o foofs -O2 -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE main.c -lfuse -lssh -pthread \
   && ./foofs zjia mike-n-ike.cs.utexas.edu /var/tmp/foofs /dev/shm/foofs_cache \
      /var/tmp/foofs -o default_permissions,auto_unmount
