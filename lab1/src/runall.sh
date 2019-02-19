#!/bin/bash
gcc -g -Wall memory_access.c -o memory_access
dd if=/dev/zero of=tmp bs=1 count=0 seek=2G

python3 run.py "./memory_access 0 0 0 0 0 0" 5 > results/anonymous_sequential.txt
python3 run.py "./memory_access 0 1 0 0 0 0" 5 > results/anonymous_random.txt

python3 run.py "./memory_access 0 0 1 0 0 0" 5 > results/file_private_sequential.txt
python3 run.py "./memory_access 0 0 2 0 0 0" 5 > results/file_shared_sequential.txt
python3 run.py "./memory_access 0 1 1 0 0 0" 5 > results/file_private_random.txt
python3 run.py "./memory_access 0 1 2 0 0 0" 5 > results/file_shared_random.txt

python3 run.py "./memory_access 0 0 1 1 0 0" 5 > results/file_private_populate_sequential.txt
python3 run.py "./memory_access 0 1 1 1 0 0" 5 > results/file_private_populate_random.txt
python3 run.py "./memory_access 0 0 2 1 0 0" 5 > results/file_shared_populate_sequential.txt
python3 run.py "./memory_access 0 1 2 1 0 0" 5 > results/file_shared_populate_random.txt

python3 run.py "./memory_access 0 0 2 0 1 0" 5 > results/file_shared_msync_sequential.txt
python3 run.py "./memory_access 0 1 2 0 1 0" 5 > results/file_shared_msync_random.txt
