#!/bin/bash
gcc -O2 memory_access.c -o memory_access
dd if=/dev/zero of=tmp bs=1 count=0 seek=2G

python3 run.py "./memory_access 0 1 0 1 0 0" 5 > results_compete/no_compete_anonymous_random.txt
python3 run.py "./memory_access 0 1 0 1 0 1" 5 > results_compete/compete_anonymous_random.txt

python3 run.py "./memory_access 0 1 1 1 0 0" 5 > results_compete/no_compete_private_random.txt
python3 run.py "./memory_access 0 1 1 1 0 1" 5 > results_compete/compete_private_random.txt

python3 run.py "./memory_access 0 1 2 1 0 0" 5 > results_compete/no_compete_shared_random.txt
python3 run.py "./memory_access 0 1 2 1 0 1" 5 > results_compete/compete_shared_random.txt
