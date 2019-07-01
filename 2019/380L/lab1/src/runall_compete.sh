#!/bin/bash
gcc -O2 memory_access.c -o memory_access
dd if=/dev/zero of=tmp bs=1 count=0 seek=2G

if [ "$1" = "m" ]; then
    save_dir=results_compete_modified
else
    save_dir=results_compete_no_modified
fi

echo "save_dir: $save_dir"
mkdir -p $save_dir

python3 run.py "./memory_access 0 1 0 1 0 0" 5 > $save_dir/no_compete_anonymous_random.txt
python3 run.py "./memory_access 0 1 0 1 0 1" 5 > $save_dir/compete_anonymous_random.txt

python3 run.py "./memory_access 0 1 1 1 0 0" 5 > $save_dir/no_compete_private_random.txt
python3 run.py "./memory_access 0 1 1 1 0 1" 5 > $save_dir/compete_private_random.txt

python3 run.py "./memory_access 0 1 2 1 0 0" 5 > $save_dir/no_compete_shared_random.txt
python3 run.py "./memory_access 0 1 2 1 0 1" 5 > $save_dir/compete_shared_random.txt
