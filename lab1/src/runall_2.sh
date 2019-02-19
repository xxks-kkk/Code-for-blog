#!/bin/bash
python3 run.py "./task2 0 0 0 0 0 0 0" 5 > results/anonymous_sequential_dl1.txt
python3 run.py "./task2 0 1 0 0 0 0 0" 5 > results/anonymous_random_dl1.txt

python3 run.py "./task2 0 0 0 0 0 1 0" 5 > results/anonymous_sequential_dtlb.txt
python3 run.py "./task2 0 1 0 0 0 1 0" 5 > results/anonymous_random_dtlb.txt

python3 run.py "./task2 0 0 1 0 0 0 0" 5 > results/file_private_sequential_dl1.txt
python3 run.py "./task2 0 0 2 0 0 0 0" 5 > results/file_shared_sequential_dl1.txt
python3 run.py "./task2 0 1 1 0 0 0 0" 5 > results/file_private_random_dl1.txt
python3 run.py "./task2 0 1 2 0 0 0 0" 5 > results/file_shared_random_dl1.txt

python3 run.py "./task2 0 0 1 0 0 1 0" 5 > results/file_private_sequential_dtlb.txt
python3 run.py "./task2 0 0 2 0 0 1 0" 5 > results/file_shared_sequential_dtlb.txt
python3 run.py "./task2 0 1 1 0 0 1 0" 5 > results/file_private_random_dtlb.txt
python3 run.py "./task2 0 1 2 0 0 1 0" 5 > results/file_shared_random_dtlb.txt

python3 run.py "./task2 0 0 1 1 0 0 0" 5 > results/file_private_populate_sequential_dl1.txt
python3 run.py "./task2 0 1 1 1 0 0 0" 5 > results/file_private_populate_random_dl1.txt
python3 run.py "./task2 0 0 2 1 0 0 0" 5 > results/file_shared_populate_sequential_dl1.txt
python3 run.py "./task2 0 1 2 1 0 0 0" 5 > results/file_shared_populate_random_dl1.txt

python3 run.py "./task2 0 0 1 1 0 1 0" 5 > results/file_private_populate_sequential_dtlb.txt
python3 run.py "./task2 0 1 1 1 0 1 0" 5 > results/file_private_populate_random_dtlb.txt
python3 run.py "./task2 0 0 2 1 0 1 0" 5 > results/file_shared_populate_sequential_dtlb.txt
python3 run.py "./task2 0 1 2 1 0 1 0" 5 > results/file_shared_populate_random_dtlb.txt

python3 run.py "./task2 0 0 2 0 1 0 0" 5 > results/file_shared_msync_sequential_dl1.txt
python3 run.py "./task2 0 1 2 0 1 0 0" 5 > results/file_shared_msync_random_dl1.txt

python3 run.py "./task2 0 0 2 0 1 1 0" 5 > results/file_shared_msync_sequential_dtlb.txt
python3 run.py "./task2 0 1 2 0 1 1 0" 5 > results/file_shared_msync_random_dtlb.txt