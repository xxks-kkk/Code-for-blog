#!/bin/bash

python3 run.py "./task2 0 1 0 1 0 0 0" 5 > results/no_compete_anonymous_random_dl1.txt
python3 run.py "./task2 0 1 0 1 0 1 0" 5 > results/no_compete_anonymous_random_dtlb.txt
python3 run.py "./task2 0 1 0 1 0 0 1" 5 > results/compete_anonymous_random_dl1.txt
python3 run.py "./task2 0 1 0 1 0 1 1" 5 > results/compete_anonymous_random_dtlb.txt

python3 run.py "./task2 0 1 1 1 0 0 0" 5 > results/no_compete_private_random_dl1.txt
python3 run.py "./task2 0 1 1 1 0 1 0" 5 > results/no_compete_private_random_dtlb.txt
python3 run.py "./task2 0 1 1 1 0 0 1" 5 > results/compete_private_random_dl1.txt
python3 run.py "./task2 0 1 1 1 0 1 1" 5 > results/compete_private_random_dtlb.txt

python3 run.py "./task2 0 1 2 1 0 0 0" 5 > results/no_compete_shared_random_dl1.txt
python3 run.py "./task2 0 1 2 1 0 1 0" 5 > results/no_compete_shared_random_dtlb.txt
python3 run.py "./task2 0 1 2 1 0 0 1" 5 > results/compete_shared_random_dl1.txt
python3 run.py "./task2 0 1 2 1 0 1 1" 5 > results/compete_shared_random_dtlb.txt