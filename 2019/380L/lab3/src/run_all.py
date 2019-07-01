#!/usr/bin/python2.7

import os
import subprocess as sp

output_dir = '../res'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
run_times = 10

bench_progs = ['bench_seq', 'bench_matrix', 'bench_revseq', 'bench_revmatrix', 'bench_random']
pager_progs = {
    'apager': ['./apager'],
    'dpager': ['./dpager'],
    'hpager_pf0': ['./hpager', '0'],
    'hpager_pf1': ['./hpager', '1'],
    'hpager_pf2': ['./hpager', '2']
}

sp.call('make clean && make release bench', shell=True)

for bench_prog in bench_progs:
    for pager_prog in pager_progs:
        sp.call([
            './measure_time.py', str(run_times),
            os.path.join(output_dir, '%s_%s.txt' % (bench_prog, pager_prog))
        ] + pager_progs[pager_prog] + [bench_prog])
