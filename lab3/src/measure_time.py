#!/usr/bin/python2.7

import sys
import subprocess as sp

import numpy as np

assert len(sys.argv) > 3

run_times = int(sys.argv[1])
save_file = sys.argv[2]
args = sys.argv[3:]

real_time, user_time, sys_time = [], [], []
for i in xrange(run_times):
    p = sp.Popen(['/usr/bin/time', '-f', '%e %U %S'] + args, stderr=sp.PIPE)
    _, stderrdata = p.communicate()
    parts = str(stderrdata).split()
    real_time.append(float(parts[0]))
    user_time.append(float(parts[1]))
    sys_time.append(float(parts[2]))

with open(save_file, 'w') as f:
    f.write('real: %.3f s (std %.3f)\n' % (np.mean(real_time), np.std(real_time)))
    f.write('user: %.3f s (std %.3f)\n' % (np.mean(user_time), np.std(user_time)))
    f.write('sys: %.3f s (std %.3f)\n' % (np.mean(sys_time), np.std(sys_time)))
