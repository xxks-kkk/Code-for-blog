
import sys
import re
import subprocess as sp
import numpy as np


def read_output(stream):
    results = []
    for line in stream:
        numbers = []
        for m in re.finditer(r"\b\d+(.\d+)?\b", line):
            numbers.append((float(m.group(0)), m.start(0), m.end(0)))
        results.append((line, numbers))
    return results


cmdline = sys.argv[1]
times = int(sys.argv[2])

results = []
for _ in range(times):
    p = sp.Popen(cmdline, shell=True, stdout=sp.PIPE)
    (stdout, stderr) = p.communicate()
    results.append(read_output(stdout.decode("utf-8").splitlines()))
template = results[0]
for line_index, line in enumerate(template):
    content = line[0]
    for i in reversed(range(len(line[1]))):
        numbers = list(map(lambda x: x[line_index][1][i][0], results))
        start, end = line[1][i][1], line[1][i][2]
        new_content = "%.3f (std %.3f)" % (np.mean(numbers), np.std(numbers))
        content = content[0:start] + new_content + content[end:]
    print(content)
