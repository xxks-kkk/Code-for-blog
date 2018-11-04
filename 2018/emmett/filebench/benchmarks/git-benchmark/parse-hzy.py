#!/bin/env/python3

import re

def process(filename):
    p = re.compile('write\(')
    with open(filename) as f:
        for line in f:
            if p.match(line):
                print(line)

if __name__ == "__main__":
    process("strace.txt")
