C_COMPILER = gcc
CFLAGS = -Wall -g -O2 -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -std=c11 -DDEBUG
MAIN_OUTPUT_FILE = barfs
BENCHMARK_OUTPUT_FILE = benchmark
MAIN_SRC_FILES = barfs.c
BENCHMARK_SRC_FILES = benchmark.c
LIBS= -lfuse -lssh -pthread

.PHONY: barfs
.PHONY: clean
.PHONY: benchmark

barfs:
	$(C_COMPILER) -o $(MAIN_OUTPUT_FILE) $(CFLAGS) $(MAIN_SRC_FILES) $(LIBS)

clean:
	rm barfs
	rm benchmark

benchmark:
	$(C_COMPILER) -o $(BENCHMARK_OUTPUT_FILE) $(CFLAGS) $(BENCHMARK_SRC_FILES)

