C_COMPILER = gcc
CFLAGS = -Wall -g -O2 -D_FILE_OFFSET_BITS=64 -D_GNU_SOURCE -std=c11
MAIN_OUTPUT_FILE = main
MAIN_SRC_FILES = main.c
LIBS= -lfuse -lssh -pthread

.PHONY: main
.PHONY: clean

main:
	$(C_COMPILER) -o $(MAIN_OUTPUT_FILE) $(CFLAGS) $(MAIN_SRC_FILES) $(LIBS)

clean:
	rm main

