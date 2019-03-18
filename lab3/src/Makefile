CC = musl-gcc
LOADER_CFLAGS = -static -Wall -Wl,--script=loader.lds
TEST_PROGS_CFLAGS = -static -O0 -g
TEST_PROGS_LDFLAGS = -lpthread
BENCH_PROGS_CFLAGS = -static -O2

LOADER_DEPS = alloc.o elf.o stack.o
LOADERS = apager dpager hpager
TEST_PROGS = test_helloworld \
	test_args_env \
	test_bss \
	test_malloc \
	test_time \
	test_pthread
BENCH_PROGS = bench_seq \
	bench_revseq \
	bench_matrix \
	bench_revmatrix \
	bench_random

all: release test bench

release: $(LOADERS)
debug: $(LOADERS)
test: $(TEST_PROGS)
bench: $(BENCH_PROGS)

release: LOADER_CFLAGS += -O2
debug: LOADER_CFLAGS += -DPRINT_LOG -g

$(LOADER_DEPS): %.o: %.c
	$(CC) $(LOADER_CFLAGS) -o $@ -c $<

$(LOADERS): % : %.c $(LOADER_DEPS)
	$(CC) $(LOADER_CFLAGS) -o $@ $^

$(TEST_PROGS): % : %.c
	$(CC) $(TEST_PROGS_CFLAGS) -o $@ $^ $(TEST_PROGS_LDFLAGS)

$(BENCH_PROGS): % : %.c
	$(CC) $(BENCH_PROGS_CFLAGS) -o $@ $^

clean:
	rm -f *.o $(LOADERS) $(TEST_PROGS) $(BENCH_PROGS)
