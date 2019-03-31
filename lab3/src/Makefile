CC = gcc
LOADER_CFLAGS = -static -Wall -Wl,--script=linker.lds -std=gnu11
TEST_PROGS_CFLAGS = -static -O0 -g -std=gnu11
TEST_PROGS_LDFLAGS = -lpthread
BENCH_PROGS_CFLAGS = -static -O2 -std=gnu11

LOADER_DEPS = alloc.o elf_file.o stack.o
LOADERS = apager dpager hpager
TEST_PROGS = test_helloworld \
	test_args_env \
	test_bss \
	test_malloc \
	test_time \
	test_pthread \
	test_helloworld_memory_map
BENCH_PROGS = bench_seq \
	bench_revseq \
	bench_matrix \
	bench_revmatrix \
	bench_random
ELF_TEST_DEPS = alloc.o elf_file.o
ELF_TEST = elf_test

all: release test bench elf_test

release: $(LOADERS)
debug: $(LOADERS)
test: $(TEST_PROGS)
bench: $(BENCH_PROGS)
elf_test: $(ELF_TEST)

release: LOADER_CFLAGS += -O2
debug: LOADER_CFLAGS += -DPRINT_LOG -g

$(LOADER_DEPS): %.o: %.c
	$(CC) $(LOADER_CFLAGS) -o $@ -c $<

$(LOADERS): % : %.c $(LOADER_DEPS)
	$(CC) $(LOADER_CFLAGS) -o $@ $^

# '$@' is a [automatic variable](https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html#Automatic-Variables), which is
# the filename of the target of the rule (e.g., test_helloworld). '$^' represents the names of all the prerequisites, with spaces between them.
# '%' is part of [pattern rules](https://www.gnu.org/software/make/manual/html_node/Pattern-Rules.html). '%' matches any nonempty substring, which other
# characters match only themselves (e.g., '%.c' as a pattern matches any file name that ends in '.c'.
# The following usage explains: for each target defined in TEST_PROGS, we use '%' to match such target and we use '%.c' to represent the corresponding
# source file name. For example, test_helloworld is part of TEST_PROGS, '%' matches 'test_helloworld' and '%.c" represents 'test_helloworld.c'
$(TEST_PROGS): % : %.c
	$(CC) $(TEST_PROGS_CFLAGS) -o $@ $^ $(TEST_PROGS_LDFLAGS)

$(BENCH_PROGS): % : %.c
	$(CC) $(BENCH_PROGS_CFLAGS) -o $@ $^

$(ELF_TEST_DEPS): %.o: %.c
	$(CC) $(TEST_PROGS_CFLAGS) -o $@ -c $<

$(ELF_TEST): % : %.c $(ELF_TEST_DEPS)
	$(CC) $(TEST_PROGS_CFLAGS) -o $@ $^ $(TEST_PROGS_LDFLAGS)

clean:
	rm -f *.o $(LOADERS) $(TEST_PROGS) $(BENCH_PROGS) $(ELF_TEST)
