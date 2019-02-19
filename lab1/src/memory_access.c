/*************************************************************************
  > File Name:       memory_access.c
  > Author:          Zeyuan Hu
  > Mail:            iamzeyuanhu@utexas.edu
  > Created Time:    02/14/19
  > Description:


 ************************************************************************/

#define _GNU_SOURCE

#include <asm/unistd.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define CACHE_LINE_SIZE 64
#define TIMEVAL_TO_DOUBLE(x) ((double)x.tv_sec + (double)x.tv_usec / 1000000.0)

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int opt_random_access;
int cpu_id;
char buffer[1 << 24]; // 16MB

long x = 1, y = 4, z = 7, w = 13;
long simplerand(void) {
  long t = x;
  t ^= t << 11;
  t ^= t >> 8;
  x = y;
  y = z;
  z = w;
  w ^= w >> 19;
  w ^= t;
  return w;
}

long get_mem_size() {
  FILE *f = fopen("/proc/meminfo", "r");
  long tmp;
  char buf[101];
  fscanf(f, "%s %ld", buf, &tmp);
  tmp *= 1024;
  fclose(f);
  return tmp;
}

int compete_for_memory(void *unused) {
  long mem_size = get_mem_size();
  int page_sz = sysconf(_SC_PAGE_SIZE);
  printf("Total memsize is %3.2f GBs\n",
         (double)mem_size / (1024 * 1024 * 1024));
  fflush(stdout);
  char *p = mmap(NULL, mem_size, PROT_READ | PROT_WRITE,
                 MAP_NORESERVE | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    perror("Failed anon MMAP competition");
    fflush(stdout);
    exit(EXIT_FAILURE);
  }

  int i = 0;
  while (1) {
    volatile char *a;
    long r = simplerand() % (mem_size / page_sz);
    char c;
    if (i >= mem_size / page_sz) {
      i = 0;
    }
    a = p + r * page_sz;
    c += *a;
    if (i % 8 == 0) {
      *a = 1;
    }
    i++;
  }
  return 0;
}

// Access a working set consists of 512 consecutive cache lines.
// 1/8 of accesses is write and 7/8 of accesses is read. The same
// working set is used 16 times. The whole routine will work on
// 2^{20} working sets.
// Working sets are selected either sequentially or in random. In sequential
// access case, offsets between working sets is 512 (e.g., since there are 512
// cache lines of a set). In random access case, working set is selected
// randomly based on the number from `simplerand()`.
void do_mem_access(char *p, int size) {
  int outer, locality, i;
  int ws_base = 0;
  int max_base = size / CACHE_LINE_SIZE - 512;
  for (outer = 0; outer < (1 << 20); outer++) {
    long r = simplerand() % max_base;
    if (opt_random_access) {
      ws_base = r;
    } else {
      ws_base += 512;
      if (ws_base >= max_base) {
        ws_base = 0;
      }
    }
    for (locality = 0; locality < 16; locality++) {
      volatile char *a;
      char c;
      for (i = 0; i < 512; i++) {
        a = p + ws_base + i * CACHE_LINE_SIZE;
        if (i % 8 == 0) {
          *a = 1;
        } else {
          c = *a;
        }
      }
    }
  }
}

int hw_cache_perf_event_open(int group_fd, int cache_id, int cache_op_id,
                             int cache_op_result_id) {
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(struct perf_event_attr));
  pe.type = PERF_TYPE_HW_CACHE;
  pe.size = sizeof(struct perf_event_attr);
  pe.config = cache_id | (cache_op_id << 8) | (cache_op_result_id << 16);
  pe.disabled = 0;
  if (group_fd == -1) {
    pe.disabled = 1;
  }
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  int fd = perf_event_open(&pe, 0, cpu_id, group_fd, 0);
  if (fd == -1) {
    perror("perf_event_open");
    exit(EXIT_FAILURE);
  }
  return fd;
}

int main(int argc, char **argv) {
  if (argc != 7) {
    fprintf(stderr, "Needs 6 arguments: cpu_id opt_random_access mmap_type "
                    "mmap_populate init_memset bg_compete\n");
    exit(EXIT_FAILURE);
  }

  int bg_compete = atoi(argv[6]);
  int child_pid = -1;
  if (bg_compete == 1) {
    child_pid = fork();
    if (child_pid == -1) {
      perror("fork");
      exit(EXIT_FAILURE);
    }
    if (child_pid == 0) {
      compete_for_memory(NULL);
      return 0;
    }
    sleep(5);
  }

  opt_random_access = atoi(argv[2]);

  cpu_id = atoi(argv[1]);
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu_id, &set);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &set) == -1) {
    perror("sched_setaffinity");
    exit(EXIT_FAILURE);
  }

  int l1_read_access_fd = hw_cache_perf_event_open(
      -1, PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,
      PERF_COUNT_HW_CACHE_RESULT_ACCESS);
  int leader_fd = l1_read_access_fd;

  int l1_read_miss_fd = hw_cache_perf_event_open(
      leader_fd, PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,
      PERF_COUNT_HW_CACHE_RESULT_MISS);
  int l1_write_access_fd = hw_cache_perf_event_open(
      leader_fd, PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_WRITE,
      PERF_COUNT_HW_CACHE_RESULT_ACCESS);
  int tlb_read_miss_fd = hw_cache_perf_event_open(
      leader_fd, PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ,
      PERF_COUNT_HW_CACHE_RESULT_MISS);
  int tlb_write_miss_fd = hw_cache_perf_event_open(
      leader_fd, PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_WRITE,
      PERF_COUNT_HW_CACHE_RESULT_MISS);

  // Attempt to flush L1 cache
  int i, n = sizeof(buffer);
  for (i = 0; i < n; i++) {
    buffer[i] = i & 255;
  }
  char tmp;
  volatile char *pc = &tmp;
  for (i = 0; i < n; i++) {
    *pc = buffer[i];
  }

  int mmap_type = atoi(argv[3]);
  int mmap_populate = atoi(argv[4]);
  int init_memset = atoi(argv[5]);

  int size = 1 << 30; // 1GB
  int fd = -1;
  char *p;

  if (mmap_type > 0) {
    fd = open("tmp", O_RDWR);
    if (fd == -1) {
      perror("open");
      exit(EXIT_FAILURE);
    }
  }

  int flags = 0;
  if (mmap_type == 0) {
    flags = MAP_PRIVATE | MAP_ANONYMOUS;
  } else if (mmap_type == 1) {
    flags = MAP_PRIVATE;
  } else if (mmap_type == 2) {
    flags = MAP_SHARED;
  }
  if (mmap_populate == 1) {
    flags |= MAP_POPULATE;
  }

  p = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, fd, 0);
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(EXIT_FAILURE);
  }

  if (init_memset == 1) {
    memset(p, 0, size);
    if (mmap_type == 2) {
      if (msync(p, size, MS_SYNC) == -1) {
        perror("msync");
        exit(EXIT_FAILURE);
      }
    }
  }

  struct rusage before_usage;
  if (getrusage(RUSAGE_SELF, &before_usage) != 0) {
    perror("getrusage");
    return 0;
  }
  printf("[Before do_mem_access]\n");
  printf("utime = %ld.%06ld s\n", before_usage.ru_utime.tv_sec,
         before_usage.ru_utime.tv_usec);
  printf("stime = %ld.%06ld s\n", before_usage.ru_stime.tv_sec,
         before_usage.ru_stime.tv_usec);
  printf("maxrss = %ld KB\n", before_usage.ru_maxrss);
  printf("minflt = %ld\n", before_usage.ru_minflt);
  printf("majflt = %ld\n", before_usage.ru_majflt);
  printf("inblock = %ld\n", before_usage.ru_inblock);
  printf("oublock = %ld\n", before_usage.ru_oublock);

  ioctl(leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

  // routine we want to analyze
  do_mem_access(p, size);

  ioctl(leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

  struct rusage after_usage;
  if (getrusage(RUSAGE_SELF, &after_usage) != 0) {
    perror("getrusage");
    return 0;
  }
  printf("[After do_mem_access]\n");
  printf("utime = %ld.%06ld s\n", after_usage.ru_utime.tv_sec,
         after_usage.ru_utime.tv_usec);
  printf("stime = %ld.%06ld s\n", after_usage.ru_stime.tv_sec,
         after_usage.ru_stime.tv_usec);
  printf("maxrss = %ld KB\n", after_usage.ru_maxrss);
  printf("minflt = %ld\n", after_usage.ru_minflt);
  printf("majflt = %ld\n", after_usage.ru_majflt);
  printf("inblock = %ld\n", after_usage.ru_inblock);
  printf("oublock = %ld\n", after_usage.ru_oublock);

  printf("[Delta]\n");
  double d_utime = TIMEVAL_TO_DOUBLE(after_usage.ru_utime) -
                   TIMEVAL_TO_DOUBLE(before_usage.ru_utime);
  double d_stime = TIMEVAL_TO_DOUBLE(after_usage.ru_stime) -
                   TIMEVAL_TO_DOUBLE(before_usage.ru_stime);
  printf("utime = %.6f s\n", d_utime);
  printf("stime = %.6f s\n", d_stime);
  printf("minflt = %ld\n", after_usage.ru_minflt - before_usage.ru_minflt);
  printf("majflt = %ld\n", after_usage.ru_majflt - before_usage.ru_majflt);
  printf("inblock = %ld\n", after_usage.ru_inblock - before_usage.ru_inblock);
  printf("oublock = %ld\n", after_usage.ru_oublock - before_usage.ru_oublock);
  printf("maxrss = %ld\n", after_usage.ru_maxrss - before_usage.ru_maxrss);

  if (munmap(p, size) != 0) {
    perror("munmap");
    exit(EXIT_FAILURE);
  }
  if (fd != -1) {
    close(fd);
  }

  uint64_t l1_read_miss = 0;
  uint64_t l1_read_access = 0;
  uint64_t l1_write_access = 0;
  uint64_t tlb_read_miss = 0;
  uint64_t tlb_write_miss = 0;

  read(l1_read_access_fd, &l1_read_access, sizeof(uint64_t));
  read(l1_read_miss_fd, &l1_read_miss, sizeof(uint64_t));
  read(l1_write_access_fd, &l1_write_access, sizeof(uint64_t));
  read(tlb_read_miss_fd, &tlb_read_miss, sizeof(uint64_t));
  read(tlb_write_miss_fd, &tlb_write_miss, sizeof(uint64_t));

  close(l1_read_access_fd);
  close(l1_read_miss_fd);
  close(l1_write_access_fd);
  close(tlb_read_miss_fd);
  close(tlb_write_miss_fd);

  printf("[Performance counters]\n");
  printf("Data L1 read access: %" PRIu64 "\n", l1_read_access);
  printf("Data L1 write access: %" PRIu64 "\n", l1_write_access);
  printf("Data L1 read miss: %" PRIu64 "\n", l1_read_miss);
  printf("Data L1 read miss rate: %.5f\n",
         (double)l1_read_miss / l1_read_access);

  printf("Data TLB read miss: %" PRIu64 "\n", tlb_read_miss);
  printf("Data TLB write miss: %" PRIu64 "\n", tlb_write_miss);
  fflush(stdout);

  if (bg_compete == 1) {
    kill(child_pid, SIGKILL);
    wait(NULL);
  }

  return 0;
}
