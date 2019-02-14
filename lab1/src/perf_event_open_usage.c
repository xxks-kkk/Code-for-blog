/*************************************************************************
  > File Name:       perf_event_open_usage.c
  > Author:          Zeyuan Hu
  > Mail:            iamzeyuanhu@utexas.edu
  > Created Time:    02/14/19
  > Description:

    Demonstrate the usage of `perf_event_open` interface to measure
    L1 data cache read, write, read misses for printf(). printf() can be
    replaced as any work you want to measure.

    Note, there may exist some extra work need to be done for measuring
    L1 data cache (e.g., flush L1 data cache first). This file simply for
    demonstration of `perf_event_open` interface usage.

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

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int cpu_id;

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

  ioctl(leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);

  // Do the work that we want to analyze
  printf("Do some work that we want to measure here\n");

  ioctl(leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

  uint64_t l1_read_miss = 0;
  uint64_t l1_read_access = 0;
  uint64_t l1_write_access = 0;

  read(l1_read_access_fd, &l1_read_access, sizeof(uint64_t));
  read(l1_read_miss_fd, &l1_read_miss, sizeof(uint64_t));
  read(l1_write_access_fd, &l1_write_access, sizeof(uint64_t));

  close(l1_read_access_fd);
  close(l1_read_miss_fd);
  close(l1_write_access_fd);

  printf("[Performance counters]\n");
  printf("Data L1 read access: %" PRIu64 "\n", l1_read_access);
  printf("Data L1 write access: %" PRIu64 "\n", l1_write_access);
  printf("Data L1 read miss: %" PRIu64 "\n", l1_read_miss);
  printf("Data L1 read miss rate: %.5f\n",
         (double)l1_read_miss / l1_read_access);

  fflush(stdout);

  return 0;
}
