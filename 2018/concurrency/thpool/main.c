/*
 * Demo:
 * - how to use "thpool.h"
 * - how to pass data into several threads
 */

#include "thpool.h"
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int NUM_TASK = 4;
int NUM_THREAD = 4;

typedef struct thread_data {
  int fd;
  int start_pos;
  int num_bytes;
} thread_data;

void doSomething(void *arg) {
  printf("thread %u\n", (int)pthread_self());
  thread_data *thdata = (thread_data *)(arg);
  char buf[] = "line1\n";
  printf("thdata->start_pos: %d\n", thdata->start_pos);
  pread(thdata->fd, buf, 6, thdata->start_pos);
  printf("buffer: %s\n", buf);
}

int main() {
  int fd = open("test_file", O_RDONLY);
  thread_data thdata[NUM_TASK];
  threadpool thpool = thpool_init(NUM_THREAD);
  for (int i = 0; i < NUM_TASK; i++) {
    thdata[i].fd = fd;
    thdata[i].start_pos = i * 6;
    printf("start_pos before thpool_add_work: %d\n", thdata[i].start_pos);
    thpool_add_work(thpool, (void *)doSomething, (void *)(&thdata[i]));
  }
  thpool_wait(thpool);
  thpool_destroy(thpool);
}
