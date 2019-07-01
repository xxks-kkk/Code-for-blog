#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

#define SIZE (1024 * 1024 * 256)

unsigned int a[SIZE];

int main() {
  srand(time(NULL));
  for (int i = 0; i < SIZE; i++) {
    a[i] = rand();
  }
  unsigned int result = 0;
  for (int i = 0; i < SIZE; i++) {
    result ^= a[i];
  }
  printf("result = 0x%x\n", result);
  struct rusage r_usage;
  getrusage(RUSAGE_SELF, &r_usage);
  // Print the maximum resident set size used (in kilobytes).
  printf("Memory usage: %ld kilobytes\n", r_usage.ru_maxrss);
  return 0;
}
