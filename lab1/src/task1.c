/*************************************************************************
  > File Name:       task1.c
  > Author:          Zeyuan Hu
  > Mail:            iamzeyuanhu@utexas.edu
  > Created Time:    02/12/19
  > Description:


 ************************************************************************/

#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#define CHAR_LEN 101

char filepath[CHAR_LEN];
char line[CHAR_LEN];
char address[CHAR_LEN];
char perms[CHAR_LEN];
char offset[CHAR_LEN];
char dev[CHAR_LEN];
char inode[CHAR_LEN];
char pathname[CHAR_LEN];

int main() {
  sprintf(filepath, "/proc/%u/maps", (unsigned)getpid());
  FILE *f = fopen(filepath, "r");

  printf("%-32s %-8s %-10s %-8s %-10s %s\n", "address", "perms", "offset",
         "dev", "inode", "pathname");
  while (fgets(line, sizeof(line), f) != NULL) {
    sscanf(line, "%s%s%s%s%s%s", address, perms, offset, dev, inode, pathname);
    printf("%-32s %-8s %-10s %-8s %-10s %s\n", address, perms, offset, dev,
           inode, pathname);
  }

  fclose(f);

  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    perror("getrusage");
    return 0;
  }

  printf("utime = %ld.%06ld s\n", usage.ru_utime.tv_sec,
         usage.ru_utime.tv_usec);
  printf("stime = %ld.%06ld s\n", usage.ru_stime.tv_sec,
         usage.ru_stime.tv_usec);
  printf("maxrss = %ld KB\n", usage.ru_maxrss);
  printf("minflt = %ld\n", usage.ru_minflt);
  printf("majflt = %ld\n", usage.ru_majflt);
  printf("inblock = %ld\n", usage.ru_inblock);
  printf("oublock = %ld\n", usage.ru_oublock);
  printf("nvcsw = %ld\n", usage.ru_nvcsw);
  printf("nivcsw = %ld\n", usage.ru_nivcsw);

  return 0;
}
