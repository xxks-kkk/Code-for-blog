/*************************************************************************
  > File Name:       memory_map.c
  > Author:          Zeyuan Hu
  > Mail:            iamzeyuanhu@utexas.edu
  > Created Time:    02/12/19
  > Description:

    A program that opens, reads, and prints the /proc/self/maps file

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
  return 0;
}
