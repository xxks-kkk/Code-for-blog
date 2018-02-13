#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>

const int MAX = 13;

static void doFib(int n, int doPrint);
static pid_t Fork(void);

/*
 * unix_error - unix-style error routine.
 */
inline static void unix_error(char *msg)
{
    fprintf(stdout, "%s: %s\n", msg, strerror(errno));
    exit(1);
}


int main(int argc, char **argv)
{
    int arg;
    int print=1;

    if(argc != 2){
        fprintf(stderr, "Usage: fib <num>\n");
        exit(-1);
    }

    arg = atoi(argv[1]);
    if(arg < 0 || arg > MAX){
        fprintf(stderr, "number must be between 0 and %d\n", MAX);
        exit(-1);
    }

    doFib(arg, print);

    return 0;
}

/* 
 * Recursively compute the specified number. If print is
 * true, print it. Otherwise, provide it to my parent process.
 *
 * NOTE: The solution must be recursive and it must fork
 * a new child for each call. Each process should call
 * doFib() exactly once.
 */
static void doFib(int n, int doPrint)
{
  pid_t pid1, pid2;
  int res = 0, status;
  //Base case, return n if n<2
  if (n <= 1){
    exit(n);
  }
  pid1 = Fork();
  /* Code executed by first child */
  if (pid1 == 0) {
    doFib(n-1, 0);
  }
  /* Code executed by second child */
  pid2 = Fork();
  if (pid2 == 0) {
    doFib(n-2, 0);
  }
  //Reap first Child
  while(waitpid(pid1, &status, 0) > 0)
    if (WIFEXITED(status))
      res += WEXITSTATUS(status);
  //Reap second child
  while(waitpid(pid2,&status, 0) > 0)
    if (WIFEXITED(status))
      res += WEXITSTATUS(status);
  if(doPrint){
    printf("%d\n", res);
  }
  exit(res);
}

static pid_t Fork(void)
{
  pid_t pid;
  if ((pid = fork()) < 0)
    unix_error("Fork error");
  return pid;
}
