#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <string.h>

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
    if(argc != 2){
        fprintf(stderr, "Usage: mykill <num>\n");
        exit(-1);
    }
    int arg;
    arg = atoi(argv[1]);
    if (kill(arg, SIGUSR1) != 0)
        unix_error("Process ID doesn't exist");
    return 0;
}
