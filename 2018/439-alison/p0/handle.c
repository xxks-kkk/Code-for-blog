#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include "util.h"


/*
 * First, print out the process ID of this process.
 *
 * Then, set up the signal handler so that ^C causes
 * the program to print "Nice try.\n" and continue looping.
 *
 * Finally, loop forever, printing "Still here\n" once every
 * second.
 */

void handler2(int sig)
{
    if (sig == SIGINT)
    {
        ssize_t bytes;
        const int STDOUT = 1;
        char *msg = "Nice try.\n";
        bytes = write(STDOUT, msg, sizeof(msg));
    }
    else if (sig == SIGUSR1)
    {
        ssize_t bytes;
        const int STDOUT = 1;
        char *msg = "exiting";
        bytes = write(STDOUT, msg, sizeof(msg));
        exit(1);
    }
}

int main(int argc, char **argv)
{
    pid_t pid;
    pid = getpid();
    printf("Process ID: %d\n", pid);
    struct timespec ts;
    ts.tv_sec = 5;
    Signal(SIGINT, handler2);
    Signal(SIGUSR1, handler2);
    while(1)
    {
        if (nanosleep(&ts, NULL) == 0)
            // nanosleep returns 0 if successfully sleeping for the requested interval
            printf("Still here\n");
    }
}


