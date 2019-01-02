/*************************************************************************
  > File Name:       t1.c
  > Author:          Zeyuan Hu
  > Mail:            ferrishu3886@gmail.com
  > Created Time:    10/3/18
  > Description:
    
    An example of race condition (more specifically, data race)

    counter += 1; can be dissambled (i.e. objdump -d main) as
    (assume counter has address 0x8049a1c)

    mov 0x8049a1c, %eax
    add $0x1, %eax
    mov %eax, 0x8049a1c

    If interrupt happens after add $0x1, %eax for p1, p2 runs and 0x8059a1c
    has value 51. Then p1 resume, and mov %eax, 0x8049a1c leads to 51 in
    0x8049a1c again because %eax for p1 is 51. However, the correct value
    should be 52.
 ************************************************************************/


#include <stdio.h>
#include <pthread.h>

static volatile int counter = 0;

void *
mythread(void *arg)
{
    printf("%s: begin\n", (char *) arg);
    int i;
    for (i = 0; i < 1e7; i++)
    {
        counter += 1;
    }
    printf("%s: done\n", (char *) arg);
    return NULL;
}

int
main(int argc, char *argv[])
{
    pthread_t p1, p2;
    printf("main: begin (counter = %d)\n", counter);
    pthread_create(&p1, NULL, mythread, "A");
    pthread_create(&p2, NULL, mythread, "B");

    // join waits for the threads to finish
    pthread_join(p1, NULL);
    pthread_join(p2, NULL);
    printf("main: done with both (counter = %d)\n", counter);
    return 0;
}
