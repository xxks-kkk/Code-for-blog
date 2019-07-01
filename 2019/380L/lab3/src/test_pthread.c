#include <stdio.h>
#include <pthread.h>

__thread long val;

void* thread(void* ptr) {
    long type = (long) ptr;
    val = type;
    fprintf(stderr, "Thread - %ld (0x%lx)\n", val, (long) &val);
    return ptr;
}

int main(int argc, char** argv) {
    pthread_t thread1, thread2;
    long thr = 1;
    long thr2 = 2;
    pthread_create(&thread1, NULL, *thread, (void*) thr);
    pthread_create(&thread2, NULL, *thread, (void*) thr2);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    return 0;
}
