#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE (1024 * 1024 * 200)
#define ACCESS_SIZE (SIZE / 8)

unsigned int a[SIZE];

int main() {
    unsigned seed = time(NULL);
    srand(seed);
    for (int i = 0; i < ACCESS_SIZE; i++) {
        a[rand() % SIZE] = i;
    }
    unsigned int result = 0;
    srand(seed);
    for (int i = 0; i < ACCESS_SIZE; i++) {
        result ^= a[rand() % SIZE];
    }
    printf("result = 0x%x\n", result);
    return 0;
}
