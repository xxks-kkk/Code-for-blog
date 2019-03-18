#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE (1024 * 1024 * 256)

unsigned int a[SIZE];

int main() {
    srand(time(NULL));
    for (int i = SIZE - 1; i >= 0; i--) {
        a[i] = rand();
    }
    unsigned int result = 0;
    for (int i = SIZE - 1; i >= 0; i--) {
        result ^= a[i];
    }
    printf("result = 0x%x\n", result);
    return 0;
}
