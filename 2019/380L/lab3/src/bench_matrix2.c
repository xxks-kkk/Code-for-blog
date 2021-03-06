#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 10000
#define DELTA

unsigned int a[SIZE][SIZE];

int main() {
    srand(time(NULL));
    for (int j = 0; j < SIZE; j++) {
        for (int i = 0; i < SIZE; i++) {
            a[i][j] = rand();
        }
    }
    unsigned int result = 0;
    for (int j = 0; j < SIZE; j++) {
        for (int i = 0; i < SIZE; i++) {
            result ^= a[i][j];
        }
    }
    printf("result = 0x%x\n", result);
    return 0;
}
