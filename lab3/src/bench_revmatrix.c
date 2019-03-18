#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 10000

unsigned int a[SIZE][SIZE];

int main() {
    srand(time(NULL));
    for (int j = SIZE - 1; j >= 0; j--) {
        for (int i = SIZE - 1; i >= 0; i--) {
            a[i][j] = rand();
        }
    }
    unsigned int result = 0;
    for (int j = SIZE - 1; j >= 0; j--) {
        for (int i = SIZE - 1; i >= 0; i--) {
            result ^= a[i][j];
        }
    }
    printf("result = 0x%x\n", result);
    return 0;
}
