#include <stdio.h>

#define SIZE (1024 * 1024 * 16)

int a[SIZE];

int main() {
    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
    }
    getchar();
    return 0;
}
