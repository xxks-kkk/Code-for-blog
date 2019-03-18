#include <stdio.h>
#include <stdlib.h>

int main() {
    for (int i = 0; i < 4096; i++) {
        int* p = malloc(sizeof(int));
        *p = i;
    }
    getchar();
    return 0;
}
