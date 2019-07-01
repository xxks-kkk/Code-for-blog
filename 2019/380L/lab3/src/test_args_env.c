#include <stdio.h>

int main(int argc, char** argv, char** envp) {
    printf("[Arguments]\n");
    for (int i = 0; i < argc; i++) {
        printf("%s\n", argv[i]);
    }
    printf("\n");
    printf("[Environment variables]\n");
    char** p = envp;
    while (*p != NULL) {
        printf("%s\n", *p);
        p++;
    }
    return 0;
}
