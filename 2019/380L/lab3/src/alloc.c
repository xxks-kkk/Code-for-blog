#include "alloc.h"

#define BUFFER_SIZE (1 << 23)  // 8MB

static char alloc_buffer[BUFFER_SIZE];
static size_t alloc_pos = 0;

void* alloc_malloc(size_t size) {
    if (alloc_pos + size > BUFFER_SIZE) {
        return NULL;
    }
    void* ptr = alloc_buffer + alloc_pos;
    alloc_pos += size;
    return ptr;
}

void alloc_free(void* ptr) {
    return;
}
