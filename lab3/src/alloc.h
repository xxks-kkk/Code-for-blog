#ifndef ALLOC_H
#define ALLOC_H

#include <stdlib.h>

void* alloc_malloc(size_t size);
void alloc_free(void* ptr);

#endif  // ALLOC_H
