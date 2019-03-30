#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>

#if defined(DEBUG) || defined(_DEBUG) || defined(PRINT_LOG)
#define LOG(M, ...)                                                            \
  do {                                                                         \
    fprintf(stderr, "[%s (%s:%d)]  " M "\n", __func__, __FILE__, __LINE__,     \
            ##__VA_ARGS__);                                                    \
  } while (0)
#else
#define LOG(M, ...)                                                            \
  do {                                                                         \
  } while (0)
#endif

#define CHECK(EXPR, M, ...)                                                    \
  do {                                                                         \
    if (!(EXPR)) {                                                             \
      fprintf(stderr, "[%s (%s:%d) ERROR]  " M "\n", __func__, __FILE__,       \
              __LINE__, ##__VA_ARGS__);                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define LOADER_START_ADDRESS 0x7f400000
#define DEFAULT_STACK_SIZE 8388608

#endif // COMMON_H
