#ifndef STACK_H
#define STACK_H

#include "elf.h"

void* build_stack(void* rsp, struct elf_file_t* elf_file, int argc, char** argv);

#endif  // STACK_h
