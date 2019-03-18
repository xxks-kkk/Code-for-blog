#include "stack.h"
#include "common.h"

#include <stdint.h>
#include <stdlib.h>
#include <sys/auxv.h>

extern char** environ;

void* push_8bytes(uint64_t* rsp, uint64_t val) {
    rsp--;
    *rsp = val;
    return rsp;
}

void* push_auxv(void* rsp, unsigned long type) {
    unsigned long val = getauxval(type);
    if (val != 0) {
        rsp = push_8bytes(rsp, (uint64_t) val);
        rsp = push_8bytes(rsp, (uint64_t) type);
    }
    return rsp;
}

void* push_auxv_with_val(void* rsp, unsigned long type, unsigned long val) {
    rsp = push_8bytes(rsp, (uint64_t) val);
    rsp = push_8bytes(rsp, (uint64_t) type);
    return rsp;
}

void* build_stack(void* rsp, struct elf_file_t* elf_file, int argc, char** argv) {
    rsp = push_8bytes(rsp, 0);

    rsp = push_auxv(rsp, AT_UID);
    rsp = push_auxv(rsp, AT_GID);
    rsp = push_auxv(rsp, AT_EUID);
    rsp = push_auxv(rsp, AT_EGID);
    rsp = push_auxv(rsp, AT_RANDOM);
    rsp = push_auxv(rsp, AT_CLKTCK);
    rsp = push_auxv(rsp, AT_HWCAP);
    rsp = push_auxv(rsp, AT_PAGESZ);
    rsp = push_auxv(rsp, AT_PLATFORM);
    rsp = push_auxv(rsp, AT_SECURE);
    rsp = push_auxv(rsp, AT_SYSINFO_EHDR);
    rsp = push_auxv(rsp, AT_ICACHEBSIZE);
    rsp = push_auxv(rsp, AT_DCACHEBSIZE);
    rsp = push_auxv(rsp, AT_UCACHEBSIZE);

    uint64_t phdr = elf_file->program_headers[0].vaddr + elf_file->file_header->phoff;
    LOG("phdr = 0x%lx", (long) phdr);
    rsp = push_auxv_with_val(rsp, AT_PHDR, phdr);
    rsp = push_auxv_with_val(rsp, AT_PHENT, elf_file->file_header->phentsize);
    rsp = push_auxv_with_val(rsp, AT_PHNUM, elf_file->file_header->phnum);

    rsp = push_8bytes(rsp, 0);

    int n_env = 0;
    while (environ[n_env] != NULL) {
        n_env++;
    }
    for (int i = n_env - 1; i >= 0; i--) {
        rsp = push_8bytes(rsp, (uint64_t) environ[i]);
    }
    rsp = push_8bytes(rsp, 0);

    for (int i = argc - 1; i >= 0; i--) {
        rsp = push_8bytes(rsp, (uint64_t) argv[i]);
    }
    rsp = push_8bytes(rsp, (uint64_t) argc);

    return rsp;
}
