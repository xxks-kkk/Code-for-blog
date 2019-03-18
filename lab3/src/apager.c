#include "common.h"
#include "elf.h"
#include "stack.h"

#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>

void __attribute__ ((noinline)) go(void* entry, void* rsp) {
    asm("movq $0, %rdx;"
        "movq %rsi, %rsp;"
        "jmp *%rdi;");
}

int main(int argc, char** argv) {
    CHECK(argc >= 2, "Usage: %s elf_file [arguments ...]", argv[0]);

    size_t page_size = sysconf(_SC_PAGESIZE);
    struct elf_file_t* elf_file = elf_read(argv[1]);
    CHECK(elf_file != NULL, "elf_read failed");

    for (int i = 0; i < elf_file->n_program_headers; i++) {
        struct elf_program_header_t* segment = elf_file->program_headers + i;

        CHECK(segment->type != PT_DYNAMIC && segment->type != PT_INTERP,
              "No support for dynamic loading");

        if (segment->type == PT_LOAD) {
            size_t vaddr = (size_t) segment->vaddr;
            size_t align = (size_t) segment->align;
            CHECK(align % page_size == 0, "align should be multiple of page size");
            size_t aligned_vaddr = vaddr & ~(align-1);
            size_t align_delta = vaddr - aligned_vaddr;
            LOG("vaddr = 0x%zx, align = 0x%zx, aligned_vaddr = 0x%zx",
                vaddr, align, aligned_vaddr);

            size_t length = (size_t) segment->memsz;
            size_t filesz = (size_t) segment->filesz;
            size_t offset = (size_t) segment->offset;
            length += align_delta;
            filesz += align_delta;
            CHECK(offset >= align_delta, "Invalid ph_align");
            offset -= align_delta;
            LOG("length = 0x%zx, filesz = 0x%zx, offset = 0x%zx",
                length, filesz, offset);
            CHECK(aligned_vaddr + length <= LOADER_START_ADDRESS,
                  "Overlapped with loader segments");

            void* addr = (void*) aligned_vaddr;
            int prot = ph_flags_to_prot(segment->flags);
            int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
            CHECK(mmap(addr, length, PROT_WRITE, flags, -1, 0) == addr,
                  "mmap failed: %s", strerror(errno));
            memcpy(addr, elf_file->content + offset, filesz);
            CHECK(mprotect(addr, length, prot) == 0, "mprotect failed");
        }
    }

    size_t stack_size = DEFAULT_STACK_SIZE;
    uint8_t* ptr = mmap(NULL, stack_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(ptr != MAP_FAILED, "mmap failed");
    void* rsp = build_stack(ptr + stack_size, elf_file, argc - 1, argv + 1);
    void* entry = (void*) elf_file->file_header->entry;
    go(entry, rsp);

    return 0;
}
