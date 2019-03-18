
#include "common.h"
#include "elf.h"
#include "stack.h"

#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <sys/mman.h>

int prefetch_num;
struct elf_file_t* elf_file;

int page_size = 4096;
int log_page_size = 12;

#define PAGE_MASK_SIZE (1 << 19)   // Assume page_size = 4k
bool page_mask[PAGE_MASK_SIZE];

int load_page(long addr) {
    LOG("Try to load page containing address 0x%lx", addr);
    long page_start_addr = addr & ~(page_size - 1);
    long page_no = page_start_addr >> log_page_size;
    if (page_no < PAGE_MASK_SIZE) {
        if (page_mask[page_no]) {
            LOG("Page is already mapped");
            return -1;
        }
        page_mask[page_no] = true;
    } else {
        LOG("page_no is larger than PAGE_MASK_SIZE");
        return -1;
    }

    for (int i = 0; i < elf_file->n_program_headers; i++) {
        struct elf_program_header_t* segment = elf_file->program_headers + i;

        if (segment->type == PT_LOAD) {
            long vaddr = (long) segment->vaddr;
            long length = (long) segment->memsz;

            if (addr < vaddr || addr >= vaddr + length) {
                continue;
            }

            int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
            if (mmap((void*) page_start_addr, page_size, PROT_WRITE, flags, -1, 0) == MAP_FAILED) {
                LOG("mmap failed: %s", strerror(errno));
                return -1;
            }

            long filesz = (long) segment->filesz;
            long offset = (long) segment->offset;

            long offset_in_segment = page_start_addr - vaddr;
            if (offset_in_segment < filesz) {
                long copy_size = page_size;
                if (offset_in_segment + copy_size > filesz) {
                    copy_size = filesz - offset_in_segment;
                }
                memcpy((void*) page_start_addr,
                       elf_file->content + offset + offset_in_segment, copy_size);
            }

            int prot = ph_flags_to_prot(segment->flags);
            if (mprotect((void*) page_start_addr, page_size, prot) != 0) {
                LOG("mprotect failed");
                return -1;
            }

            return 0;
        }
    }

    LOG("Cannot find segment containing the address");
    return -1;
}

int prefetch_direction = 1;

void sigsegv_handler(int sig, siginfo_t* info, void* ucontext) {
    CHECK(sig == SIGSEGV, "Unexpected signal");
    long addr = (long) info->si_addr;
    LOG("Segmentation fault with address 0x%lx", addr);
    CHECK(load_page(addr) == 0, "Failed to load desired page");

    if (prefetch_num == 1) {
        LOG("prefetch_direction = %d", prefetch_direction);
        if (load_page(addr + page_size * prefetch_direction) != 0) {
            prefetch_direction *= -1;
            LOG("Reverse prefetch_direction");
            load_page(addr + page_size * prefetch_direction);
        }
    } else if (prefetch_num == 2) {
        LOG("prefetch_direction = %d", prefetch_direction);
        if (load_page(addr + page_size * prefetch_direction) != 0) {
            prefetch_direction *= -1;
            LOG("Reverse prefetch_direction");
            if (load_page(addr + page_size * prefetch_direction) == 0) {
                load_page(addr + page_size * prefetch_direction * 2);
            }
        } else {
            load_page(addr + page_size * prefetch_direction * 2);
        }
    }
}

void __attribute__ ((noinline)) go(void* entry, void* rsp) {
    asm("movq $0, %rdx;"
        "movq %rsi, %rsp;"
        "jmp *%rdi;");
}

int main(int argc, char** argv) {
    CHECK(argc >= 3, "Usage: %s prefetch_num elf_file [arguments ...]", argv[0]);

    CHECK(sysconf(_SC_PAGESIZE) == page_size, "page_size must be %d", page_size);
    memset(page_mask, 0, sizeof(page_mask));

    prefetch_num = atoi(argv[1]);
    CHECK(prefetch_num == 0 || prefetch_num == 1 || prefetch_num == 2, "Invalid prefetch_num");

    stack_t ss;
    ss.ss_sp = mmap(NULL, DEFAULT_STACK_SIZE, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(ss.ss_sp != MAP_FAILED, "mmap failed");
    ss.ss_size = DEFAULT_STACK_SIZE;
    ss.ss_flags = 0;
    CHECK(sigaltstack(&ss, NULL) == 0, "sigaltstack failed");

    struct sigaction act;
    act.sa_sigaction = sigsegv_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_ONSTACK | SA_SIGINFO;
    CHECK(sigaction(SIGSEGV, &act, NULL) == 0, "sigaction failed");

    elf_file = elf_read(argv[2]);
    CHECK(elf_file != NULL, "elf_read failed");

    for (int i = 0; i < elf_file->n_program_headers; i++) {
        struct elf_program_header_t* segment = elf_file->program_headers + i;

        CHECK(segment->type != PT_DYNAMIC && segment->type != PT_INTERP,
              "No support for dynamic loading");

        if (segment->type == PT_LOAD) {
            long vaddr = (long) segment->vaddr;
            long align = (long) segment->align;
            CHECK(align % page_size == 0, "align should be multiple of page size");
            long aligned_vaddr = vaddr & ~(align-1);
            long align_delta = vaddr - aligned_vaddr;
            LOG("vaddr = 0x%lx, align = 0x%lx, aligned_vaddr = 0x%lx",
                vaddr, align, aligned_vaddr);

            long length = (long) segment->memsz;
            long filesz = (long) segment->filesz;
            long offset = (long) segment->offset;
            length += align_delta;
            filesz += align_delta;
            CHECK(offset >= align_delta, "Invalid ph_align");
            offset -= align_delta;
            LOG("length = 0x%lx, filesz = 0x%lx, offset = 0x%lx",
                length, filesz, offset);
            CHECK(aligned_vaddr + length <= LOADER_START_ADDRESS,
                  "Overlapped with loader segments");

            void* addr = (void*) aligned_vaddr;
            int prot = ph_flags_to_prot(segment->flags);
            int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
            CHECK(mmap(addr, filesz, PROT_WRITE, flags, -1, 0) == addr,
                  "mmap failed: %s", strerror(errno));
            memcpy(addr, elf_file->content + offset, filesz);
            CHECK(mprotect(addr, filesz, prot) == 0, "mprotect failed");
        }
    }

    int stack_size = DEFAULT_STACK_SIZE;
    char* ptr = mmap(NULL, stack_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(ptr != MAP_FAILED, "mmap failed");
    void* rsp = build_stack(ptr + stack_size, elf_file, argc - 2, argv + 2);
    void* entry = (void*) elf_file->file_header->entry;
    go(entry, rsp);

    return 0;
}
