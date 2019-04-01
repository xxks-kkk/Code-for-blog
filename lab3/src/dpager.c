#include "common.h"
#include "elf.h"
#include "stack.h"

#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

size_t page_size;
struct elf_file_t *elf_file;

int load_page(size_t addr) {
  for (int i = 0; i < elf_file->file_header->e_phnum; i++) {
    Elf64_Phdr *segment = elf_file->program_header_table + i;

    if (segment->p_type == PT_LOAD) {
      size_t vaddr = (size_t)segment->p_vaddr;
      size_t length = (size_t)segment->p_memsz;

      if (addr < vaddr || addr >= vaddr + length) {
        continue;
      }

      void *page_start_addr = (void *)(addr & ~(page_size - 1));
      int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
      if (mmap(page_start_addr, page_size, PROT_WRITE, flags, -1, 0) !=
          page_start_addr) {
        LOG("mmap failed");
        return -1;
      }

      int64_t filesz = (int64_t)segment->p_filesz;
      int64_t offset = (int64_t)segment->p_offset;

      int64_t offset_in_segment = (int64_t)page_start_addr - (int64_t)vaddr;
      if (offset_in_segment < filesz) {
        int64_t copy_size = page_size;
        if (offset_in_segment + copy_size > filesz) {
          copy_size = filesz - offset_in_segment;
        }
        memcpy(page_start_addr, elf_file->content + offset + offset_in_segment,
               copy_size);
      }

      int prot = ph_flags_to_prot(segment->p_flags);
      if (mprotect(page_start_addr, page_size, prot) != 0) {
        LOG("mprotect failed");
        return -1;
      }

      return 0;
    }
  }
  return -1;
}

void sigsegv_handler(int sig, siginfo_t *info, void *ucontext) {
  CHECK(sig == SIGSEGV, "Unexpected signal");
  size_t addr = (size_t)info->si_addr;
  LOG("Segmentation fault with address 0x%zx", addr);
  CHECK(load_page(addr) == 0, "Failed to load desired page");
}

void __attribute__((noinline)) go(void *entry, void *rsp) {
  __asm__("movq $0, %rdx;"
          "movq %rsi, %rsp;"
          "jmp *%rdi;");
}

int main(int argc, char **argv) {
  CHECK(argc >= 2, "Usage: %s elf_file [arguments ...]", argv[0]);

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

  page_size = sysconf(_SC_PAGESIZE);
  elf_file = elf_read(argv[1]);
  CHECK(elf_file != NULL, "elf_read failed");

  for (int i = 0; i < elf_file->file_header->e_phnum; i++) {
    Elf64_Phdr *segment = elf_file->program_header_table + i;

    CHECK(segment->p_type != PT_DYNAMIC && segment->p_type != PT_INTERP,
          "No support for dynamic loading");

    if (segment->p_type == PT_LOAD) {
      size_t vaddr = (size_t)segment->p_vaddr;
      size_t length = (size_t)segment->p_memsz;
      size_t align = (size_t)segment->p_align;
      CHECK(align % page_size == 0, "align should be multiple of page size");
      CHECK(vaddr + length <= LOADER_START_ADDRESS,
            "Overlapped with loader segments");
    }
  }

  size_t stack_size = DEFAULT_STACK_SIZE;
  uint8_t *ptr = mmap(NULL, stack_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  CHECK(ptr != MAP_FAILED, "mmap failed");
  void *rsp = build_stack(ptr + stack_size, elf_file, argc - 1, argv + 1);

  void *entry = (void *)elf_file->file_header->e_entry;
  load_page((size_t)entry);

  go(entry, rsp);

  return 0;
}
