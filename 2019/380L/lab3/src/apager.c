#include "common.h"
#include "elf_file.h"
#include "stack.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

void __attribute__((noinline)) go(void *entry, void *rsp) {
  __asm__("movq $0, %rdx;"
          "movq %rsi, %rsp;"
          "jmp *%rdi;");
}

int main(int argc, char **argv) {
  CHECK(argc >= 2, "Usage: %s elf_file [arguments ...]", argv[0]);

  size_t page_size = sysconf(_SC_PAGESIZE);
  struct elf_file_t *elf_file = elf_read(argv[1]);
  CHECK(elf_file != NULL, "elf_read failed");
  int fd = open(elf_file->file, O_RDONLY);
  LOG("executable: %s\n", elf_file->file);
  CHECK(fd > 0, "Open file: %s failed\n", elf_file->file);

  for (int i = 0; i < elf_file->file_header->e_phnum; i++) {
    LOG("iteration: %d\n", i);
    Elf64_Phdr *segment = elf_file->program_header_table + i;

    CHECK(segment->p_type != PT_DYNAMIC && segment->p_type != PT_INTERP,
          "No support for dynamic loading");

    if (segment->p_type == PT_LOAD) {
      size_t vaddr = (size_t)segment->p_vaddr;
      size_t align = (size_t)segment->p_align;
      size_t memsz = (size_t)segment->p_memsz;
      size_t filesz = (size_t)segment->p_filesz;
      size_t offset = (size_t)segment->p_offset;

      CHECK(align % page_size == 0, "align should be multiple of page size");
      LOG("vaddr = 0x%zx, align = 0x%zx, memsz = 0x%zx, "
          "filesz = 0x%zx, offset = 0x%zx",
          vaddr, align, memsz, filesz, offset);

      size_t size = filesz + PAGE_OFFSET(vaddr, page_size);
      size_t off = offset - PAGE_OFFSET(vaddr, page_size);
      size = PAGE_ALIGN(size, page_size);
      size_t addr = PAGE_START(vaddr, page_size);
      CHECK(addr % page_size == 0, "addr should be multiple of page size");
      CHECK(addr + memsz <= LOADER_START_ADDRESS,
            "Overlapped with loader segments");

      int prot = ph_flags_to_prot(segment->p_flags);
      int flags;
      if (filesz != 0) {
        flags = MAP_PRIVATE | MAP_FIXED;
        CHECK(mmap((void *)addr, size, PROT_EXEC | PROT_READ, flags, fd, off) ==
                  (void *)addr,
              "mmap failed: %s", strerror(errno));
        CHECK(mprotect((void *)addr, size, prot) == 0, "mprotect failed");
      }

      if (memsz > filesz) {
        size_t diff =
            PAGE_ALIGN(memsz, page_size) - PAGE_ALIGN(filesz, page_size);
        flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED;
        CHECK(mmap((void *)addr + size, diff, PROT_WRITE, flags, -1, 0) ==
                  (void *)addr + size,
              "mmap failed: %d, %s", errno, strerror(errno));
      }
    }
  }

  size_t stack_size = DEFAULT_STACK_SIZE;
  uint8_t *ptr = mmap(NULL, stack_size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_GROWSDOWN, -1, 0);
  CHECK(ptr != MAP_FAILED, "mmap failed");
  void *rsp = build_stack(ptr + stack_size, elf_file, argc - 1, argv + 1);
  void *entry = (void *)elf_file->file_header->e_entry;
  go(entry, rsp);

  return 0;
}
