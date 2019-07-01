#ifndef ELF_FILE_H
#define ELF_FILE_H

#include <assert.h>
#include <elf.h>
#include <stdint.h>
#include <sys/mman.h>

struct elf_file_t {
  char *file;
  uint8_t *content;
  Elf64_Ehdr *file_header;
  Elf64_Phdr *program_header_table;
  Elf64_Shdr *section_header_table;
};

struct elf_file_t *elf_read(const char *filepath);
int elf_free(struct elf_file_t *elf_file);
int ph_flags_to_prot(int ph_flags);

#endif // ELF_FILE_H
