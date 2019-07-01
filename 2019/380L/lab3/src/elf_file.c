#include "elf_file.h"
#include "alloc.h"
#include "common.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

struct elf_file_t *elf_read(const char *filepath) {
  struct elf_file_t *elf_file = alloc_malloc(sizeof(struct elf_file_t));
  CHECK(elf_file != NULL, "malloc failed");

  FILE *fp = fopen(filepath, "rb");
  CHECK(fp != NULL, "fopen failed");
  fseek(fp, 0, SEEK_END);
  size_t length = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  elf_file->content = alloc_malloc(length);
  CHECK(elf_file->content != NULL, "malloc failed");
  CHECK(fread(elf_file->content, length, 1, fp) == 1, "fread failed");
  CHECK(fclose(fp) == 0, "fclose failed");

  elf_file->file = filepath;
  elf_file->file_header = (Elf64_Ehdr *)elf_file->content;

  CHECK(elf_file->file_header->e_ident[EI_MAG0] == 0x7f, "Invalid ELF file");
  CHECK(elf_file->file_header->e_ident[EI_MAG1] == 0x45, "Invalid ELF file");
  CHECK(elf_file->file_header->e_ident[EI_MAG2] == 0x4c, "Invalid ELF file");
  CHECK(elf_file->file_header->e_ident[EI_MAG3] == 0x46, "Invalid ELF file");
  CHECK(elf_file->file_header->e_ident[EI_CLASS] == 0x02,
        "Unsupported ELF file"); // 64bit
  CHECK(elf_file->file_header->e_ident[EI_DATA] == 0x01,
        "Unsupported ELF file");                         // Little endian
  CHECK(elf_file->file_header->e_ident[EI_OSABI] == 0x00 // System V ABI
            || elf_file->file_header->e_ident[EI_OSABI] == 0x03, // Linux ABI
        "Unsupported ELF file");
  CHECK(elf_file->file_header->e_ehsize == 0x40, "Invalid ELF file");
  CHECK(elf_file->file_header->e_phentsize == 0x38, "Invalid ELF file");
  CHECK(elf_file->file_header->e_shentsize == 0x40, "Invalid ELF file");

  void *ph_addr = elf_file->content + elf_file->file_header->e_phoff;
  elf_file->program_header_table = (Elf64_Phdr *)ph_addr;

  void *sh_addr = elf_file->content + elf_file->file_header->e_shoff;
  elf_file->section_header_table = (Elf64_Shdr *)sh_addr;

  return elf_file;
}

int elf_free(struct elf_file_t *elf_file) {
  alloc_free(elf_file->content);
  alloc_free(elf_file);
  return 0;
}

int ph_flags_to_prot(int ph_flags) {
  int prot = 0;
  if ((ph_flags & PF_R) != 0) {
    prot |= PROT_READ;
  }
  if ((ph_flags & PF_W) != 0) {
    prot |= PROT_WRITE;
  }
  if ((ph_flags & PF_X) != 0) {
    prot |= PROT_EXEC;
  }
  return prot;
}
