#include "common.h"
#include "elf_file.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  CHECK(argc == 2, "Usage: %s elf_file", argv[0]);

  struct elf_file_t *elf_file = elf_read(argv[1]);
  CHECK(elf_file != NULL, "elf_read failed");

  printf("[File header]\n");
  printf("e_type = 0x%x\n", elf_file->file_header->e_type);
  printf("e_machine = 0x%x\n", elf_file->file_header->e_machine);
  printf("e_version = 0x%x\n", elf_file->file_header->e_version);
  printf("e_entry = 0x%lx\n", elf_file->file_header->e_entry);
  printf("e_phoff = 0x%lx\n", elf_file->file_header->e_phoff);
  printf("e_shoff = 0x%lx\n", elf_file->file_header->e_shoff);
  printf("e_flags = 0x%x\n", elf_file->file_header->e_flags);
  printf("e_ehsize = 0x%x\n", elf_file->file_header->e_ehsize);
  printf("e_phentsize = 0x%x\n", elf_file->file_header->e_phentsize);
  printf("e_phnum = 0x%x\n", elf_file->file_header->e_phnum);
  printf("e_shentsize = 0x%x\n", elf_file->file_header->e_shentsize);
  printf("e_shnum = 0x%x\n", elf_file->file_header->e_shnum);
  printf("e_shstrndx = 0x%x\n", elf_file->file_header->e_shstrndx);
  printf("\n");

  for (int i = 0; i < elf_file->file_header->e_phnum; i++) {
    printf("[Program header %d]\n", i);
    printf("p_type = 0x%x\n", elf_file->program_header_table[i].p_type);
    printf("p_offset = 0x%lx\n", elf_file->program_header_table[i].p_offset);
    printf("p_vaddr = 0x%lx\n", elf_file->program_header_table[i].p_vaddr);
    printf("p_paddr = 0x%lx\n", elf_file->program_header_table[i].p_paddr);
    printf("p_filesz = 0x%lx\n", elf_file->program_header_table[i].p_filesz);
    printf("p_memsz = 0x%lx\n", elf_file->program_header_table[i].p_memsz);
    printf("p_flags = 0x%x\n", elf_file->program_header_table[i].p_flags);
    printf("p_align = 0x%lx\n", elf_file->program_header_table[i].p_align);
    printf("\n");
  }

  for (int i = 0; i < elf_file->file_header->e_shnum; i++) {
    printf("[Section header %d]\n", i);
    printf("sh_name = 0x%x\n", elf_file->section_header_table[i].sh_name);
    printf("sh_type = 0x%x\n", elf_file->section_header_table[i].sh_type);
    printf("sh_flags = 0x%lx\n", elf_file->section_header_table[i].sh_flags);
    printf("sh_addr = 0x%lx\n", elf_file->section_header_table[i].sh_addr);
    printf("sh_offset = 0x%lx\n", elf_file->section_header_table[i].sh_offset);
    printf("sh_size = 0x%lx\n", elf_file->section_header_table[i].sh_size);
    printf("sh_link = 0x%x\n", elf_file->section_header_table[i].sh_link);
    printf("sh_info = 0x%x\n", elf_file->section_header_table[i].sh_info);
    printf("sh_addralign = 0x%lx\n",
           elf_file->section_header_table[i].sh_addralign);
    printf("sh_entsize = 0x%lx\n",
           elf_file->section_header_table[i].sh_entsize);
    printf("\n");
  }

  CHECK(elf_free(elf_file) == 0, "elf_free failed");

  return 0;
}
