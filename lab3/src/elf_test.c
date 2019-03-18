#include "common.h"
#include "elf.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    CHECK(argc == 2, "Usage: %s elf_file", argv[0]);

    struct elf_file_t* elf_file = elf_read(argv[1]);
    CHECK(elf_file != NULL, "elf_read failed");

    printf("[File header]\n");
    printf("type = 0x%x\n", elf_file->file_header->type);
    printf("machine = 0x%x\n", elf_file->file_header->machine);
    printf("version = 0x%x\n", elf_file->file_header->version);
    printf("entry = 0x%lx\n", elf_file->file_header->entry);
    printf("phoff = 0x%lx\n", elf_file->file_header->phoff);
    printf("shoff = 0x%lx\n", elf_file->file_header->shoff);
    printf("flags = 0x%x\n", elf_file->file_header->flags);
    printf("ehsize = 0x%x\n", elf_file->file_header->ehsize);
    printf("phentsize = 0x%x\n", elf_file->file_header->phentsize);
    printf("phnum = 0x%x\n", elf_file->file_header->phnum);
    printf("shentsize = 0x%x\n", elf_file->file_header->shentsize);
    printf("shnum = 0x%x\n", elf_file->file_header->shnum);
    printf("shstrndx = 0x%x\n", elf_file->file_header->shstrndx);
    printf("\n");

    for (int i = 0; i < elf_file->n_program_headers; i++) {
        printf("[Program header %d]\n", i);
        printf("type = 0x%x\n", elf_file->program_headers[i].type);
        printf("flags = 0x%x\n", elf_file->program_headers[i].flags);
        printf("offset = 0x%lx\n", elf_file->program_headers[i].offset);
        printf("vaddr = 0x%lx\n", elf_file->program_headers[i].vaddr);
        printf("paddr = 0x%lx\n", elf_file->program_headers[i].paddr);
        printf("filesz = 0x%lx\n", elf_file->program_headers[i].filesz);
        printf("memsz = 0x%lx\n", elf_file->program_headers[i].memsz);
        printf("align = 0x%lx\n", elf_file->program_headers[i].align);
        printf("\n");
    }

    for (int i = 0; i < elf_file->n_section_headers; i++) {
        printf("[Section header %d]\n", i);
        printf("name = 0x%x\n", elf_file->section_headers[i].name);
        printf("type = 0x%x\n", elf_file->section_headers[i].type);
        printf("flags = 0x%lx\n", elf_file->section_headers[i].flags);
        printf("addr = 0x%lx\n", elf_file->section_headers[i].addr);
        printf("offset = 0x%lx\n", elf_file->section_headers[i].offset);
        printf("size = 0x%lx\n", elf_file->section_headers[i].size);
        printf("link = 0x%x\n", elf_file->section_headers[i].link);
        printf("info = 0x%x\n", elf_file->section_headers[i].info);
        printf("addralign = 0x%lx\n", elf_file->section_headers[i].addralign);
        printf("entsize = 0x%lx\n", elf_file->section_headers[i].entsize);
        printf("\n");
    }

    CHECK(elf_free(elf_file) == 0, "elf_free failed");

    return 0;
}
