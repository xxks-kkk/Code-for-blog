#include "elf.h"
#include "common.h"
#include "alloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

struct elf_file_t* elf_read(const char* filepath) {
    struct elf_file_t* elf_file = alloc_malloc(sizeof(struct elf_file_t));
    CHECK(elf_file != NULL, "malloc failed");

    FILE* fp = fopen(filepath, "rb");
    CHECK(fp != NULL, "fopen failed");
    fseek(fp, 0, SEEK_END);
    size_t length = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    elf_file->content = alloc_malloc(length);
    CHECK(elf_file->content != NULL, "malloc failed");
    CHECK(fread(elf_file->content, length, 1, fp) == 1, "fread failed");
    CHECK(fclose(fp) == 0, "fclose failed");

    elf_file->file_header = (struct elf_file_header_t*) elf_file->content;

    CHECK(elf_file->file_header->ident[EI_MAG0] == 0x7f, "Invalid ELF file");
    CHECK(elf_file->file_header->ident[EI_MAG1] == 0x45, "Invalid ELF file");
    CHECK(elf_file->file_header->ident[EI_MAG2] == 0x4c, "Invalid ELF file");
    CHECK(elf_file->file_header->ident[EI_MAG3] == 0x46, "Invalid ELF file");
    CHECK(elf_file->file_header->ident[EI_CLASS] == 0x02, "Unsupported ELF file"); // 64bit
    CHECK(elf_file->file_header->ident[EI_DATA] == 0x01, "Unsupported ELF file");  // Little endian
    CHECK(elf_file->file_header->ident[EI_OSABI] == 0x00       // System V ABI
          || elf_file->file_header->ident[EI_OSABI] == 0x03,   // Linux ABI
          "Unsupported ELF file");
    CHECK(elf_file->file_header->ehsize == 0x40, "Invalid ELF file");
    CHECK(elf_file->file_header->phentsize == 0x38, "Invalid ELF file");
    CHECK(elf_file->file_header->shentsize == 0x40, "Invalid ELF file");

    void* ph_addr = elf_file->content + elf_file->file_header->phoff;
    elf_file->program_headers = (struct elf_program_header_t*) ph_addr;
    elf_file->n_program_headers = elf_file->file_header->phnum;

    void* sh_addr = elf_file->content + elf_file->file_header->shoff;
    elf_file->section_headers = (struct elf_section_header_t*) sh_addr;
    elf_file->n_section_headers = elf_file->file_header->shnum;

    return elf_file;
}

int elf_free(struct elf_file_t* elf_file) {
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
