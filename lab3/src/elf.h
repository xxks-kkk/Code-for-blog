#ifndef ELF_H
#define ELF_H

#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>

#define EI_MAG0   0
#define EI_MAG1   1
#define EI_MAG2   2
#define EI_MAG3   3
#define EI_CLASS  4
#define EI_DATA   5
#define EI_OSABI  7

#define PT_LOAD     1
#define PT_DYNAMIC  2
#define PT_INTERP   3

#define PF_X  0x1
#define PF_W  0x2
#define PF_R  0x4

#define SHT_PROGBITS    1
#define SHT_NOBITS      8
#define SHT_INIT_ARRAY  14
#define SHT_FINI_ARRAY  15

#define SHF_WRITE      0x1
#define SHF_ALLOC      0x2
#define SHF_EXECINSTR  0x4

struct __attribute__ ((__packed__)) elf_file_header_t {
    uint8_t   ident[16];
    uint16_t  type;
    uint16_t  machine;
    uint32_t  version;
    uint64_t  entry;
    uint64_t  phoff;
    uint64_t  shoff;
    uint32_t  flags;
    uint16_t  ehsize;
    uint16_t  phentsize;
    uint16_t  phnum;
    uint16_t  shentsize;
    uint16_t  shnum;
    uint16_t  shstrndx;
};

struct __attribute__ ((__packed__)) elf_program_header_t {
    uint32_t  type;
    uint32_t  flags;
    uint64_t  offset;
    uint64_t  vaddr;
    uint64_t  paddr;
    uint64_t  filesz;
    uint64_t  memsz;
    uint64_t  align;
};

struct __attribute__ ((__packed__)) elf_section_header_t {
    uint32_t  name;
    uint32_t  type;
    uint64_t  flags;
    uint64_t  addr;
    uint64_t  offset;
    uint64_t  size;
    uint32_t  link;
    uint32_t  info;
    uint64_t  addralign;
    uint64_t  entsize;
};

static_assert(sizeof(struct elf_file_header_t) == 64,
              "struct elf_file_header_t must be 64 bytes");
static_assert(sizeof(struct elf_program_header_t) == 56,
              "struct elf_program_header_t must be 64 bytes");
static_assert(sizeof(struct elf_section_header_t) == 64,
              "struct elf_section_header_t must be 64 bytes");

struct elf_file_t {
    uint8_t*                      content;
    struct elf_file_header_t*     file_header;
    int                           n_program_headers;
    struct elf_program_header_t*  program_headers;
    int                           n_section_headers;
    struct elf_section_header_t*  section_headers;
};

struct elf_file_t* elf_read(const char* filepath);
int elf_free(struct elf_file_t* elf_file);
int ph_flags_to_prot(int ph_flags);

#endif  // ELF_H
