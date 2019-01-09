# Lab 1

## Exercise 3

> At what point does the processor start executing 32-bit code? What exactly causes the switch from 16- to 32-bit mode?

``` assembly
movl    %cr0, %eax
orl     $CR0_PE_ON, %eax
movl    %eax, %cr0

ljmp    $PROT_MODE_CSEG, $protcseg
```

We enable protected mode by set 0 bit of [CR0](https://wiki.osdev.org/CPU_Registers_x86#CR0) register. Then the program
issues [ljmp](https://stackoverflow.com/questions/5211541/bootloader-switching-processor-to-protected-mode), which requires
a code segment to switch to (e.g., `$protcseg`) and the address to jump to (e.g., `$PROT_MODE_CSEG`). Then, 
`movw    $PROT_MODE_DSEG, %ax    # Our data segment selector` is the point the processor starts executing 32-bit code.

> What is the last instruction of the boot loader executed, and what is the first instruction of the kernel it just loaded?

The last instruction that boot loader executed is

``` c
// call the entry point from the ELF header
// note: does not return!
((void (*)(void)) (ELFHDR->e_entry))();
```

To be more specific, if we check `boot.asm`, we see

``` assembly
    ((void (*)(void)) (ELFHDR->e_entry))();
7d6b:       ff 15 18 00 01 00       call   *0x10018
```

Thus, the last execution that boot loader executed is `call *0x10018`. To let gdb fast forward to this instruction, we do
`b *0x7d6b` and then `c`.

``` gdb
(gdb) b *0x7d6b
Breakpoint 2 at 0x7d6b
(gdb) c
Continuing.
=> 0x7d6b:      call   *0x10018

Breakpoint 2, 0x00007d6b in ?? ()
(gdb) si
=> 0x10000c:    movw   $0x1234,0x472
```

As one can see, the first instruction of the kernel the boot loader just loaded is `movw   $0x1234,0x472`. To ensure this is indeed the first instruction of 
the kernel, we can do 

``` shell
$ objdump -f ~/lab/obj/kern/kernel

kernel:     file format elf32-i386
architecture: i386, flags 0x00000112:
EXEC_P, HAS_SYMS, D_PAGED
start address 0x0010000c
```

We see the start address `0x0010000c` matches the address of instruction we see from gdb.

> Where is the first instruction of the kernel?

To find the location of the instruction, we first check `kernel.asm`, we see

``` assembly
f0100000 <_start+0xeffffff4>:
.globl          _start
_start = RELOC(entry)

.globl entry
entry:
        movw    $0x1234,0x472                   # warm boot
        f0100000:       02 b0 ad 1b 00 00       add    0x1bad(%eax),%dh
        f0100006:       00 00                   add    %al,(%eax)
        f0100008:       fe 4f 52                decb   0x52(%edi)
        f010000b:       e4                      .byte 0xe4
        
        f010000c <entry>:
        f010000c:       66 c7 05 72 04 00 00    movw   $0x1234,0x472
        f0100013:       34 12
```

`_start = RELOC(entry)` and `f01000c <entry>` indicate we should take a look at `entry.S`:

``` assembly
.globl entry
entry:
        movw    $0x1234,0x472                   # warm boot
```

> How does the boot loader decide how many sectors it must read in order to fetch the entire kernel from disk? Where does it find this information?

The information lies in the following code segment from `main.c`:

``` c
struct Proghdr *ph, *eph;
        
// read 1st page off disk
readseg((uint32_t) ELFHDR, SECTSIZE*8, 0);
                        
// is this a valid ELF?
if (ELFHDR->e_magic != ELF_MAGIC)
    goto bad;
                                                        
// load each program segment (ignores ph flags)
ph = (struct Proghdr *) ((uint8_t *) ELFHDR + ELFHDR->e_phoff);
eph = ph + ELFHDR->e_phnum;
for (; ph < eph; ph++)
    // p_pa is the load address of this segment (as well
    // as the physical address)
    readseg(ph->p_pa, ph->p_memsz, ph->p_offset);
```

First, the program reads 8 sectors from the disk into the memory to initialize `ELFHDR` struct 
(e.g., `readseg((uint32_t) ELFHDR, SECTSIZE*8, 0);`). Since a sector is 512 bytes, and 8 sectors
is 4096 bytes, which is 4KB (i.e., 1st page). 

Once `ELFHDR` struct is initialized, we first check 
if the kernel image follows [ELF format](https://pdos.csail.mit.edu/6.828/2018/readings/elf.pdf)
and in specific, we check for [ELF_MAGIC](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-828-operating-system-engineering-fall-2012/lecture-notes-and-readings/MIT6_828F12_xv6-book-rev7.pdf). If ELF header (i.e., `ELFHDR`) has the right magic number (i.e., `ELF_MAGIC` (0952)), then
the kernel image binary is well-formed.

Next, we read program header table and the start of the table is pointed by `ph`. `e_phoff` holds the program header table's file offset in bytes.
Then, the end of the table is pointed by `eph`. `e_phnum` holds the number of entries in the program header table. Each entry of the table corresponds
to a memory segment in the binary image. More information about ELF, see [wiki](https://en.wikipedia.org/wiki/Executable_and_Linkable_Format) and
[ELF specification](https://pdos.csail.mit.edu/6.828/2018/readings/elf.pdf).

Then, we can walk through the entries of the table (i.e., program header) and read each segment from disk to memory. `p_pa` refers to the segment's destination physical address,
`p_memsz` gives number of bytes in the memory image of the segment, and `p_offset` gives he offset from the beginning of the file at which the first byte 
of the segment resides.

Thus, the boot loader decide how many sectors it must read in order to fetch the entire kernel from disk based on ELF header (i.e., first 8 sectors of the disk).

## Exercise 5

> Trace through the first few instructions of the boot loader again and identify the first instruction that would "break" or otherwise do the wrong thing if you were to get the boot loader's link address wrong. Then change the link address in boot/Makefrag to something wrong, run make clean, recompile the lab with make, and trace into the boot loader again to see what happens. Don't forget to change the link address back and make clean again afterward! 

Let's modify the link address from `0x7c00` into `0x7c0d` and see what's going on. We still break `0x7c00` as it is hardwired physical address of the sector that BIOS will load from disk into memory. 
We hit `c` for multiple times and we see that we are essentially trapped inside a infinite loop (i.e., beginning part of `boot.asm` will get executed repeatedly). If we compare `boot.asm` with correct link address with our incorrect one, the first instruction is different is following (e.g., we ignore the instructions are different only due to the address differences)

``` assembly
 7c21:	64 7c 0f             	fs jl  7c33 <protcseg+0x1>
```

This line happens immediately after `lgdtl (%esi)`. Let's compare two `boot.asm` in this area:

``` assembly
# boot.asm with correct link address
lgdt    gdtdesc
    7c1e:	0f 01 16             	lgdtl  (%esi)
    7c21:	64 7c 0f             	fs jl  7c33 <protcseg+0x1>
```

``` assembly
# boot.asm with incorrect link address
lgdt    gdtdesc
    7c2e:	0f 01 16             	lgdtl  (%esi)
    7c31:	74 7c                	je     7caf <readsect+0x23>
```

Thus, the first instruction would break is `lgdt gdtdesc`.


