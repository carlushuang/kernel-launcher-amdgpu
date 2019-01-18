# kernel-launcher-amdgpu

launcher kernel for amdgpu. using HSA/OCL for backend

for disassemble:
```
llvm-objdump -disassemble -mcpu=gfx900 asm-kernel.o
llvm-objdump -disassemble-all -mcpu=gfx900 asm-kernel.o
```

# build
```
# I use llvm-7.0 to compile assembly, so change LLVM_DIR in rebuild.sh to your llvm path
sh rebuild.sh
```

# amd_kernel_code_t
amdgpu assembly need store some infomation in [amd_kernel_code_t](http://llvm.org/docs/AMDGPUUsage.html#amd-kernel-code-t), in `.text` section. [amd_kernel_code_t.cpp](utils/dump_amd_kernel_code_t/amd_kernel_code_t.cpp) is a util to dump this section from a elf file. It should be note that `.amdgpu_metadata
` is not needed in assembly code, as noted in  https://www.llvm.org/docs/AMDGPUUsage.html#amdgpu-metadata

# reduce ELF binary size

## remove useless section
```
  [Nr] Name              Type             Address           Offset
       Size              EntSize          Flags  Link  Info  Align
  [ 0]                   NULL             0000000000000000  00000000
       0000000000000000  0000000000000000           0     0     0
  [ 1] .note             NOTE             0000000000000248  00000248
       0000000000000044  0000000000000000   A       0     0     4
  [ 2] .text             PROGBITS         0000000000001000  00001000
       00000000000001a8  0000000000000000  AX       0     0     256
  [ 3] .symtab           SYMTAB           0000000000000000  00002070
       0000000000000060  0000000000000018           5     4     8
  [ 4] .shstrtab         STRTAB           0000000000000000  000020d0
       0000000000000027  0000000000000000           0     0     1
  [ 5] .strtab           STRTAB           0000000000000000  000020f7
       0000000000000019  0000000000000000           0     0     1
```
`readelf -a vector-add-2.co`  
after my test on hsaruntime(rocm 2.0), remain above sectoin (header, .note, .text, .symtab, .shstrtab, .strtab) is sufficient to run.

hence one may remove other useless section by `llvm-objcopy --remove-section=<section>`. below is my sequence
```
# after build, remove useless section in build/kernel/src/vector-add-2.co
# llvm-objcopy can be find in llvm folder you build this project

llvm-objcopy  --remove-section .comment     vector-add-2.co
# below section have dependency, need follow sequence
llvm-objcopy  --remove-section .gnu.hash    vector-add-2.co
llvm-objcopy  --remove-section .hash        vector-add-2.co
llvm-objcopy  --remove-section .dynsym      vector-add-2.co
llvm-objcopy  --remove-section .dynamic     vector-add-2.co
llvm-objcopy  --remove-section .dynstr      vector-add-2.co
```

## remove useless program header
in rocm ABI, several program header is not used. The only usefull program header should be phdr point to .text and .note, hence other useless program header can be ommited.

There are no available tool to remove such program header, but according to [ELF wiki](https://en.wikipedia.org/wiki/Executable_and_Linkable_Format), you can set `p_type` of each program header to `PT_NULL` to ignore this header.

after this step, the command `readelf  -l vector-add-2.co` will looks like:
```
Program Headers:
  Type           Offset             VirtAddr           PhysAddr
                 FileSiz            MemSiz              Flags  Align
  NULL           0x0000000000000040 0x0000000000000040 0x0000000000000040
                 0x00000000000001c0 0x00000000000001c0  R      8
  NULL           0x0000000000000000 0x0000000000000000 0x0000000000000000
                 0x000000000000028c 0x000000000000028c  R      1000
  LOAD           0x0000000000000300 0x0000000000001000 0x0000000000001000
                 0x00000000000001a8 0x00000000000001a8  R E    100
  NULL           0x0000000000001000 0x0000000000002000 0x0000000000002000
                 0x0000000000000070 0x0000000000000070  RW     1000
  NULL           0x0000000000001000 0x0000000000002000 0x0000000000002000
                 0x0000000000000070 0x0000000000000070  RW     8
  NULL           0x0000000000001000 0x0000000000002000 0x0000000000002000
                 0x0000000000000070 0x0000000000001000  R      1
  NULL           0x0000000000000000 0x0000000000000000 0x0000000000000000
                 0x0000000000000000 0x0000000000000000  RW     0
  NOTE           0x0000000000000248 0x0000000000000248 0x0000000000000248
                 0x0000000000000044 0x0000000000000044  R      4
```

## re-organize, remove useless file content
In fact, there are many useless space in the ELF binary even we do above steps. so a final re-organization of the ELF fille it self is needed to prune useless file content

At this stage, it's better to write your own assembler to generate the ELF file, to have a better control of all the addr/offset calculateion. But here is only an illustration of the final result of this step(note the field of prog/sec/elf header need recalculate, i.e. p_offset):
```
# file structure after remove useless program header

        ELF_BINARY         addr_in_file
    +-------------------+  0x0
    |   ELF_HEADER      |     e_phoff=0x40, e_phnum=8
    |                   |     e_shoff=0x1110, e_shnum=6
    +-------------------+  0x40
    |   phdr:NULL       |
    +-------------------+  0x78
    |   phdr:NULL       |
    +-------------------+  0xb0
    |   phdr:LOAD(.text)|   p_offset=0x300,p_vaddr=0x1000
    |                   |   p_paddr=0x1000,,p_align=0x100
    +-------------------+  0xe8
    |   phdr:NULL       |
    +-------------------+  0x120
    |   phdr:NULL       |
    +-------------------+  0x158
    |   phdr:NULL       |
    +-------------------+  0x190
    |   phdr:NULL       |
    +-------------------+  0x1c8
    |   phdr:NOTE(.note)|    p_offset=0x248,p_vaddr=0x248
    |                   |    p_paddr=0x248,p_align=4
    +-------------------+  0x200
    |                   |
    +-------------------+  0x248
    |   .note           |
    +-------------------+  0x28c
    |                   |
    +-------------------+  0x300
    |   .text           |
    +-------------------+  0x4a8
    |                   |
    +-------------------+  0x1070
    |   .symtab         |
    +-------------------+  0x10d0
    |   .shstrtab       |
    +-------------------+  0x10f7(not align 2 power)
    |   .strtab         |
    +-------------------+  0x1110
    |    shdr:NULL      |     
    |                   |     
    +-------------------+  0x1150
    |    shdr:note      |     sh_addr=0x248, sh_offset=0x248
    |                   |     sh_size=0x44, sh_addralign=4
    +-------------------+  0x1190
    |    shdr:text      |     sh_addr=0x1000, sh_offset=0x300
    |                   |     sh_size=0x1a8, sh_addralign=256
    +-------------------+  0x11d0
    |    shdr:symtab    |     sh_addr=0, sh_offset=0x1070
    |                   |     sh_size=0x60, sh_addralign=8
    +-------------------+  0x1210
    |    shdr:shstrtab  |     sh_addr=0, sh_offset=0x10d0
    |                   |     sh_size=0x27, sh_addralign=1
    +-------------------+  0x1250
    |    shdr:strtab    |     sh_addr=0, sh_offset=0x10f7
    |                   |     sh_size=0x19, sh_addralign=1
    +-------------------+  0x1290
```
-> above result is in this file [vector-add-2.co_v1](blob/vector-add-2.co_v1), 4752(0x1290) byte


after manually re-organize file structure and modify the addr/offset:
```
# file structure after final prune

        ELF_BINARY         addr_in_file
    +-------------------+  0x0
    |   ELF_HEADER      |     e_phoff=0x40, e_phnum=2
    |                   |     e_shoff=0x350, e_shnum=6
    +-------------------+  0x40
    |   phdr:LOAD(.text)|   p_offset=0x100,p_vaddr=0x1000
    |                   |   p_paddr=0x1000,p_align=0x100
    +-------------------+  0x78
    |   phdr:NOTE(.note)|    p_offset=0xb0,p_vaddr=0xb0
    |                   |    p_paddr=0xb0,p_align=4
    +-------------------+  0xb0
    |   .note           |
    +-------------------+  0xfc
    |                   | 
    +-------------------+  0x100
    |   .text           |
    +-------------------+  0x2a8
    |                   |
    +-------------------+  0x2b0
    |   .symtab         |
    +-------------------+  0x310
    |   .shstrtab       |
    +-------------------+  0x337(not align 2 power)
    |   .strtab         |
    +-------------------+  0x350
    |    shdr:NULL      |     
    |                   |     
    +-------------------+  0x390
    |    shdr:note      |     sh_addr=0xb0, sh_offset=0xb0
    |                   |     sh_size=0x44, sh_addralign=4
    +-------------------+  0x3d0
    |    shdr:text      |     sh_addr=0x1000, sh_offset=0x100
    |                   |     sh_size=0x1a8, sh_addralign=256
    +-------------------+  0x410
    |    shdr:symtab    |     sh_addr=0, sh_offset=0x2b0
    |                   |     sh_size=0x60, sh_addralign=8
    +-------------------+  0x450
    |    shdr:shstrtab  |     sh_addr=0, sh_offset=0x310
    |                   |     sh_size=0x27, sh_addralign=1
    +-------------------+  0x490
    |    shdr:strtab    |     sh_addr=0, sh_offset=0x337
    |                   |     sh_size=0x19, sh_addralign=1
    +-------------------+  0x4d0

```
above result is in this file [vector-add-2.co_v2](blob/vector-add-2.co_v2), 1232(0x4d0) byte

To consider how much we save the binary size, we have:

* vector-add-2.co, 9368 byte (original gen by LLVM-7.0)
* vector-add-2.co_v1, 4752 byte (gen by remove sec header)
* vector-add-2.co_v2, 1232 byte (gen by manually re-orgnize ELF file)

we have about **13.2%** (1232/4752) of origin file size.

## note
`e_entry` of ELF header and `p_vaddr` of text phdr, `sh_addr` of .text shdr should all be `0x1000`, 4096, size of a page. This is virtual addr where code should be loaded in memory.