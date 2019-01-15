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
amdgpu assembly need store some infomation in [amd_kernel_code_t](http://llvm.org/docs/AMDGPUUsage.html#amd-kernel-code-t), in `.text` section. [amd_kernel_code_t.cpp](utils/dump_amd_kernel_code_t/amd_kernel_code_t.cpp) is a util to dump this section from a elf file.


# remove useless section
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