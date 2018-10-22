# kernel-launcher-amdgpu

launcher kernel for amdgpu. using HSA/OCL for backend

for disassemble:
```
llvm-objdump -disassemble -mcpu=gfx900 asm-kernel.o
llvm-objdump -disassemble-all -mcpu=gfx900 asm-kernel.o
```
