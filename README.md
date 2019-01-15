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
