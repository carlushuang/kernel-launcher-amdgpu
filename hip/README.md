recipes for simple hip kernel

to dissemble the output file:

/opt/rocm/bin/extractkernel -i <exe>

will generate <exe>-gfx900.hasco, <exe>-gfx900.isa, <exe>.bundle
<exe>-gfx900.isa is the asm code of text segment

or you can use following command to dump:
llvm-objdump -disassemble -mcpu=gfx900 <exe>-gfx900.hasco