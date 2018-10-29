#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>

/******************************************************************/
// https://llvm.org/doxygen/AMDKernelCodeT_8h_source.html

typedef struct amd_kernel_code_s {
   uint32_t amd_kernel_code_version_major;
   uint32_t amd_kernel_code_version_minor;
   uint16_t amd_machine_kind;
   uint16_t amd_machine_version_major;
   uint16_t amd_machine_version_minor;
   uint16_t amd_machine_version_stepping;
 
   /// Byte offset (possibly negative) from start of amd_kernel_code_t
   /// object to kernel's entry point instruction. The actual code for
   /// the kernel is required to be 256 byte aligned to match hardware
   /// requirements (SQ cache line is 16). The code must be position
   /// independent code (PIC) for AMD devices to give runtime the
   /// option of copying code to discrete GPU memory or APU L2
   /// cache. The Finalizer should endeavour to allocate all kernel
   /// machine code in contiguous memory pages so that a device
   /// pre-fetcher will tend to only pre-fetch Kernel Code objects,
   /// improving cache performance.
   int64_t kernel_code_entry_byte_offset;
 
   /// Range of bytes to consider prefetching expressed as an offset
   /// and size. The offset is from the start (possibly negative) of
   /// amd_kernel_code_t object. Set both to 0 if no prefetch
   /// information is available.
   int64_t kernel_code_prefetch_byte_offset;
   uint64_t kernel_code_prefetch_byte_size;
 
   /// Reserved. Must be 0.
   uint64_t reserved0;
 
   /// Shader program settings for CS. Contains COMPUTE_PGM_RSRC1 and
   /// COMPUTE_PGM_RSRC2 registers.
   uint64_t compute_pgm_resource_registers;
 
   /// Code properties. See amd_code_property_mask_t for a full list of
   /// properties.
   uint32_t code_properties;
 
   /// The amount of memory required for the combined private, spill
   /// and arg segments for a work-item in bytes. If
   /// is_dynamic_callstack is 1 then additional space must be added to
   /// this value for the call stack.
   uint32_t workitem_private_segment_byte_size;
 
   /// The amount of group segment memory required by a work-group in
   /// bytes. This does not include any dynamically allocated group
   /// segment memory that may be added when the kernel is
   /// dispatched.
   uint32_t workgroup_group_segment_byte_size;
 
   /// Number of byte of GDS required by kernel dispatch. Must be 0 if
   /// not using GDS.
   uint32_t gds_segment_byte_size;
 
   /// The size in bytes of the kernarg segment that holds the values
   /// of the arguments to the kernel. This could be used by CP to
   /// prefetch the kernarg segment pointed to by the dispatch packet.
   uint64_t kernarg_segment_byte_size;
 
   /// Number of fbarrier's used in the kernel and all functions it
   /// calls. If the implementation uses group memory to allocate the
   /// fbarriers then that amount must already be included in the
   /// workgroup_group_segment_byte_size total.
   uint32_t workgroup_fbarrier_count;
 
   /// Number of scalar registers used by a wavefront. This includes
   /// the special SGPRs for VCC, Flat Scratch Base, Flat Scratch Size
   /// and XNACK (for GFX8 (VI)). It does not include the 16 SGPR added if a
   /// trap handler is enabled. Used to set COMPUTE_PGM_RSRC1.SGPRS.
   uint16_t wavefront_sgpr_count;
 
   /// Number of vector registers used by each work-item. Used to set
   /// COMPUTE_PGM_RSRC1.VGPRS.
   uint16_t workitem_vgpr_count;
 
   /// If reserved_vgpr_count is 0 then must be 0. Otherwise, this is the
   /// first fixed VGPR number reserved.
   uint16_t reserved_vgpr_first;
 
   /// The number of consecutive VGPRs reserved by the client. If
   /// is_debug_supported then this count includes VGPRs reserved
   /// for debugger use.
   uint16_t reserved_vgpr_count;
 
   /// If reserved_sgpr_count is 0 then must be 0. Otherwise, this is the
   /// first fixed SGPR number reserved.
   uint16_t reserved_sgpr_first;
 
   /// The number of consecutive SGPRs reserved by the client. If
   /// is_debug_supported then this count includes SGPRs reserved
   /// for debugger use.
   uint16_t reserved_sgpr_count;
 
   /// If is_debug_supported is 0 then must be 0. Otherwise, this is the
   /// fixed SGPR number used to hold the wave scratch offset for the
   /// entire kernel execution, or uint16_t(-1) if the register is not
   /// used or not known.
   uint16_t debug_wavefront_private_segment_offset_sgpr;
 
   /// If is_debug_supported is 0 then must be 0. Otherwise, this is the
   /// fixed SGPR number of the first of 4 SGPRs used to hold the
   /// scratch V# used for the entire kernel execution, or uint16_t(-1)
   /// if the registers are not used or not known.
   uint16_t debug_private_segment_buffer_sgpr;
 
   /// The maximum byte alignment of variables used by the kernel in
   /// the specified memory segment. Expressed as a power of two. Must
   /// be at least HSA_POWERTWO_16.
   uint8_t kernarg_segment_alignment;
   uint8_t group_segment_alignment;
   uint8_t private_segment_alignment;
 
   /// Wavefront size expressed as a power of two. Must be a power of 2
   /// in range 1..64 inclusive. Used to support runtime query that
   /// obtains wavefront size, which may be used by application to
   /// allocated dynamic group memory and set the dispatch work-group
   /// size.
   uint8_t wavefront_size;
 
   int32_t call_convention;
   uint8_t reserved3[12];
   uint64_t runtime_loader_kernel_symbol;
   uint64_t control_directives[16];
} amd_kernel_code_t;

#define AMD_KERNEL_CODE_T_BYTES 256

/******************************************************************/

typedef struct {
    char * file_name;
    FILE * fd;
    size_t file_len;

    int is_x64;
    int is_little_endian;

    uint64_t section_offset;
    uint64_t section_byte;
} elf_handle_t;

static int elf_handle_dtor(elf_handle_t * handle){
    if(handle->file_name)
        free(handle->file_name);
    if(handle->fd)
        fclose(handle->fd);
    return 0;
}

#define ELF_MAGIC 0x464c457f
static int elf_valid_magic(elf_handle_t * handle){
    uint32_t magic;
    fseek (handle->fd, 0, SEEK_SET);
    fread(&magic, 1, 4, handle->fd);
    if(magic == ELF_MAGIC)
        return 1;
    return 0;
}

static int parse_elf_header_eident(elf_handle_t * handle){
    // the first 16 byte is not change
    uint8_t header[16];
    fseek (handle->fd, 0, SEEK_SET);
    fread(header, 1, 16, handle->fd);

    handle->is_x64 = header[4] == 2 ? 1:0;
    handle->is_little_endian = header[5] == 1 ? 1:0;

    return 0;
}

static int elf_handle_init(elf_handle_t * handle, char * file_name){
    FILE * fd = fopen (file_name, "rb");
    size_t len;
    if(!fd){
        printf("fail to open elf file %s\n", file_name);
        return -1;
    }
    fseek (fd, 0, SEEK_END);
    len = ftell (fd);
    fseek (fd, 0, SEEK_SET);

    handle->file_name = strdup(file_name);
    handle->fd = fd;
    handle->file_len = len;

    if (!elf_valid_magic(handle)){
        printf("not valid elf file\n");
        elf_handle_dtor(handle);
        return -1;
    }

    parse_elf_header_eident(handle);

    return 0;
}
// assume dst le
static void get_byte_le(uint8_t * dst, uint8_t * src, size_t bytes){
    size_t idx = 0;
    for(idx = 0;idx < bytes; idx++){
        dst[idx] = src[idx];
    }
}
// assume dst le
static void get_byte_be(uint8_t * dst, uint8_t * src, size_t bytes){
    size_t idx = 0;
    for(idx = 0;idx < bytes; idx++){
        dst[idx] = src[bytes-1-idx];
    }
}

// consider le/be
static void get_byte_from_file(void * dst, long offset, size_t bytes, FILE * fd, int is_le){
    fseek(fd, offset, SEEK_SET );
    if(is_le){
        fread(dst, bytes, 1, fd);
    }else{
        // TODO: better alloc
        uint8_t * tmp = (uint8_t *)malloc(bytes);
        fread(tmp, bytes, 1, fd);
        get_byte_be((uint8_t *)dst, tmp, bytes );
        free(tmp);
    }
}
// pure copy content to dst
static void get_content_from_file(void * dst, long offset, size_t bytes, FILE * fd){
    fseek(fd, offset, SEEK_SET );
    fread(dst, bytes, 1, fd);
}

static int _next_string(const char * str, const int cur_idx, const int max_idx){
    int idx = cur_idx;
    while(str[idx] != '\0' && idx < max_idx)
        idx++;
    while(str[idx] == '\0' && idx < max_idx)
        idx++;
    return idx;
}
static int _locate_section_index(const char * strtbl, int len){
    // \0 seperated string
    int idx = 0;
    int section = 0;
    char * str;
    char * sec_name;
    int sec_name_len;

    sec_name = getenv("TARGET_SECTION");
    if(!sec_name){
        sec_name = ".text";
    }
    sec_name_len = strlen(sec_name);
    while(1){
        str  = (char*)strtbl + idx;
        if(!strncmp(str, sec_name, sec_name_len)){
            break;
        }
        idx = _next_string(strtbl, idx, len);
        if(idx >= len){
            section = -1;   // not found
            break;
        }
        section++;
    }
    return section;
}

static int elf_locate_text_sec(elf_handle_t * handle){
    FILE *   fd = handle->fd;
    int      is_le = handle->is_little_endian;
    long     offset;
    size_t   bytes;

    uint16_t e_shstrndx = 0;
    uint64_t e_shoff = 0;

    uint8_t * strtbl_content = 0;
    uint64_t  strtbl_bytes = 0;
    uint64_t  strtbl_offset = 0;

    int text_section_index = 0;

    // e_shoff, Points to the start of the section header table.
    offset = handle->is_x64 ? 0x28:0x20;
    bytes  = handle->is_x64 ? 8:4;
    get_byte_from_file(&e_shoff, offset, bytes, fd, is_le);

    // e_shstrndx, Contains index of the section header table entry that contains the section names.
    offset = handle->is_x64 ? 0x3E:0x32;
    bytes = 2;
    get_byte_from_file(&e_shstrndx, offset, bytes, fd, is_le);

    // get strtbl size, .sh_size off section header of strtbl
    offset = e_shoff + (handle->is_x64 ? 0x40:0x28) * e_shstrndx + 
                        (handle->is_x64 ? 0x20:0x14) /*sh_size offset*/;
    bytes = handle->is_x64 ? 8:4;
    get_byte_from_file(&strtbl_bytes, offset, bytes, fd, is_le);

    // get strtbl offset
    offset = e_shoff + (handle->is_x64 ? 0x40:0x28) * e_shstrndx + 
                        (handle->is_x64 ? 0x18:0x10) /*sh_offset offset*/;
    bytes = handle->is_x64 ? 8:4;
    get_byte_from_file(&strtbl_offset, offset, bytes, fd, is_le);

    strtbl_content = (uint8_t *)malloc(strtbl_bytes);
    assert(strtbl_content);
    get_content_from_file(strtbl_content, strtbl_offset, strtbl_bytes, fd);

    text_section_index = _locate_section_index((const char*)strtbl_content, strtbl_bytes);
    if(text_section_index<0){
        printf("can't find text section, should not happen\n");
        free(strtbl_content);
        return -1;
    }

    // get text section size
    offset = e_shoff + (handle->is_x64 ? 0x40:0x28) * text_section_index + 
                        (handle->is_x64 ? 0x20:0x14) /*sh_size offset*/;
    bytes = handle->is_x64 ? 8:4;
    get_byte_from_file(&handle->section_byte, offset, bytes, fd, is_le);

    // get text section offset
    offset = e_shoff + (handle->is_x64 ? 0x40:0x28) * text_section_index + 
                        (handle->is_x64 ? 0x18:0x10) /*sh_offset offset*/;
    bytes = handle->is_x64 ? 8:4;
    get_byte_from_file(&handle->section_offset, offset, bytes, fd, is_le);

    free(strtbl_content);
    return 0;
}


template<typename T>
void dump_field(const char * label, T v){
    std::cout<<label<< v<<std::endl;
}
template<>
void dump_field(const char * label, uint8_t v){
    // avoid cout print uint8_t as ascii
    std::cout<<label<<(uint32_t)v<<std::endl;
}

//#define DUMP_FIELD(ptr, field)    std::cout<<"  ." #field " = "<<ptr->field<<std::endl
#define DUMP_FIELD(ptr, field)  dump_field("  " #field " = ", ptr->field)
#define GET_BIT(u32, bit) (  ( (u32) & (1<<(bit)) )?  1:0  )
#define GET_NUM_BITS(u32, offset, num_bit) ( ((u32)>>offset) & ((0x1UL<<num_bit) - 1) )

static int _dump_amd_kernel_code_t(amd_kernel_code_t * kernel_code){
    std::cout<<".amd_kernel_code_t"<<std::endl;
    // https://github.com/ROCm-Developer-Tools/ROCm-ComputeABI-Doc/blob/master/AMDGPU-ABI.md#amd-kernel-code-object-amd_kernel_code_t
    // AMDKernelCodeTInfo.h, following print should match that in this header. current use llvm-7.0 source
    DUMP_FIELD(kernel_code, amd_kernel_code_version_major);
    DUMP_FIELD(kernel_code, amd_kernel_code_version_minor);
    DUMP_FIELD(kernel_code, amd_machine_kind);
    DUMP_FIELD(kernel_code, amd_machine_version_major);
    DUMP_FIELD(kernel_code, amd_machine_version_minor);
    DUMP_FIELD(kernel_code, amd_machine_version_stepping);
    DUMP_FIELD(kernel_code, kernel_code_entry_byte_offset);
    DUMP_FIELD(kernel_code, kernel_code_prefetch_byte_offset);
    DUMP_FIELD(kernel_code, kernel_code_prefetch_byte_size);

    //DUMP_FIELD(kernel_code, reserved0);
    //8 bytes  max_scratch_backing_memory_byte_size
    std::cout<<"  max_scratch_backing_memory_byte_size = "<<kernel_code->reserved0<<std::endl;
    {
        //DUMP_FIELD(kernel_code, compute_pgm_resource_registers);
        uint32_t compute_pgm_rsrc1, compute_pgm_rsrc2;
        compute_pgm_rsrc1 = (uint32_t)((kernel_code->compute_pgm_resource_registers)&0xffffffff);
        compute_pgm_rsrc2 = (uint32_t)((kernel_code->compute_pgm_resource_registers>>32)&0xffffffff);
        //std::cout<<"  compute_pgm_rsrc1 = "<<compute_pgm_rsrc1<<std::endl;
        //std::cout<<"  compute_pgm_rsrc2 = "<<compute_pgm_rsrc2<<std::endl;

        // AMDKernelCodeTInfo.h, we use alias name here
        std::cout<<"  compute_pgm_rsrc1_vgprs = "      <<GET_NUM_BITS(compute_pgm_rsrc1, 0, 6)<<std::endl;
        std::cout<<"  compute_pgm_rsrc1_sgprs = "      <<GET_NUM_BITS(compute_pgm_rsrc1, 6, 4)<<std::endl;
        std::cout<<"  compute_pgm_rsrc1_priority = "   <<GET_NUM_BITS(compute_pgm_rsrc1, 10, 2)<<std::endl;
        std::cout<<"  compute_pgm_rsrc1_float_mode = " <<GET_NUM_BITS(compute_pgm_rsrc1, 12, 8)<<std::endl;
        std::cout<<"  compute_pgm_rsrc1_priv = "       <<GET_NUM_BITS(compute_pgm_rsrc1, 20, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc1_dx10_clamp = " <<GET_NUM_BITS(compute_pgm_rsrc1, 21, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc1_debug_mode = " <<GET_NUM_BITS(compute_pgm_rsrc1, 22, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc1_ieee_mode = "  <<GET_NUM_BITS(compute_pgm_rsrc1, 23, 1)<<std::endl;
        // TODO: bulky cdbg_user

        std::cout<<"  compute_pgm_rsrc2_scratch_en = " <<GET_NUM_BITS(compute_pgm_rsrc2, 0, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc2_user_sgpr = "  <<GET_NUM_BITS(compute_pgm_rsrc2, 1, 5)<<std::endl;
        std::cout<<"  compute_pgm_rsrc2_trap_handler = " <<GET_NUM_BITS(compute_pgm_rsrc2, 6, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc2_tgid_x_en = "  <<GET_NUM_BITS(compute_pgm_rsrc2, 7, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc2_tgid_y_en = "  <<GET_NUM_BITS(compute_pgm_rsrc2, 8, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc2_tgid_z_en = "  <<GET_NUM_BITS(compute_pgm_rsrc2, 9, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc2_tg_size_en = " <<GET_NUM_BITS(compute_pgm_rsrc2, 10, 1)<<std::endl;
        std::cout<<"  compute_pgm_rsrc2_tidig_comp_cnt = " <<GET_NUM_BITS(compute_pgm_rsrc2, 11, 2)<<std::endl;
        std::cout<<"  compute_pgm_rsrc2_excp_en_msb = " <<GET_NUM_BITS(compute_pgm_rsrc2, 13, 2)<<std::endl;
        // TODO: split enable_exception_msb
        std::cout<<"  compute_pgm_rsrc2_lds_size = "   <<GET_NUM_BITS(compute_pgm_rsrc2, 15, 9)<<std::endl;
        // TODO: split enable_exception
        std::cout<<"  compute_pgm_rsrc2_excp_en = "    <<GET_NUM_BITS(compute_pgm_rsrc2, 24, 7)<<std::endl;

    }
    {
        //DUMP_FIELD(kernel_code, code_properties);
        /*
        448	1 bit	enable_sgpr_private_segment_buffer	Enable the setup of Private Segment Buffer
        449	1 bit	enable_sgpr_dispatch_ptr	Enable the setup of Dispatch Ptr
        450	1 bit	enable_sgpr_queue_ptr	Enable the setup of Queue Ptr
        451	1 bit	enable_sgpr_kernarg_segment_ptr	Enable the setup of Kernarg Segment Ptr
        452	1 bit	enable_sgpr_dispatch_id	Enable the setup of Dispatch Id
        453	1 bit	enable_sgpr_flat_scratch_init	Enable the setup of Flat Scratch Init
        454	1 bit	enable_sgpr_private_segment_size	Enable the setup of Private Segment Size
        455	1 bit	enable_sgpr_grid_workgroup_count_X	Enable the setup of Grid Work-Group Count X
        456	1 bit	enable_sgpr_grid_workgroup_count_Y	Enable the setup of Grid Work-Group Count Y
        457	1 bit	enable_sgpr_grid_workgroup_count_Z	Enable the setup of Grid Work-Group Count Z
        463:458	6 bits		Reserved. Must be 0.
        464	1 bit	enable_ordered_append_gds	Control wave ID base counter for GDS ordered-append. Used to set COMPUTE_DISPATCH_INITIATOR.ORDERED_APPEND_ENBL.
        466:465	2 bits	private_element_size	Interleave (swizzle) element size in bytes.
        467	1 bit	is_ptr64	1 if global memory addresses are 64 bits, otherwise 0. Must match SH_MEM_CONFIG.PTR32 (GFX7), SH_MEM_CONFIG.ADDRESS_MODE (GFX8+).
        468	1 bit	is_dynamic_call_stack	Indicates if the generated machine code is using dynamic call stack.
        469	1 bit	is_debug_enabled	Indicates if the generated machine code includes code required by the debugger.
        470	1 bit	is_xnack_enabled	Indicates if the generated machine code uses conservative XNACK register allocation.
        479:471	9 bits	reserved	Reserved. Must be 0.
        */
       uint32_t u32 = kernel_code->code_properties;
       std::cout<<"  enable_sgpr_private_segment_buffer = " << GET_BIT(u32, 0) <<std::endl;
       std::cout<<"  enable_sgpr_dispatch_ptr = "           << GET_BIT(u32, 1) <<std::endl;
       std::cout<<"  enable_sgpr_queue_ptr = "              << GET_BIT(u32, 2) <<std::endl;
       std::cout<<"  enable_sgpr_kernarg_segment_ptr = "    << GET_BIT(u32, 3) <<std::endl;
       std::cout<<"  enable_sgpr_dispatch_id = "            << GET_BIT(u32, 4) <<std::endl;
       std::cout<<"  enable_sgpr_flat_scratch_init = "      << GET_BIT(u32, 5) <<std::endl;
       std::cout<<"  enable_sgpr_private_segment_size = "   << GET_BIT(u32, 6) <<std::endl;
       std::cout<<"  enable_sgpr_grid_workgroup_count_x = " << GET_BIT(u32, 7) <<std::endl;
       std::cout<<"  enable_sgpr_grid_workgroup_count_y = " << GET_BIT(u32, 8) <<std::endl;
       std::cout<<"  enable_sgpr_grid_workgroup_count_z = " << GET_BIT(u32, 9) <<std::endl;
       std::cout<<"  enable_ordered_append_gds = "          << GET_BIT(u32, 16) <<std::endl;
       //std::cout<<"  private_element_size = "               << ((u32>>17) & 0x3) <<std::endl;
       std::cout<<"  private_element_size = "               << GET_NUM_BITS(u32, 17, 2) <<std::endl;
       std::cout<<"  is_ptr64 = "                           << GET_BIT(u32, 19) <<std::endl;
       std::cout<<"  is_dynamic_call_stack = "              << GET_BIT(u32, 20) <<std::endl;
       std::cout<<"  is_debug_enabled = "                   << GET_BIT(u32, 21) <<std::endl;
       std::cout<<"  is_xnack_enabled = "                   << GET_BIT(u32, 22) <<std::endl;
    }
    DUMP_FIELD(kernel_code, workitem_private_segment_byte_size);
    DUMP_FIELD(kernel_code, workgroup_group_segment_byte_size);
    DUMP_FIELD(kernel_code, gds_segment_byte_size);
    DUMP_FIELD(kernel_code, kernarg_segment_byte_size);
    DUMP_FIELD(kernel_code, workgroup_fbarrier_count);
    DUMP_FIELD(kernel_code, wavefront_sgpr_count);
    DUMP_FIELD(kernel_code, workitem_vgpr_count);
    DUMP_FIELD(kernel_code, reserved_vgpr_first);
    DUMP_FIELD(kernel_code, reserved_vgpr_count);
    DUMP_FIELD(kernel_code, reserved_sgpr_first);
    DUMP_FIELD(kernel_code, reserved_sgpr_count);
    DUMP_FIELD(kernel_code, debug_wavefront_private_segment_offset_sgpr);
    DUMP_FIELD(kernel_code, debug_private_segment_buffer_sgpr);
    DUMP_FIELD(kernel_code, kernarg_segment_alignment);
    DUMP_FIELD(kernel_code, group_segment_alignment);
    DUMP_FIELD(kernel_code, private_segment_alignment);
    DUMP_FIELD(kernel_code, wavefront_size);
    DUMP_FIELD(kernel_code, call_convention);
    //DUMP_FIELD(kernel_code, reserved3[12]);
    DUMP_FIELD(kernel_code, runtime_loader_kernel_symbol);
    //DUMP_FIELD(kernel_code, control_directives[16]);
    std::cout<<".end_amd_kernel_code_t"<<std::endl;
    return 0;
}

static int elf_dump_amd_kernel_code_t(elf_handle_t * handle){
    char * v = getenv("PRINT_SECTION");
    if(v){
        // dump the section content

    }else{
        uint8_t * content;
        amd_kernel_code_t kernel_code;

        content = (uint8_t *)malloc(handle->section_byte);
        assert(content);
        get_content_from_file(content, handle->section_offset, 
                        handle->section_byte, handle->fd);
        
        assert(sizeof(kernel_code) == AMD_KERNEL_CODE_T_BYTES);
        memcpy(&kernel_code, content, AMD_KERNEL_CODE_T_BYTES);

        printf(".text section offset:%x, len:%lu\n", handle->section_offset, handle->section_byte);
        _dump_amd_kernel_code_t(&kernel_code);
        free(content);
    }

    return 0;
}

static void usage(){
    printf("usage: ./a.out <file>\n" );
}

int main(int argc, char ** argv){
    int rtn;
    elf_handle_t handle;

    if(argc < 2){
        usage();
        return -1;
    }
    rtn = elf_handle_init(&handle, argv[1]);
    if(rtn != 0 )
        return rtn;
    
    rtn = elf_locate_text_sec(&handle);
    if(rtn != 0)
        goto out;

    elf_dump_amd_kernel_code_t(&handle);

out:
    elf_handle_dtor(&handle);
    return rtn;
}