.hsa_code_object_version 2,0
.hsa_code_object_isa 9, 0, 0, "AMD", "AMDGPU"

.text
.p2align 8
.amdgpu_hsa_kernel vector_add

// include some helper macros
//.include "gpr_alloc.inc"
//.include "common.inc"
//.include "inst_wrappers.inc"

// GPR allocation
//.GPR_ALLOC_BEGIN


//.GPR_ALLOC_END

vector_add:
    .amd_kernel_code_t
        // on init, this struct is zero memset-ed, and some will set default value
        // default value:http://llvm.org/docs/AMDGPUUsage.html#amd-kernel-code-t

        // amd_kernel_code_version_major = 1
        // amd_kernel_code_version_minor = 1
        // amd_machine_kind = 1
        // amd_machine_version_major = 9
        // amd_machine_version_minor = 0
        // amd_machine_version_stepping = 0
        // kernel_code_entry_byte_offset = 256
        // kernel_code_prefetch_byte_offset = 0
        // kernel_code_prefetch_byte_size = 0
        // max_scratch_backing_memory_byte_size = 0
        compute_pgm_rsrc1_vgprs = 1         // max(0, ceil(vgprs_used / 4) - 1)
        compute_pgm_rsrc1_sgprs = 1         // gfx9: 2 * max(0, ceil(sgprs_used / 16) - 1), gfx6-8: max(0, ceil(sgprs_used / 8) - 1)
        // compute_pgm_rsrc1_priority = 0   // default 0
        compute_pgm_rsrc1_float_mode = 192
        // compute_pgm_rsrc1_priv = 0
        // compute_pgm_rsrc1_dx10_clamp = 1  // default 1
        // compute_pgm_rsrc1_debug_mode = 0
        // compute_pgm_rsrc1_ieee_mode = 1  // default 1
        // compute_pgm_rsrc2_scratch_en = 0
        compute_pgm_rsrc2_user_sgpr = 8
        compute_pgm_rsrc2_trap_handler = 1      //?
        compute_pgm_rsrc2_tgid_x_en = 1         // ENABLE_SGPR_WORKGROUP_ID_X
        // compute_pgm_rsrc2_tgid_y_en = 0
        // compute_pgm_rsrc2_tgid_z_en = 0
        // compute_pgm_rsrc2_tg_size_en = 0
        // compute_pgm_rsrc2_tidig_comp_cnt = 0
        // compute_pgm_rsrc2_excp_en_msb = 0
        // compute_pgm_rsrc2_lds_size = 0
        // compute_pgm_rsrc2_excp_en = 0
        enable_sgpr_private_segment_buffer = 1
        enable_sgpr_dispatch_ptr = 1
        // enable_sgpr_queue_ptr = 0
        enable_sgpr_kernarg_segment_ptr = 1
        // enable_sgpr_dispatch_id = 0
        // enable_sgpr_flat_scratch_init = 0
        // enable_sgpr_private_segment_size = 0
        // enable_sgpr_grid_workgroup_count_x = 0
        // enable_sgpr_grid_workgroup_count_y = 0
        // enable_sgpr_grid_workgroup_count_z = 0
        // enable_ordered_append_gds = 0
        private_element_size = 1
        is_ptr64 = 1
        // is_dynamic_call_stack = 0
        // is_debug_enabled = 0
        // is_xnack_enabled = 0
        // workitem_private_segment_byte_size = 0
        // workgroup_group_segment_byte_size = 0
        // gds_segment_byte_size = 0
        kernarg_segment_byte_size = 20
        // workgroup_fbarrier_count = 0
        wavefront_sgpr_count = 11
        workitem_vgpr_count = 7
        // reserved_vgpr_first = 0
        // reserved_vgpr_count = 0
        // reserved_sgpr_first = 0
        // reserved_sgpr_count = 0
        // debug_wavefront_private_segment_offset_sgpr = 0
        // debug_private_segment_buffer_sgpr = 0
        // kernarg_segment_alignment = 4  // default to 4
        // group_segment_alignment = 4  // default to 4
        // private_segment_alignment = 4  // default to 4
        // wavefront_size = 6  // default to 6  initDefaultAMDKernelCodeT()
        // call_convention = -1  // default to -1
        // runtime_loader_kernel_symbol = 0
		
    .end_amd_kernel_code_t

    s_load_dword s1, s[4:5], 0x4                               //
    s_load_dword s2, s[4:5], 0xc                               //
    s_load_dword s0, s[6:7], 0x10                              //
    s_waitcnt lgkmcnt(0)                                       //
    s_and_b32 s3, s1, 0xffff                                   //
    s_mul_i32 s1, s8, s3                                       //
    s_sub_i32 s1, s2, s1                                       //
    s_min_u32 s1, s1, s3                                       //
    s_mul_i32 s4, s1, s8                                       //
    v_add_u32_e32 v0, s4, v0                                   //
    v_cmp_gt_i32_e32 vcc, s0, v0                               //
    s_and_saveexec_b64 s[4:5], vcc                             //
    s_cbranch_execz BB0_3                                      //

BB0_1:
    v_cvt_f32_u32_e32 v1, s3                                   //
    v_rcp_iflag_f32_e32 v1, v1                                 //
    v_mul_f32_e32 v1, 0x4f800000, v1                           //
    v_cvt_u32_f32_e32 v1, v1                                   //
    v_mul_lo_u32 v2, v1, s3                                    //
    v_mul_hi_u32 v3, v1, s3                                    //
    v_sub_u32_e32 v4, 0, v2                                    //
    v_cmp_eq_u32_e32 vcc, 0, v3                                //
    v_cndmask_b32_e32 v2, v2, v4, vcc                          //
    v_mul_hi_u32 v2, v2, v1                                    //
    v_add_u32_e32 v3, v1, v2                                   //
    v_sub_u32_e32 v1, v1, v2                                   //
    v_cndmask_b32_e32 v1, v1, v3, vcc                          //
    v_mul_hi_u32 v1, v1, s2                                    //
    v_mul_lo_u32 v2, v1, s3                                    //
    v_add_u32_e32 v3, 1, v1                                    //
    v_add_u32_e32 v4, -1, v1                                   //
    v_cmp_ge_u32_e32 vcc, s2, v2                               //
    v_sub_u32_e32 v2, s2, v2                                   //
    v_cndmask_b32_e64 v5, 0, -1, vcc                           //
    v_cmp_le_u32_e32 vcc, s3, v2                               //
    v_cndmask_b32_e64 v2, 0, -1, vcc                           //
    v_and_b32_e32 v2, v2, v5                                   //
    v_cmp_eq_u32_e32 vcc, 0, v2                                //
    v_cndmask_b32_e32 v1, v3, v1, vcc                          //
    v_cmp_eq_u32_e32 vcc, 0, v5                                //
    v_cndmask_b32_e32 v1, v1, v4, vcc                          //
    s_load_dwordx2 s[2:3], s[6:7], 0x0                         //
    s_load_dwordx2 s[4:5], s[6:7], 0x8                         //
    v_mul_lo_u32 v2, v1, s1                                    //
    s_mov_b64 s[6:7], 0                                        //
    s_waitcnt lgkmcnt(0)                                       //

BB0_2:
    v_ashrrev_i32_e32 v1, 31, v0                               //
    v_lshlrev_b64 v[3:4], 2, v[0:1]                            //
    v_mov_b32_e32 v1, s3                                       //
    v_add_co_u32_e32 v5, vcc, s2, v3                           //
    v_addc_co_u32_e32 v6, vcc, v1, v4, vcc                     //
    v_mov_b32_e32 v1, s5                                       //
    v_add_co_u32_e32 v3, vcc, s4, v3                           //
    v_addc_co_u32_e32 v4, vcc, v1, v4, vcc                     //
    flat_load_dword v1, v[3:4]                                 //
    s_nop 0                                                    //
    flat_load_dword v5, v[5:6]                                 //
    v_add_u32_e32 v0, v0, v2                                   //
    v_cmp_le_i32_e32 vcc, s0, v0                               //
    s_or_b64 s[6:7], vcc, s[6:7]                               //
    s_waitcnt vmcnt(0) lgkmcnt(0)                              //
    v_add_f32_e32 v1, v5, v1                                   //
    flat_store_dword v[3:4], v1                                //
    s_andn2_b64 exec, exec, s[6:7]                             //
    s_waitcnt vmcnt(0) lgkmcnt(0)                              //
    s_cbranch_execnz BB0_2                                     //

BB0_3:
    s_endpgm                                                   //


.amd_amdgpu_hsa_metadata
  Version: [ 1, 0 ]
  Kernels:
    - Name: vector_add
      SymbolName: 'vector_add@kd'
      Language: OpenCL C
      LanguageVersion: [ 2, 0 ]
      Args:
        - Name: in
          Size: 8
          Align: 8
          ValueKind: GlobalBuffer
          ValueType: F32
          AddrSpaceQual: Generic
        - Name: out
          Size: 8
          Align: 8
          ValueKind: GlobalBuffer
          ValueType: F32
          AddrSpaceQual: Generic
        - Name: num
          Size: 4
          Align: 4
          ValueKind: ByValue
          ValueType: I32

.end_amd_amdgpu_hsa_metadata