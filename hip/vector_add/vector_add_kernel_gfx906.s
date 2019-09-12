	.text
	.hsa_code_object_version 2,1
	.hsa_code_object_isa 9,0,6,"AMD","AMDGPU"
	.weak	vector_add              ; -- Begin function vector_add
	.p2align	8
	.type	vector_add,@function
	.amdgpu_hsa_kernel vector_add
vector_add:                             ; @vector_add
	.amd_kernel_code_t
		amd_code_version_major = 1
		amd_code_version_minor = 2
		amd_machine_kind = 1
		amd_machine_version_major = 9
		amd_machine_version_minor = 0
		amd_machine_version_stepping = 6
		kernel_code_entry_byte_offset = 256
		kernel_code_prefetch_byte_size = 0
		granulated_workitem_vgpr_count = 1
		granulated_wavefront_sgpr_count = 4
		priority = 0
		float_mode = 192
		priv = 0
		enable_dx10_clamp = 1
		debug_mode = 0
		enable_ieee_mode = 1
		enable_wgp_mode = 0
		enable_mem_ordered = 0
		enable_fwd_progress = 0
		enable_sgpr_private_segment_wave_byte_offset = 0
		user_sgpr_count = 8
		enable_trap_handler = 0
		enable_sgpr_workgroup_id_x = 1
		enable_sgpr_workgroup_id_y = 0
		enable_sgpr_workgroup_id_z = 0
		enable_sgpr_workgroup_info = 0
		enable_vgpr_workitem_id = 0
		enable_exception_msb = 0
		granulated_lds_size = 0
		enable_exception = 0
		enable_sgpr_private_segment_buffer = 1
		enable_sgpr_dispatch_ptr = 1
		enable_sgpr_queue_ptr = 0
		enable_sgpr_kernarg_segment_ptr = 1
		enable_sgpr_dispatch_id = 0
		enable_sgpr_flat_scratch_init = 0
		enable_sgpr_private_segment_size = 0
		enable_sgpr_grid_workgroup_count_x = 0
		enable_sgpr_grid_workgroup_count_y = 0
		enable_sgpr_grid_workgroup_count_z = 0
		enable_wavefront_size32 = 0
		enable_ordered_append_gds = 0
		private_element_size = 1
		is_ptr64 = 1
		is_dynamic_callstack = 0
		is_debug_enabled = 0
		is_xnack_enabled = 0
		workitem_private_segment_byte_size = 0
		workgroup_group_segment_byte_size = 0
		gds_segment_byte_size = 0
		kernarg_segment_byte_size = 20
		workgroup_fbarrier_count = 0
		wavefront_sgpr_count = 35
		workitem_vgpr_count = 7
		reserved_vgpr_first = 0
		reserved_vgpr_count = 0
		reserved_sgpr_first = 0
		reserved_sgpr_count = 0
		debug_wavefront_private_segment_offset_sgpr = 0
		debug_private_segment_buffer_sgpr = 0
		kernarg_segment_alignment = 4
		group_segment_alignment = 4
		private_segment_alignment = 4
		wavefront_size = 6
		call_convention = -1
		runtime_loader_kernel_symbol = 0
	.end_amd_kernel_code_t
; %bb.0:                                ; %entry
	s_load_dword s1, s[4:5], 0x4
	s_load_dword s0, s[4:5], 0xc
	s_load_dword s2, s[6:7], 0x10
	s_mov_b32 s10, s9
	s_mov_b32 s32, s10
	s_waitcnt lgkmcnt(0)
	s_and_b32 s1, s1, 0xffff
	s_mul_i32 s3, s8, s1
	s_sub_i32 s3, s0, s3
	s_min_u32 s3, s3, s1
	s_mul_i32 s4, s3, s8
	v_add_u32_e32 v0, s4, v0
	v_cmp_gt_i32_e32 vcc, s2, v0
	s_and_saveexec_b64 s[4:5], vcc
	; mask branch BB0_3
	s_cbranch_execz BB0_3
BB0_1:                                  ; %for.body.lr.ph
	v_cvt_f32_u32_e32 v1, s1
	s_load_dwordx4 s[4:7], s[6:7], 0x0
	v_rcp_iflag_f32_e32 v1, v1
	v_mul_f32_e32 v1, 0x4f800000, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v2, v1, s1
	v_mul_hi_u32 v3, v1, s1
	v_sub_u32_e32 v4, 0, v2
	v_cmp_eq_u32_e32 vcc, 0, v3
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_mul_hi_u32 v2, v2, v1
	v_add_u32_e32 v3, v1, v2
	v_sub_u32_e32 v1, v1, v2
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_mul_hi_u32 v1, v1, s0
	v_mul_lo_u32 v2, v1, s1
	v_add_u32_e32 v3, 1, v1
	v_add_u32_e32 v4, -1, v1
	v_sub_u32_e32 v5, s0, v2
	v_cmp_ge_u32_e32 vcc, s0, v2
	v_cmp_le_u32_e64 s[0:1], s1, v5
	s_and_b64 s[0:1], s[0:1], vcc
	v_cndmask_b32_e64 v1, v1, v3, s[0:1]
	v_cndmask_b32_e32 v1, v4, v1, vcc
	v_mul_lo_u32 v2, v1, s3
	s_mov_b64 s[0:1], 0
BB0_2:                                  ; %for.body
                                        ; =>This Inner Loop Header: Depth=1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[3:4], 2, v[0:1]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s5
	v_add_co_u32_e32 v5, vcc, s4, v3
	v_addc_co_u32_e32 v6, vcc, v1, v4, vcc
	v_mov_b32_e32 v1, s7
	v_add_co_u32_e32 v3, vcc, s6, v3
	v_addc_co_u32_e32 v4, vcc, v1, v4, vcc
	global_load_dword v1, v[5:6], off
	global_load_dword v5, v[3:4], off
	v_add_u32_e32 v0, v0, v2
	v_cmp_le_i32_e32 vcc, s2, v0
	s_or_b64 s[0:1], vcc, s[0:1]
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v1, v1, v5
	global_store_dword v[3:4], v1, off
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz BB0_2
BB0_3:                                  ; %for.end
	s_endpgm
.Lfunc_end0:
	.size	vector_add, .Lfunc_end0-vector_add
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 316
; NumSgprs: 35
; NumVgprs: 7
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 1
; NumSGPRsForWavesPerEU: 35
; NumVGPRsForWavesPerEU: 7
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.protected	__ocml_ldexp_f16
	.protected	__ocml_nan_f16

	.ident	"HCC clang version 9.0.0 (/data/jenkins_workspace/compute-rocm-rel-2.7/external/hcc-tot/clang 5c2570257a8bbd74eff632fbc60692ef61ef8ecb) (/data/jenkins_workspace/compute-rocm-rel-2.7/external/hcc-tot/compiler 9a9477021e6998100ff64d1360dcfe64f65cebe5) (based on HCC 2.7.19315-346267d-5c25702-9a94770 )"
	.section	".note.GNU-stack"
	.amd_amdgpu_isa "amdgcn-amd-amdhsa--gfx906+sram-ecc"
	.amd_amdgpu_hsa_metadata
---
Version:         [ 1, 0 ]
Kernels:         
  - Name:            vector_add
    SymbolName:      'vector_add@kd'
    Language:        OpenCL C
    LanguageVersion: [ 2, 0 ]
    Args:            
      - Name:            in
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Generic
      - Name:            out
        Size:            8
        Align:           8
        ValueKind:       GlobalBuffer
        ValueType:       F32
        AddrSpaceQual:   Generic
      - Name:            num
        Size:            4
        Align:           4
        ValueKind:       ByValue
        ValueType:       I32
    CodeProps:       
      KernargSegmentSize: 20
      GroupSegmentFixedSize: 0
      PrivateSegmentFixedSize: 0
      KernargSegmentAlign: 8
      WavefrontSize:   64
      NumSGPRs:        35
      NumVGPRs:        7
      MaxFlatWorkGroupSize: 256
...

	.end_amd_amdgpu_hsa_metadata
