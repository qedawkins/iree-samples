transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  // Step 1. Apply im2col patterns
  // ==============================================================
  %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %im2col, %expand = transform.iree.convert_conv2d_to_img2col_and_adjust_workgroup_count_region %conv : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %matmul = transform.get_producer_of_operand %expand[0]
    : (!pdl.operation) -> (!pdl.operation)
  transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Step 2. First level of tiling + fusion parallelizes to blocks.
  // ==============================================================
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall, %tiled_matmul =
    transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [1, 16, 64, 0]
    ( mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>] )
  transform.structured.fuse_into_containing_op %fill into %forall
  %tiled_im2col = transform.structured.fuse_into_containing_op %im2col into %forall
  transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Step 3. Tile + Fuse
  // ==============================================================
  %matmul_loopk, %loop = transform.structured.tile_to_scf_for %tiled_matmul [0, 0, 0, 16]
  %im2col_loopk = transform.structured.fuse_into_containing_op %tiled_im2col into %loop
  transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Step 4. Promote to shared mem
  // ==============================================================
  %promoted_matmul_l2, %alloc_1 = transform.iree.promote_operands %matmul_loopk [0]
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Step 5. Map Im2col to threads (SIMT)
  // ==============================================================
  transform.structured.tile_to_forall_op %im2col_loopk num_threads [0, 0, 64]
    ( mapping = [#gpu.thread<x>] )
  transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Contraction part mapped to threads with a **SIMD** programming model.
  // =============================================================================
  %forall_l3, %matmul_padded_l3 = 
    transform.structured.tile_to_forall_op %promoted_matmul_l2 num_threads [0, 0, 2]
      ( mapping = [#gpu.warp<x>])

  // Step 6. Vectorize
  // ==============================================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
  %func_v_3 = transform.structured.vectorize %func_v_2 { vectorize_nd_extract }
  transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_wmma }
  //%contractions = transform.structured.match ops{["vector.contract"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  //%parents = transform.loop.get_parent_for %contractions { check_interface = false, loop_op = "scf.forall" } : (!pdl.operation) -> !pdl.operation
  //transform.print %parents : !pdl.operation
  //transform.print %contractions : !pdl.operation
  //transform.iree.apply_patterns %parents { unroll_vectors_gpu_wmma }
  //transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Step 7. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  %func_v_4 = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %func_4 = transform.iree.apply_patterns %func_v_4 { fold_reassociative_reshapes }
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %func_5 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!pdl.operation) -> !pdl.operation
  %func_6 = transform.iree.apply_patterns %func_5 { erase_unnecessary_tensor_operands }
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op_2
  transform.iree.apply_patterns %variant_op_3 {canonicalization, cse}

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %func_8 = transform.iree.forall_to_workgroup %func_7
  %func_9 = transform.iree.map_nested_forall_to_gpu_threads %func_8
      { workgroup_size = [64, 1, 1] }
  transform.iree.apply_patterns %variant_op_3 {canonicalization, cse}
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %hoisted_func = transform.iree.hoist_static_alloc %memref_func
    : (!pdl.operation) -> !pdl.operation
  %func_10 = transform.iree.gpu_distribute_shared_memory_copy %hoisted_func
    : (!pdl.operation) -> !pdl.operation
  %func_11 = transform.iree.erase_hal_descriptor_type_from_memref %func_10
  transform.iree.apply_patterns %variant_op_3 {canonicalization, cse, licm, tiling_canonicalization}
  %func_12 = transform.iree.vector.vector_to_mma_conversion %func_11 { use_wmma }
  transform.iree.apply_patterns %variant_op_3 {canonicalization, cse, licm, tiling_canonicalization}
}
