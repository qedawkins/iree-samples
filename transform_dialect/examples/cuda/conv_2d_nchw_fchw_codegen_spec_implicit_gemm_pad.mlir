// Instructions; TL;DR
// ===================
//
// ```
//    export IREE_BUILD_DIR=/home/quinn/nod/iree-build; \
//    export IREE_SAMPLES_DIR=/home/quinn/nod/iree-samples; \
//    cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/conv_2d_nchw_fchw.mlir |\
//    sed "s/\${N}/2/g" | sed "s/\${C}/16/g" | sed "s/\${F}/32/g" | \
//    sed "s/\${H}/130/g" | sed "s/\${W}/130/g" | \
//    sed "s/\${KH}/3/g" | sed "s/\${KW}/3/g" | \
//    sed "s/\${OH}/128/g" | sed "s/\${OW}/128/g" | \
//    ${IREE_BUILD_DIR}/tools/iree-opt \
//      --iree-hal-target-backends=cuda \
//      --iree-abi-transformation-pipeline \
//      --iree-flow-transformation-pipeline \
//      --iree-stream-transformation-pipeline \
//      --iree-hal-configuration-pipeline | \
//    ${IREE_BUILD_DIR}/build/tools/iree-opt \
//       --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//       --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/conv_2d_nchw_fchw_codegen_spec_implicit_gemm.mlir \
//       --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  // Step 2. Apply im2col patterns
  // ==============================================================
  %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  //%im2col, %expand = transform.structured.convert_conv2d_to_img2col %conv : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %im2col, %expand = transform.iree.convert_conv2d_to_img2col_and_adjust_workgroup_count_region %conv : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %matmul = transform.get_producer_of_operand %expand[0]
    : (!pdl.operation) -> (!pdl.operation)
  transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Step 1. First level of tiling + fusion parallelizes to blocks.
  // ==============================================================
  //%conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall, %tiled_matmul =
    transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [1, 32, 128, 0]
    ( mapping = [#gpu.block<x>, #gpu.block<y>, #gpu.block<z>] )
  //%forall, %tiled_matmul =
  //  transform.structured.tile_to_forall_op %matmul tile_sizes [1, 32, 32, 0]
  //  ( mapping = [#gpu.block<z>, #gpu.block<x>, #gpu.block<y>] )
  transform.structured.fuse_into_containing_op %fill into %forall
  %tiled_im2col = transform.structured.fuse_into_containing_op %im2col into %forall
  transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Step 3. Tile + Fuse
  // ==============================================================
  //%in_pad = transform.get_producer_of_operand %matmul_padded_l1[0]
  //  : (!pdl.operation) -> !pdl.operation
  //%weight_pad = transform.get_producer_of_operand %matmul_padded_l1[1]
  //  : (!pdl.operation) -> !pdl.operation
  %matmul_loopk, %loop:3 = transform.structured.tile_to_scf_for %tiled_matmul [0, 16, 16, 16]
  //%in_pad_loopk = transform.structured.fuse_into_containing_op %in_pad into %loop
  //%weight_pad_loopk = transform.structured.fuse_into_containing_op %weight_pad into %loop
  %im2col_loopk = transform.structured.fuse_into_containing_op %tiled_im2col into %loop
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm, tiling_canonicalization}

  // Step 3.5. Pad the matmul and force packing to create the buffer in shared memory
  // Note: hoisting here may be dangerous memory-consumption-wise and we may be
  // better off with pipelining only.
  // ==============================================================================
  %matmul_padded_l2 = transform.structured.pad %matmul_loopk {
    padding_values = [0.0 : f16, 0.0 : f16, 0.0 : f16, 0.0 : f16], 
    padding_dimensions = [0, 1, 2, 3], 
    pack_paddings=[1, 1, 1, 1]
  }
  // Post-padding canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm, tiling_canonicalization}
  // Hoist the padding of the result tensor.
  //%res_pad = transform.get_producer_of_operand %matmul_padded_l2[2]
  //  : (!pdl.operation) -> !transform.op<"tensor.pad">
  //transform.structured.hoist_pad %res_pad by 1 loops
  //   : (!transform.op<"tensor.pad">) -> !pdl.operation

  //// Step 3. Rewrite tensor.pad in DPS.
  //// ==================================
  //%pad = transform.structured.match ops{["tensor.pad"]} in %variant_op 
  //  : (!pdl.operation) -> !pdl.operation
  //%padded = transform.structured.rewrite_in_destination_passing_style %pad 
  //  : (!pdl.operation) -> !pdl.operation

  //// Step 4. Promote to shared mem
  //// ==============================================================
  ////%promoted_img2col_l2, %alloc_1 = transform.iree.promote_operands %im2col_loopk [0]
  ////  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  ////%promoted_matmul_l2, %alloc_1, %alloc_2 = transform.iree.promote_operands %matmul_loopk [0, 2]
  ////  : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  //%promoted_matmul_l2, %alloc_1 = transform.iree.promote_operands %matmul_loopk [0]
  //  : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  //transform.iree.apply_patterns %variant_op {canonicalization, cse}

  // Step 5. Map to threads
  // ==============================================================
  //%forall_l3, %matmul_l3 =
  //transform.structured.tile_to_forall_op %padded_matmul_l3 num_threads [0, 2, 8]
  //  ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  //transform.structured.tile_to_forall_op %im2col_loopk num_threads [0, 2, 8]
  //  ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  ////%im2col_l3 = transform.structured.fuse_into_containing_op %promoted_img2col_l2 into %forall_l3
  //transform.iree.apply_patterns %variant_op {canonicalization, cse}
  //%fill_l2 = transform.structured.match ops{["linalg.fill"]} in %variant_op
  //  : (!pdl.operation) -> !pdl.operation
  //transform.structured.tile_to_forall_op %fill_l2 num_threads [16, 2]
  //    ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  //%copy = transform.structured.match ops{["linalg.copy"]} in %variant_op
  //  : (!pdl.operation) -> !pdl.operation
  //transform.structured.tile_to_forall_op %copy num_threads [0, 2, 8]
  //    ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  //transform.iree.apply_patterns %variant_op {canonicalization, cse}

  //// Step 6. Vectorize
  //// ==============================================================
  //%func_v = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  //%func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
  //%func_v_3 = transform.structured.vectorize %func_v_2 { vectorize_nd_extract, vectorize_padding }
  //%func_v_4 = transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_wmma }
  //transform.iree.apply_patterns %variant_op {canonicalization, cse}
  //%func_v_5 = transform.structured.hoist_redundant_tensor_subsets %func_v_4
  //  : (!pdl.operation) -> !pdl.operation

  // Step 7. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  //%func_v_5 = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  //%func_4 = transform.iree.apply_patterns %func_v_5 { fold_reassociative_reshapes }
  //%variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  //%func_5 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!pdl.operation) -> !pdl.operation
  //%func_6 = transform.iree.apply_patterns %func_5 { erase_unnecessary_tensor_operands }
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op
  ////transform.iree.apply_patterns %variant_op_3 {canonicalization, cse, licm, tiling_canonicalization}

  //// Step 6. Post-bufferization mapping to blocks and threads.
  //// ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  //transform.print %func_7 {name = "func"} : !pdl.operation
  //transform.print %variant_op_3 {name = "variant_op"} : !pdl.operation
  //%func_11 = transform.iree.erase_hal_descriptor_type_from_memref %func_7
  %func_8 = transform.iree.forall_to_workgroup %func_7
  //%func_9 = transform.iree.map_nested_forall_to_gpu_threads %func_8
  //    { workgroup_size = [8, 2, 1] }
  //transform.iree.apply_patterns %variant_op_3 {canonicalization, cse}
  //%memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  //%hoisted_func = transform.iree.hoist_static_alloc %memref_func
  //  : (!pdl.operation) -> !pdl.operation
  //%func_10 = transform.iree.gpu_distribute_shared_memory_copy %hoisted_func
  //  : (!pdl.operation) -> !pdl.operation
  //%func_11 = transform.iree.erase_hal_descriptor_type_from_memref %func_9
  //transform.iree.apply_patterns %variant_op_3 {canonicalization, cse, licm, tiling_canonicalization}
  //%func_12 = transform.iree.vector.vector_to_mma_conversion %func_11 { use_wmma }
  //transform.iree.apply_patterns %variant_op_3 {canonicalization, cse, licm, tiling_canonicalization}
}
