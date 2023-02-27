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


// CHECK: %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xindex>
// CHECK: %cst_0 = arith.constant dense<3> : vector<16x16xindex>
// CHECK: %cst_1 = arith.constant dense<9> : vector<16x16xindex>
// CHECK: %cst_2 = arith.constant dense<64> : vector<16xindex>
// CHECK: %cst_3 = arith.constant dense<true> : vector<16x16xi1>
// CHECK: %cst_4 = arith.constant dense<0.000000e+00> : vector<16x16xf32>
// CHECK: %cst_5 = arith.constant dense<10> : vector<16x16xindex>
// CHECK: %cst_6 = arith.constant dense<66> : vector<16x16xindex>
// CHECK: %cst_7 = arith.constant 0.000000e+00 : f32
// CHECK: %c144 = arith.constant 144 : index
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<2x16x130x130xf32>
// CHECK: memref.assume_alignment %0, 64 : memref<2x16x130x130xf32>
// CHECK: %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<2x32x128x128xf32>
// CHECK: memref.assume_alignment %1, 64 : memref<2x32x128x128xf32>
// CHECK: %workgroup_id_z = hal.interface.workgroup.id[2] : index
// CHECK: %workgroup_id_x = hal.interface.workgroup.id[0] : index
// CHECK: %workgroup_id_y = hal.interface.workgroup.id[1] : index
// CHECK: %2 = affine.apply #map2()[%workgroup_id_x]
// CHECK: %3 = affine.apply #map3()[%workgroup_id_y]
// CHECK: %subview = memref.subview %0[%workgroup_id_z, 0, %2, %3] [1, 16, 10, 66] [1, 1, 1, 1] : memref<2x16x130x130xf32> to memref<1x16x10x66xf32, strided<[270400, 16900, 130, 1], offset: ?>>
// CHECK: %subview_8 = memref.subview %1[%workgroup_id_z, 0, %2, %3] [1, 32, 8, 64] [1, 1, 1, 1] : memref<2x32x128x128xf32> to memref<1x32x8x64xf32, strided<[524288, 16384, 128, 1], offset: ?>>
// CHECK: %4 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x144xf32>
// CHECK: memref.assume_alignment %4, 64 : memref<32x144xf32>
// CHECK: %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x32x8x64xf32, #gpu.address_space<workgroup>>
// CHECK: linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%subview_8 : memref<1x32x8x64xf32, strided<[524288, 16384, 128, 1], offset: ?>>) outs(%alloc : memref<1x32x8x64xf32, #gpu.address_space<workgroup>>) {
// CHECK: ^bb0(%in: f32, %out: f32):
// CHECK:   linalg.yield %in : f32
// CHECK: }
// CHECK: %5 = bufferization.to_tensor %alloc : memref<1x32x8x64xf32, #gpu.address_space<workgroup>>
// CHECK: %6 = bufferization.to_memref %5 : memref<1x32x8x64xf32>
// CHECK: %collapse_shape = memref.collapse_shape %6 [[0], [1], [2, 3]] : memref<1x32x8x64xf32> into memref<1x32x512xf32>
// CHECK: %7 = gpu.thread_id  x
// CHECK: %8 = gpu.thread_id  y
// CHECK: %9 = affine.apply #map5()[%8]
// CHECK: %10 = affine.apply #map6()[%7]
// CHECK: %11 = vector.broadcast %cst : vector<16xindex> to vector<16x16xindex>
// CHECK: %12 = vector.transpose %11, [1, 0] : vector<16x16xindex> to vector<16x16xindex>
// CHECK: %13 = arith.muli %7, %c16 : index
// CHECK: %14 = vector.broadcast %13 : index to vector<16xindex>
// CHECK: %15 = arith.addi %14, %cst : vector<16xindex>
// CHECK: %16 = arith.remui %15, %cst_2 : vector<16xindex>
// CHECK: %17 = arith.divui %15, %cst_2 : vector<16xindex>
// CHECK: %18 = vector.broadcast %17 : vector<16xindex> to vector<16x16xindex>
// CHECK: %19 = vector.broadcast %16 : vector<16xindex> to vector<16x16xindex>
// CHECK: %subview_9 = memref.subview %collapse_shape[0, %9, %10] [1, 4, 16] [1, 1, 1] : memref<1x32x512xf32> to memref<1x4x16xf32, strided<[16384, 512, 1], offset: ?>>
// CHECK: scf.for %arg0 = %c0 to %c144 step %c16 {
// CHECK:   %20 = vector.broadcast %arg0 : index to vector<16x16xindex>
// CHECK:   %21 = arith.addi %12, %20 : vector<16x16xindex>
// CHECK:   %22 = arith.remui %21, %cst_0 : vector<16x16xindex>
// CHECK:   %23 = arith.remui %21, %cst_1 : vector<16x16xindex>
// CHECK:   %24 = arith.divui %23, %cst_0 : vector<16x16xindex>
// CHECK:   %25 = arith.divui %21, %cst_1 : vector<16x16xindex>
// CHECK:   %26 = arith.addi %18, %24 : vector<16x16xindex>
// CHECK:   %27 = arith.addi %19, %22 : vector<16x16xindex>
// CHECK:   %28 = arith.muli %25, %cst_5 : vector<16x16xindex>
// CHECK:   %29 = arith.addi %26, %28 : vector<16x16xindex>
// CHECK:   %30 = arith.muli %29, %cst_6 : vector<16x16xindex>
// CHECK:   %31 = arith.addi %27, %30 : vector<16x16xindex>
// CHECK:   %32 = vector.gather %subview[%c0, %c0, %c0, %c0] [%31], %cst_3, %cst_4 : memref<1x16x10x66xf32, strided<[270400, 16900, 130, 1], offset: ?>>, vector<16x16xindex>, vector<16x16xi1>, vector<16x16xf32> into vector<16x16xf32>
// CHECK:   %33 = vector.transfer_read %4[%9, %arg0], %cst_7 {in_bounds = [true, true]} : memref<32x144xf32>, vector<4x16xf32>
// CHECK:   %34 = vector.transfer_read %collapse_shape[%c0, %9, %10], %cst_7 {in_bounds = [true, true]} : memref<1x32x512xf32>, vector<4x16xf32>
// CHECK:   %35 = vector.contract {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %33, %32, %34 : vector<4x16xf32>, vector<16x16xf32> into vector<4x16xf32>
// CHECK:   vector.transfer_write %35, %subview_9[%c0, %c0, %c0] {in_bounds = [true, true]} : vector<4x16xf32>, memref<1x4x16xf32, strided<[16384, 512, 1], offset: ?>>
// CHECK:   gpu.barrier
// CHECK: }
// CHECK: linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : memref<1x32x8x64xf32>) outs(%subview_8 : memref<1x32x8x64xf32, strided<[524288, 16384, 128, 1], offset: ?>>) {
// CHECK: ^bb0(%in: f32, %out: f32):
// CHECK:   linalg.yield %in : f32
// CHECK: }
// CHECK: return

transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  // Step 1. First level of tiling + fusion parallelizes to blocks.
  // ==============================================================
  %conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %forall, %tiled_conv =
    transform.iree.tile_to_forall_and_workgroup_count_region %conv tile_sizes [1, 0, 8, 64]
    ( mapping = [#gpu.block<z>, #gpu.block<x>, #gpu.block<y>] )
  transform.structured.fuse_into_containing_op %fill into %forall

  // Step 2. Apply im2col patterns
  // ==============================================================
  %im2col, %expand = transform.structured.convert_conv2d_to_img2col %tiled_conv : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  %matmul = transform.get_producer_of_operand %expand[0]
    : (!pdl.operation) -> (!pdl.operation)

  // Step 3. Tile + Fuse
  // ==============================================================
  %tiled_matmul, %loop = transform.structured.tile_to_scf_for %matmul [0, 0, 0, 16]
  %tiled_im2col = transform.structured.fuse_into_containing_op %im2col into %loop

  //// Step 4. Promote to shared mem
  //// ==============================================================
  //%promoted_matmul_l2, %alloc_1 , %alloc_2 = transform.iree.promote_operands %tiled_matmul [0, 1] 
  //  : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)

  // Step 5. Map to threads
  // ==============================================================
  %forall_l3, %matmul_l3 =
  transform.structured.tile_to_forall_op %tiled_matmul num_threads [0, 8, 32]
    ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  %im2col_l3 = transform.structured.fuse_into_containing_op %tiled_im2col into %forall_l3

  // Step 6. Vectorize
  // ==============================================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
  %func_v_3 = transform.structured.vectorize %func_v_2 { vectorize_nd_extract }
  //%func_v_4 = transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_wmma }

  // Step 7. Bufferize and drop HAL decriptor from memref ops.
  // ===========================================================================
  %func_4 = transform.iree.apply_patterns %func_v_3 { fold_reassociative_reshapes }
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %func_5 = transform.structured.match ops{["func.func"]} in %variant_op_2 : (!pdl.operation) -> !pdl.operation
  %func_6 = transform.iree.apply_patterns %func_5 { erase_unnecessary_tensor_operands }
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op_2
  %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %memref_func

  // Step 6. Post-bufferization mapping to blocks and threads.
  // ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %func_8 = transform.iree.forall_to_workgroup %func_7
  %func_9 = transform.iree.map_nested_forall_to_gpu_threads %func_8
      { workgroup_size = [32, 8, 1] }
}
