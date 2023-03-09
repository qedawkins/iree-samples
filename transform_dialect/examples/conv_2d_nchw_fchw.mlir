!input_tensor_t = tensor<${N}x${C}x${H}x${W}x${DT}>
!weight_tensor_t = tensor<${F}x${C}x${KH}x${KW}x${DT}>
!output_tensor_t = tensor<${N}x${F}x${OH}x${OW}x${DT}>
func.func @conv_2d_nchw_fchw(%in: !input_tensor_t, %wei: !weight_tensor_t,
                             %out: !output_tensor_t) -> !output_tensor_t {
  %res = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}
