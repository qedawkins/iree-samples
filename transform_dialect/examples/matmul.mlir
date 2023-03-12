!A_size = tensor<${M}x${K}x${DT}>
!B_size = tensor<${K}x${N}x${DT}>
!C_size = tensor<${M}x${N}x${DT}>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}
