# amx-gemm-handwritten

手写Intel AMX矩阵乘法算子。

- 1-amx-gemm-l0-tiling：Tile Register级矩阵分块，TM = TN = 32
- 2-amx-gemm-l2-tiling：L2 Cache级矩阵分块，TM = TN = TK = 512
- todo：prefetch
- 4-amx-gemm-l2-tiling-multi-thread：多线程版本，TM = TN = TK = 512