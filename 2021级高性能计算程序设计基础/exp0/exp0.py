import numpy as np
import time

# 随机生成M*N和N*K的矩阵 A 和 B，元素为单精度浮点数
M = np.random.randint(512, 2049)
N = np.random.randint(512, 2049)
K = np.random.randint(512, 2049)

A = np.random.rand(M, N).astype(np.float32)
B = np.random.rand(N, K).astype(np.float32)

# 初始化结果矩阵 C
C = np.zeros((M, K), dtype=np.float32)

# 执行矩阵乘法并测量时间
start_time = time.time()

for i in range(M):
    for j in range(K):
        for k in range(N):
            C[i, j] += A[i, k] * B[k, j]

end_time = time.time()

# 打印矩阵 A、B 和 C 以及执行时间
print('N=',N,' M=',M,' K=',K)
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nMatrix C (Result of A * B):")
print(C)
print("\nMatrix multiplication time:", end_time - start_time, "seconds")
