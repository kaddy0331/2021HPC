#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Print the matrix 
void printMatrix(float *matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			std::cout << matrix[i * cols + j] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char const *argv[]) {
	// 初始化CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	// 定义矩阵规模
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	
	// 在主机上分配内存
	float *h_A = new float[m * n];
	float *h_B = new float[n * k];
	float *h_C = new float[m * k];
	
	// 随机初始化矩阵
	srand(static_cast<unsigned>(2333));
	for (int i = 0; i < m * n; ++i) {
		h_A[i] = rand() % 10;
	}
	
	for (int i = 0; i < n * k; ++i) {
		h_B[i] = rand() % 10;
	}
	
	// 在GPU上分配内存
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, m * n * sizeof(float));
	cudaMalloc((void**)&d_B, n * k * sizeof(float));
	cudaMalloc((void**)&d_C, m * k * sizeof(float));
	
	// 将数据从主机内存传输到GPU内存
	cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice);
	
	// 调用CUBLAS函数执行矩阵相乘
	float alpha = 1.0f;
	float beta = 0.0f;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &alpha, d_B, m, d_A, k, &beta, d_C, m);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	// 将结果从GPU内存传输回主机内存
	cudaMemcpy(h_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);
	
	// 打印矩阵和执行时间
	//std::cout << "Matrix A:" << std::endl;
	//printMatrix(h_A, m, n);
	
	//std::cout << "Matrix B:" << std::endl;
	//printMatrix(h_B, n, k);
	
	//std::cout << "Matrix C:" << std::endl;
	//printMatrix(h_C, m, k);
	
	printf("Elapsed Time: %f ms\n", elapsedTime);
	
	// 释放内存
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	// 销毁CUBLAS句柄
	cublasDestroy(handle);
	
	return 0;
}


