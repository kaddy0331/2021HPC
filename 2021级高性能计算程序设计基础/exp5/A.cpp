#include <iostream>
#include <cstdlib>
#include <ctime>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplication(int *a, int *b, int *c, int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Thread (%d, %d) - Row: %d, Col: %d\n", threadIdx.x, threadIdx.y, row, col);
	if (row < M && col < K) {
		int sum = 0;
		for (int i = 0; i < N; ++i) {
			sum += a[row * N + i] * b[i * K + col];
		}
		c[row * K + col] = sum;
	}
}

// Print the matrix 
void printMatrix(int *matrix, int rows, int cols) {
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			std::cout << matrix[i * cols + j] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char const *argv[]) {
	
	int block_size = atoi(argv[1]);
	
	// Set matrix dimensions
	int M = atoi(argv[2]);
	int N = atoi(argv[3]);
	int K = atoi(argv[4]);
	
	// Allocate memory for matrices on host
	int *h_A = new int[M * N];
	int *h_B = new int[N * K];
	int *h_C = new int[M * K];
	
	int maxBlockSizeX, maxBlockSizeY;
	cudaDeviceGetAttribute(&maxBlockSizeX, cudaDevAttrMaxBlockDimX, 0);
	cudaDeviceGetAttribute(&maxBlockSizeY, cudaDevAttrMaxBlockDimY, 0);
	
	// Check if the selected block size is within the supported range
	if (block_size > maxBlockSizeX || block_size > maxBlockSizeY) {
		fprintf(stderr, "Error: Invalid block size\n");
		exit(EXIT_FAILURE);
	}
	
	// Initialize matrices with random values
	srand(static_cast<unsigned>(2333));
	for (int i = 0; i < M * N; ++i) {
		h_A[i] = rand() % 10;
	}
	
	for (int i = 0; i < N * K; ++i) {
		h_B[i] = rand() % 10;
	}
	
	// Allocate memory for matrices on device
	int *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, M * N * sizeof(int));
	cudaMalloc((void**)&d_B, N * K * sizeof(int));
	cudaMalloc((void**)&d_C, M * K * sizeof(int));
	
	// Copy matrices from host to device
	cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N * K * sizeof(int), cudaMemcpyHostToDevice);
	
	// Set block size and grid size
	dim3 blockSize(block_size, block_size);
	dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
	
	// Launch kernel and measure time
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
	
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
	}
	
	cudaDeviceSynchronize();
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	// Copy result matrix from device to host
	cudaMemcpy(h_C, d_C, M * K * sizeof(int), cudaMemcpyDeviceToHost);
	
	// Print matrices and execution time
	std::cout << "Matrix A:" << std::endl;
	printMatrix(h_A, M, N);
	
	std::cout << "Matrix B:" << std::endl;
	printMatrix(h_B, N, K);
	
	std::cout << "Matrix C:" << std::endl;
	printMatrix(h_C, M, K);
	
	// 打印线程块和网格的大小
	std::cout << "Block size: (" << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << ")" << std::endl;
	std::cout << "Grid size: (" << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << ")" << std::endl;
	
	printf("Elapsed Time: %f ms\n", elapsedTime);
	
	// Free allocated memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	return 0;
}

