#include <iostream>
#include <cstdlib>
#include <ctime>

// Serial matrix multiplication
void serialMatrixMultiplication(int *a, int *b, int *c, int M, int N, int K) {
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < K; ++j) {
			int sum = 0;
			for (int k = 0; k < N; ++k) {
				sum += a[i * N + k] * b[k * K + j];
			}
			c[i * K + j] = sum;
		}
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
	// Set matrix dimensions
	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);
	
	// Allocate memory for matrices on host
	int *h_A = new int[M * N];
	int *h_B = new int[N * K];
	int *h_C = new int[M * K];
	
	// Initialize matrices with random values
	srand(static_cast<unsigned>(2333));
	for (int i = 0; i < M * N; ++i) {
		h_A[i] = rand() % 10;
	}
	
	for (int i = 0; i < N * K; ++i) {
		h_B[i] = rand() % 10;
	}

	// Perform serial matrix multiplication
	serialMatrixMultiplication(h_A, h_B, h_C, M, N, K);
	
	// Print matrices and execution time
	std::cout << "Matrix A:" << std::endl;
	printMatrix(h_A, M, N);
	
	std::cout << "Matrix B:" << std::endl;
	printMatrix(h_B, N, K);
	
	std::cout << "Matrix C:" << std::endl;
	printMatrix(h_C, M, K);
	
	// Free allocated memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	
	return 0;
}

