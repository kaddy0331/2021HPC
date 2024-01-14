#include <iostream>
#include <random>
#include <ctime>
#include <mpi.h>

void initializeMatrix(double matrix[], int size, unsigned int seed) {
	std::mt19937 rng(seed); // 使用Mersenne Twister 19937伪随机数生成器
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	
	for (int i = 0; i < size; i++) {
		matrix[i] = dist(rng); // 生成随机数填充矩阵
	}
}

void matrixMultiply(double A[], double B[], double C[], int size, int blockSize) {
	for (int i = 0; i < blockSize; i++) {
		for (int j = 0; j < size; j++) {
			C[i*size + j] = 0.0;
			for (int k = 0; k < size; k++) {
				C[i*size + j] += A[i*size + k] * B[k*size + j];
			}
		}
	}
}

void printMatrix(double matrix[], int size, int printRows, int printCols) {
	for (int i = 0; i < printRows; i++) {
		for (int j = 0; j < printCols; j++) {
			std::cout << matrix[i * size + j] << " ";
		}
		if (printCols < size) {
			std::cout << " ..."; // 中间部分省略号
		}
		std::cout << std::endl;
	}
	if (printRows < size) {
		std::cout << "..." << std::endl; // 中间部分省略号
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	const int matrixSize = 128; // 矩阵大小
	const int blockSize = matrixSize / size; // 每个进程的块大小
	const int printRows = 2; // 打印前几行
	const int printCols = 2; // 打印前几列
	int pr=0;
	
	double* A = new double[matrixSize * matrixSize];
	double* B = new double[matrixSize * matrixSize];
	double* C = new double[matrixSize * matrixSize];
	double* localA = new double[blockSize * matrixSize];
	double* localC = new double[blockSize * matrixSize];
	
	// 初始化矩阵 A 和 B
	if (rank == 0) {
		initializeMatrix(A, matrixSize * matrixSize, 42);
		initializeMatrix(B, matrixSize * matrixSize, 24);
	}
	
	double startTime = MPI_Wtime(); // 记录开始时间
	
	// 分发数据块到各进程
	MPI_Scatter(A, blockSize * matrixSize, MPI_DOUBLE, localA, blockSize * matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, matrixSize * matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// 进行局部矩阵乘法计算
	matrixMultiply(localA, B, localC, matrixSize, blockSize);
	
	// 收集结果到根进程
	MPI_Gather(localC, blockSize * matrixSize, MPI_DOUBLE, C, blockSize * matrixSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	double endTime = MPI_Wtime(); // 记录结束时间
	
	// 打印结果
	if (rank == 0) {
		
		if(pr == 1){
			std::cout << "Matrix A:" << std::endl;
			printMatrix(A, matrixSize, printRows, printCols);
			
			std::cout << "Matrix B:" << std::endl;
			printMatrix(B, matrixSize, printRows, printCols);
			
			std::cout << "Matrix C:" << std::endl;
			printMatrix(C, matrixSize, printRows, printCols);
		}
		
		std::cout << "Execution time: " << (endTime - startTime)*1000 << " ms" << std::endl;
		
	}
	
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] localA;
	delete[] localC;
	
	MPI_Finalize();
	
	return 0;
}

