#include <iostream>
#include <random>
#include <ctime>
#include <mpi.h>

struct MatrixBlock {
	int rows;
	int cols;
	double data[1]; // 实际上是一个长度为1的动态数组，用于存储矩阵数据
};

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

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	const int matrixSize = 512; // 矩阵大小
	const int blockSize = matrixSize / size; // 每个进程的块大小
	
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
	
	// 创建自定义数据类型
	MPI_Datatype matrixBlockType;
	int blockLengths[2] = {1, blockSize * matrixSize}; // 两个字段，第一个是 rows，第二个是 data
	MPI_Aint displacements[2] = {offsetof(MatrixBlock, rows), offsetof(MatrixBlock, data)};
	MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
	MPI_Type_create_struct(2, blockLengths, displacements, types, &matrixBlockType);
	MPI_Type_commit(&matrixBlockType);
	
	// 进行局部矩阵乘法计算
	MatrixBlock localBlock;
	localBlock.rows = blockSize;
	localBlock.cols = matrixSize;
	for (int i = 0; i < blockSize; i++) {
		for (int j = 0; j < matrixSize; j++) {
			localBlock.data[i * matrixSize + j] = localA[i * matrixSize + j];
		}
	}
	
	MatrixBlock resultBlock;
	resultBlock.rows = blockSize;
	resultBlock.cols = matrixSize;
	
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&localBlock, 1, matrixBlockType, 0, MPI_COMM_WORLD);
	
	// 进行局部矩阵乘法计算
	matrixMultiply(localBlock.data, B, resultBlock.data, matrixSize, blockSize);
	
	// 收集结果到根进程
	MPI_Gather(&resultBlock, 1, matrixBlockType, C, 1, matrixBlockType, 0, MPI_COMM_WORLD);
	
	double endTime = MPI_Wtime(); // 记录结束时间
	
	// 打印结果
	if (rank == 0) {
		std::cout << "Execution time: " << (endTime - startTime) * 1000 << " ms" << std::endl;
	}
	
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] localA;
	delete[] localC;
	
	MPI_Type_free(&matrixBlockType);
	MPI_Finalize();
	
	return 0;
}

