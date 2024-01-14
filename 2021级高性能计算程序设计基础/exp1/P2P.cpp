#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <mpi.h>

using namespace std;

const int N = 128; // 矩阵大小
const int MASTER = 0;

void initializeMatrix(double matrix[], int size, unsigned int seed) {
	mt19937 rng(seed); // 使用Mersenne Twister 19937伪随机数生成器
	uniform_real_distribution<double> dist(0.0, 1.0);
	
	for (int i = 0; i < size; i++) {
		matrix[i] = dist(rng); // 生成随机数填充矩阵
	}
}

void matrixMultiply(double A[], double B[], double C[], int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			C[i*size + j] = 0.0;
			for (int k = 0; k < size; k++) {
				C[i*size + j] += A[i*size + k] * B[k*size + j];
			}
		}
	}
}

void print_matrix(double mat[], int N, int outputSize){
	for (int i = 0; i < outputSize; i++) {
		for (int j = 0; j < outputSize; j++) {
			cout << mat[i * N + j] << " ";
		}
		cout << " ... ";
		for (int j = N - outputSize; j < N; j++) {
			cout << mat[i * N + j] << " ";
		}
		cout << endl;
	}
	cout << " ... " << endl;
	for (int i = N - outputSize; i < N; i++) {
		for (int j = 0; j < outputSize; j++) {
			cout << mat[i * N + j] << " ";
		}
		cout << " ... ";
		for (int j = N - outputSize; j < N; j++) {
			cout << mat[i * N + j] << " ";
		}
		cout << endl;
	} 
}   

int main(int argc, char *argv[]) {
	int rank, size;
	double *A, *B, *C, *A_temp;
	double startTime, endTime;
	unsigned int seed1 = 12345; 
	unsigned int seed2 = 54321; 
	
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int pr=0;
	// 矩阵的局部大小
	int localSize;
	if(size != 1) {
		localSize = N / (size - 1);
	}
	
	if (rank == MASTER) {
		A = new double[N * N];
		B = new double[N * N];
		C = new double[N * N];
		
		// 使用相同的种子初始化矩阵A和B，确保可重复性
		initializeMatrix(A, N * N, seed1);
		initializeMatrix(B, N * N, seed2);
	} else {
		A = new double[N * localSize];
		B = new double[N * N];
		C = new double[N * localSize];
	}
	
	startTime = MPI_Wtime();
	
	// 广播矩阵B给所有进程
	MPI_Bcast(B, N * N, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	
	if (size > 1) {
		if (rank == MASTER) {
			// 将A的各行分发给其他进程
			for (int dest = 1; dest < size; dest++) {
				int startIndex = (dest-1) * localSize;

				MPI_Send(&A[startIndex], localSize * N, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
				
			}
		} else {
			// 接收分配给本地进程的A的子矩阵
			MPI_Recv(A, localSize * N, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			// 矩阵乘法
			matrixMultiply(A, B, C, localSize);
		}
		
		
	} else {
		// 串行矩阵乘法
		matrixMultiply(A, B, C, N);
	}
	
	if (rank == MASTER) {
		// 收集各进程的结果
		for (int source = 1; source < size; source++) {
			int startIndex = (source-1) * localSize;
			MPI_Recv(&C[startIndex * N], localSize * N, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		endTime = MPI_Wtime();
		cout << "矩阵乘法执行时间: " << (endTime - startTime)*1000 << " ms" << endl;
		
		if (pr == 1){
			// 输出矩阵
			int outputSize = 2; // 输出的子矩阵大小
			printf("矩阵A：\n");
			print_matrix(A, N, outputSize);
			printf("矩阵B：\n");
			print_matrix(B, N, outputSize);
			printf("矩阵C：\n");
			print_matrix(C, N, outputSize);
		}
		
		// 清理内存
		delete[] A;
		delete[] B;
		delete[] C;
	} else {
		// 发送本地计算的结果给主进程
		MPI_Send(C, localSize * N, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
		
		// 清理内存
		delete[] A;
		delete[] B;
		delete[] C;
	}
	
	MPI_Finalize();
	
	return 0;
}

