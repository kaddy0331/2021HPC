#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// 定义单精度浮点数矩阵结构
typedef struct {
	int rows;
	int cols;
	float** data;
} Matrix;

// 创建一个随机矩阵
Matrix createRandomMatrix(int rows, int cols) {
	Matrix mat;
	mat.rows = rows;
	mat.cols = cols;
	mat.data = (float**)malloc(rows * sizeof(float*));
	
	#pragma omp parallel for
	for (int i = 0; i < rows; i++) {
		mat.data[i] = (float*)malloc(cols * sizeof(float));
		for (int j = 0; j < cols; j++) {
			mat.data[i][j] = (float)rand() / RAND_MAX; // 生成0到1之间的随机浮点数
		}
	}
	
	return mat;
}

// 矩阵乘法
Matrix matrixMultiply(const Matrix* A, const Matrix* B) {
	if (A->cols != B->rows) {
		fprintf(stderr, "Matrix dimensions are incompatible for multiplication\n");
		exit(1);
	}
	
	Matrix C;
	C.rows = A->rows;
	C.cols = B->cols;
	C.data = (float**)malloc(C.rows * sizeof(float*));
	
	#pragma omp parallel for
	for (int i = 0; i < C.rows; i++) {
		C.data[i] = (float*)calloc(C.cols, sizeof(float));
		for (int j = 0; j < C.cols; j++) {
			for (int k = 0; k < A->cols; k++) {
				C.data[i][j] += A->data[i][k] * B->data[k][j];
			}
		}
	}
	
	return C;
}
//static
Matrix matrixMultiply_sta(const Matrix* A, const Matrix* B) {
	if (A->cols != B->rows) {
		fprintf(stderr, "Matrix dimensions are incompatible for multiplication\n");
		exit(1);
	}
	
	Matrix C;
	C.rows = A->rows;
	C.cols = B->cols;
	C.data = (float**)malloc(C.rows * sizeof(float*));
	
	#pragma omp parallel for schedule(static, 1)
	for (int i = 0; i < C.rows; i++) {
		C.data[i] = (float*)calloc(C.cols, sizeof(float));
		for (int j = 0; j < C.cols; j++) {
			for (int k = 0; k < A->cols; k++) {
				C.data[i][j] += A->data[i][k] * B->data[k][j];
			}
		}
	}
	
	return C;
}

//dynamic
Matrix matrixMultiply_dyn(const Matrix* A, const Matrix* B) {
	if (A->cols != B->rows) {
		fprintf(stderr, "Matrix dimensions are incompatible for multiplication\n");
		exit(1);
	}
	
	Matrix C;
	C.rows = A->rows;
	C.cols = B->cols;
	C.data = (float**)malloc(C.rows * sizeof(float*));
	
	#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < C.rows; i++) {
		C.data[i] = (float*)calloc(C.cols, sizeof(float));
		for (int j = 0; j < C.cols; j++) {
			for (int k = 0; k < A->cols; k++) {
				C.data[i][j] += A->data[i][k] * B->data[k][j];
			}
		}
	}
	
	return C;
}

// 释放矩阵内存
void freeMatrix(Matrix* mat) {
	#pragma omp parallel for
	for (int i = 0; i < mat->rows; i++) {
		free(mat->data[i]);
	}
	free(mat->data);
}

// 打印矩阵的内容，只输出部分内容，用省略号代替其余部分
void printMatrix(const Matrix* mat, int maxRows, int maxCols) {
	int numRows = mat->rows;
	int numCols = mat->cols;
	
	for (int i = 0; i < numRows && i < maxRows; i++) {
		for (int j = 0; j < numCols && j < maxCols; j++) {
			printf("%f ", mat->data[i][j]);
		}
		
		if (numCols > maxCols) {
			printf("..."); // 用省略号代替超出部分的列
		}
		
		printf("\n");
	}
	
	if (numRows > maxRows) {
		printf("...\n"); // 用省略号代替超出部分的行
	}
}

int main(int argc, char * argv[]) {
	srand(time(NULL)); // 初始化随机数生成器
	
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    int choose = atoi(argv[4]);
	
	printf("N=%d, M=%d, K=%d\n", N, M, K);
	
	// 创建随机矩阵 A 和 B
	Matrix A = createRandomMatrix(M, N);
	Matrix B = createRandomMatrix(N, K);
	
	// 初始化结果矩阵 C
	Matrix C;
	C.rows = M;
	C.cols = K;
	C.data = (float**)malloc(M * sizeof(float*));
	
	#pragma omp parallel for
	for (int i = 0; i < M; i++) {
		C.data[i] = (float*)calloc(K, sizeof(float));
	}
	
	// 执行矩阵乘法并测量时间
	double start_time = omp_get_wtime();
	if (choose == 0) {
	    C = matrixMultiply(&A, &B);
	}else if (choose == 1) {
    	    C = matrixMultiply_sta(&A, &B);
	}else if (choose ==2) {
	    C = matrixMultiply_dyn(&A, &B);
	}
	double end_time = omp_get_wtime();
	
	// 打印执行时间（以秒为单位）
	double execution_time = end_time - start_time;
	printf("Matrix multiplication time: %f seconds\n", execution_time);
	
	// 打印矩阵 A、B 和 C 的内容
	printf("\nMatrix A:\n");
	printMatrix(&A, 2, 2);
	printf("\nMatrix B:\n");
	printMatrix(&B, 2, 2);
	printf("\nMatrix C (Result of A * B):\n");
	printMatrix(&C, 2, 2);
	
	// 释放矩阵内存
	freeMatrix(&A);
	freeMatrix(&B);
	freeMatrix(&C);
	
	return 0;
}

