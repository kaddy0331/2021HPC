#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


// 定义单精度浮点数矩阵结构
typedef struct {
    int rows;
    int cols;
    float** data;
} Matrix;

// 数据结构传递给线程的参数
typedef struct {
    int thread_id;
    int num_threads;
    const Matrix* A;
    const Matrix* B;
    Matrix* C;
} ThreadArgs;

// 创建一个随机矩阵
Matrix createRandomMatrix(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (float**)malloc(rows * sizeof(float*));
    
    for (int i = 0; i < rows; i++) {
        mat.data[i] = (float*)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {
            mat.data[i][j] = (float)rand() / RAND_MAX; // 生成0到1之间的随机浮点数
        }
    }
    
    return mat;
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

// 释放矩阵内存
void freeMatrix(Matrix* mat) {
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
}

// 用于线程的矩阵乘法函数
void* matrixMultiplyThread(void* thread_args) {
    ThreadArgs* args = (ThreadArgs*)thread_args;
    const Matrix* A = args->A;
    const Matrix* B = args->B;
    Matrix* C = args->C;
    int num_threads = args->num_threads;
    int thread_id = args->thread_id;
    
    int start_row = (A->rows / num_threads) * thread_id;
    int end_row = (thread_id == num_threads - 1) ? A->rows : start_row + (A->rows / num_threads);
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < B->cols; j++) {
            for (int k = 0; k < A->cols; k++) {
                C->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }
    
    return NULL;
}

int main(int argc, char * argv[] ) {
    srand(time(NULL)); // 初始化随机数生成器
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int num_threads = atoi(argv[4]); // 选择线程数量
    

    printf("N=%d, M=%d, K=%d\n", N, M, K);
    printf("num_threads=%d\n", num_threads);
    
    // 创建随机矩阵 A 和 B
    Matrix A = createRandomMatrix(M, N);
    Matrix B = createRandomMatrix(N, K);
    
    // 初始化结果矩阵 C
    Matrix C;
    C.rows = M;
    C.cols = K;
    C.data = (float**)malloc(M * sizeof(float*));
    for (int i = 0; i < M; i++) {
        C.data[i] = (float*)calloc(K, sizeof(float));
    }
    
    pthread_t threads[num_threads];
    ThreadArgs thread_args[num_threads];

    // 创建和启动线程
    for (int i = 0; i < num_threads; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].num_threads = num_threads;
        thread_args[i].A = &A;
        thread_args[i].B = &B;
        thread_args[i].C = &C;

        if (pthread_create(&threads[i], NULL, matrixMultiplyThread, &thread_args[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    // 计算执行时间
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start); // 记录开始时间

    // 等待线程结束
    for (int i = 0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("pthread_join");
            return 1;
        }
    }

    // 计算执行时间
    clock_gettime(CLOCK_MONOTONIC, &end); // 记录结束时间

    // 计算执行时间（以毫秒为单位）
    double start_time = (double)start.tv_sec * 1e3 + (double)start.tv_nsec * 1e-6;
    double end_time = (double)end.tv_sec * 1e3 + (double)end.tv_nsec * 1e-6;
    double execution_time = end_time - start_time;

    printf("Execution time: %f milliseconds\n", execution_time);

    // 打印矩阵 A、B 和 C 的内容
    printf("\nMatrix A:\n");
    printMatrix(&A,2,2);
    printf("\nMatrix B:\n");
    printMatrix(&B,2,2);
    printf("\nMatrix C (Result of A * B):\n");
    printMatrix(&C,2,2);
    
    // 释放矩阵内存
    freeMatrix(&A);
    freeMatrix(&B);
    freeMatrix(&C);
    
    return 0;
}

