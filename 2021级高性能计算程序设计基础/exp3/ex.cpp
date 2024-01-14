#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "parallel_for.h"

struct args{
    int *A, *B, *C;
    int *m, *n, *k;
    args(int *tA, int *tB, int *tC, int *tm, int *tn, int *tk){
        A = tA;
        B = tB;
        C = tC;
        m = tm;
        n = tn;
        k = tk;
    }
};

// 矩阵乘法
void* matrixMultiply(void *arg) {
	struct for_index *index = (struct for_index *)arg;
    	struct args *true_arg = (struct args *)(index->args);
    	for (int i = index->start; i < index->end; i = i + index->increment){
            for (int j = 0; j < *true_arg->k; ++j){
            int temp = 0;
            for (int z = 0; z < *true_arg->n; ++z)
                temp += true_arg->A[i * (*true_arg->n) + z] * true_arg->B[z * (*true_arg->k) + j];
            true_arg->C[i * (*true_arg->k) + j] = temp;
            }
   	}
    
}

void printMatrix(const int* mat, int numRows, int numCols, int maxRows, int maxCols) {
    for (int i = 0; i < numRows && i < maxRows; i++) {
        for (int j = 0; j < numCols && j < maxCols; j++) {
            std::cout << mat[i * numCols + j] << ' ';
        }

        if (numCols > maxCols) {
            std::cout << "..."; // 用省略号代替超出部分的列
        }

        std::cout << std::endl;
    }

    if (numRows > maxRows) {
        std::cout << "..." << std::endl; // 用省略号代替超出部分的行
    }
}

int main(int argc, char * argv[]) {
    srand((unsigned)time(0)); // 初始化随机数生成器
	
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    int thread_count = atoi(argv[4]);
	
	printf("N=%d, M=%d, K=%d\n", N, M, K);
	
	clock_t start,end;
	
	int *A, *B, *C;
	A = new int[M * N];
   	B = new int[N * K];
    	C = new int[M * K];
    	
    	for (int i = 0; i < M; i++){
            for (int j = 0; j < N; j++){
                A[i*M+j] = (int)rand() %10;
                B[i*N+j] = (int)rand()  %10;
            }
    	}
	
 	struct args *arg = new args(A, B, C, &M, &N, &K);
   	 start=clock();
   	 parallel_for(0, M, 1, matrixMultiply, arg, thread_count);
   	 end=clock();
   	 double endtime=(double)(end-start)/CLOCKS_PER_SEC/thread_count;
	
	printf("用时：%f ms.\n",endtime*1000);
	
	// 打印矩阵 A、B 和 C 的内容
	printf("\nMatrix A:\n");
	printMatrix(A, M, N, 2, 2);
	printf("\nMatrix B:\n");
	printMatrix(B, N, K, 2, 2);
	printf("\nMatrix C (Result of A * B):\n");
	printMatrix(C, M, K, 2, 2);
	
	// 释放矩阵内存
	free(A);
	free(B);
	free(C);
	
	return 0;
}

