#include <stdio.h>
#include "matrix_multiply.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <Matrix_A_Rows> <Matrix_A_Cols> <Matrix_B_Cols>\n", argv[0]);
        return 1;
    }

    int rows_A = atoi(argv[1]);
    int cols_A = atoi(argv[2]);
    int cols_B = atoi(argv[3]);

    if (cols_A != cols_B) {
        fprintf(stderr, "Matrix dimensions are incompatible for multiplication\n");
        return 1;
    }

    // 创建矩阵 A 和 B
    Matrix A = createRandomMatrix(rows_A, cols_A);
    Matrix B = createRandomMatrix(cols_A, cols_B);


    // 打印矩阵 A 和 B
    printf("Matrix A:\n");
    printMatrix(&A, 2, 2);

    printf("Matrix B:\n");
    printMatrix(&B, 2, 2);

    // 记录开始时间
    clock_t start_time = clock();

    // 执行矩阵乘法
    Matrix C = matrixMultiply(&A, &B);

    // 记录结束时间
    clock_t end_time = clock();

    // 打印矩阵 C
    printf("Matrix C (Result of A * B):\n");
    printMatrix(&C, 2, 2);

    // 释放矩阵内存
    freeMatrix(&A);
    freeMatrix(&B);
    freeMatrix(&C);

    // 计算并打印运行时间（以秒为单位）
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", execution_time);

    return 0;
}

