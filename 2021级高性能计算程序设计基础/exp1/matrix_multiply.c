#include "matrix_multiply.h"
#include <stdio.h>
#include <stdlib.h>


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

Matrix createMatrix(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (float **)malloc(rows * sizeof(float *));
    
    for (int i = 0; i < rows; i++) {
        mat.data[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++) {
            mat.data[i][j] = 0.0f;
        }
    }
    
    return mat;
}

Matrix matrixMultiply(const Matrix* A, const Matrix* B) {
	if (A->cols != B->rows) {
		fprintf(stderr, "Matrix dimensions are incompatible for multiplication\n");
		exit(1);
	}
	
	Matrix C;
	C.rows = A->rows;
	C.cols = B->cols;
	C.data = (float**)malloc(C.rows * sizeof(float*));
	
	for (int i = 0; i < C.rows; i++) {
		C.data[i] = (float*)calloc(C.cols, sizeof(float));
		for (int k = 0; k < A->cols; k++) {
			for (int j = 0; j < C.cols; j++) {
				C.data[i][j] += A->data[i][k] * B->data[k][j];
			}
		}
	}
	
	return C;
}

void printMatrix(const Matrix* mat, int maxRows, int maxCols) {
	int numRows = mat->rows;
	int numCols = mat->cols;
	
	for (int i = 0; i < numRows && i < maxRows; i++) {
		for (int j = 0; j < numCols && j < maxCols; j++) {
			printf("%f ", mat->data[i][j]);
		}
		
		if (numCols > maxCols) {
			printf("...");
		}
		
		printf("\n");
	}
	
	if (numRows > maxRows) {
		printf("...\n"); 
	}
}

void freeMatrix(Matrix *mat) {
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
}

