#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

typedef struct {
    int rows;
    int cols;
    float **data;
} Matrix;

Matrix createRandomMatrix(int rows, int cols);
Matrix matrixMultiply(const Matrix *A, const Matrix *B);
Matrix createMatrix(int rows, int cols);
void printMatrix(const Matrix* mat, int maxRows, int maxCols);
void freeMatrix(Matrix *mat);

#endif

