#include <stdio.h>

extern "C" {
#include "matrix.h"
Matrix* matrix_create(int row, int col);
void matrix_free(Matrix *m);
void matrix_print(Matrix *m);
Matrix* matrix_copy(Matrix *m);
void matrix_save(Matrix* m, char* file_string);
Matrix* matrix_load(char* file_string);
void matrix_randomize(Matrix* m, int n);
int matrix_argmax(Matrix* m);
Matrix* matrix_flatten(Matrix* m, int axis);
Matrix* matrix_expand(double* values, int rows, int cols);
int check_dimensions(Matrix *m1, Matrix *m2);
}

__global__ void MatFill(double* a, int m, int n, int val) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
		a[row*n + col] = val; 
}

extern "C" void matrix_fill(Matrix* mat, int val) {
	int m = mat->rows, n = mat->cols;

	double *d_a;
	cudaMalloc((void **) &d_a, sizeof(double)*m*n);
	
	cudaMemcpy(d_a, mat->entries, sizeof(double)*m*n, cudaMemcpyHostToDevice);

	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	MatFill<<<dimGrid, dimBlock>>>(d_a, m, n, val);

	cudaMemcpy(mat->entries, d_a, sizeof(double)*m*n, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
}
