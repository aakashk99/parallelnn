#include <stdio.h>

extern "C" {
#include "matrix.h"
Matrix* matrix_create(int row, int col);
}

__global__ void MatApply(double* a, double* c, int m, int n, double (*func)(double)) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
		c[row*n + col] = 1.0 / (1 + exp(-1 * a[row*n + col]));
}

extern "C" Matrix* apply(double (*func)(double), Matrix* mat) {
	int m = mat->rows, n = mat->cols;

	double *d_a, *d_c;
	cudaMalloc((void **) &d_a, sizeof(double)*m*n);
	cudaMalloc((void **) &d_c, sizeof(double)*m*n);

	cudaMemcpy(d_a, mat->entries, sizeof(double)*m*n, cudaMemcpyHostToDevice);

	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	MatApply<<<dimGrid, dimBlock>>>(d_a, d_c, m, n, func);

	Matrix* res = matrix_create(m, n);
	cudaMemcpy(res->entries, d_c, sizeof(double)*m*n, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);

	return res;
}
