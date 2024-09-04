#include <stdio.h>

extern "C" {
#include "ops.h"
Matrix* matrix_create(int row, int col);
}

__global__ void MatScale(double* a, double* c, int m, int n, double factor) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
		c[row*n + col] = a[row*n + col] * factor; 
}

extern "C" Matrix* scale(double factor, Matrix* mat) {
	int m = mat->rows, n = mat->cols;

	double *d_a, *d_c;
	cudaMalloc((void **) &d_a, sizeof(double)*m*n);
	cudaMalloc((void **) &d_c, sizeof(double)*m*n);
	
	cudaMemcpy(d_a, mat->entries, sizeof(double)*m*n, cudaMemcpyHostToDevice);

	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	MatScale<<<dimGrid, dimBlock>>>(d_a, d_c, m, n, factor);

	Matrix* res = matrix_create(m, n);
	cudaMemcpy(res->entries, d_c, sizeof(double)*m*n, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_c);

	return res;
}
