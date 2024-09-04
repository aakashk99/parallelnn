#include <stdio.h>

extern "C" {
#include "ops.h"
Matrix* matrix_create(int row, int col);
int check_dimensions(Matrix *m1, Matrix *m2);
}

__global__ void MatSub(double* a, double* b, double* c, int m, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
		c[row*n + col] = a[row*n + col] - b[row*n + col];
}

extern "C" Matrix* subtract(Matrix* m1, Matrix* m2) {
	if (check_dimensions(m1, m2)) {
		int m = m1->rows, n = m1->cols;

		double *d_a, *d_b, *d_c;
		cudaMalloc((void **) &d_a, sizeof(double)*m*n);
		cudaMalloc((void **) &d_b, sizeof(double)*m*n);
		cudaMalloc((void **) &d_c, sizeof(double)*m*n);
		
		cudaMemcpy(d_a, m1->entries, sizeof(double)*m*n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, m2->entries, sizeof(double)*m*n, cudaMemcpyHostToDevice);

		unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
		unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 dimGrid(grid_cols, grid_rows);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		MatSub<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n);

		Matrix* res = matrix_create(m, n);
		cudaMemcpy(res->entries, d_c, sizeof(double)*m*n, cudaMemcpyDeviceToHost);
	
		cudaFree(d_a);
    		cudaFree(d_b);
    		cudaFree(d_c);

		return res;
	}
	else {
		printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);	
	}
}
