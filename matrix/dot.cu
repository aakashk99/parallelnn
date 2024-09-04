#include <stdio.h>

extern "C" {
#include "ops.h"
Matrix* matrix_create(int row, int col);
}

__global__ void matrix_mult(double *a, double *b, double *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    if(col < k && row < m) {
        for(int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

extern "C" Matrix* dot(Matrix* m1, Matrix* m2) {
	if (m1->cols == m2->rows) {
		int m = m1->rows, n = m1->cols, k = m2->cols;

		double *d_a, *d_b, *d_c;
		cudaMalloc((void **) &d_a, sizeof(double)*m*n);
		cudaMalloc((void **) &d_b, sizeof(double)*n*k);
		cudaMalloc((void **) &d_c, sizeof(double)*m*k);
		
		cudaMemcpy(d_a, m1->entries, sizeof(double)*m*n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, m2->entries, sizeof(double)*n*k, cudaMemcpyHostToDevice);

		unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
		unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 dimGrid(grid_cols, grid_rows);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);

		Matrix* res = matrix_create(m, k);
		cudaMemcpy(res->entries, d_c, sizeof(double)*m*k, cudaMemcpyDeviceToHost);

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
