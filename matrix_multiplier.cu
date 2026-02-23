#include <cuda_runtime.h>

#define TILE 32

__global__ void mm_kernel(int n, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float val = 0.0f;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    for (int m = 0; m < (n + TILE - 1) / TILE; ++m) {
        if (row < n && m * TILE + tx < n)
            As[ty][tx] = A[row * n + m * TILE + tx];
        else
            As[ty][tx] = 0.0f;

        if (m * TILE + ty < n && col < n)
            Bs[ty][tx] = B[(m * TILE + ty) * n + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k)
            val += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = val;
}

void matmul(int n, float* A, float* B, float* C) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    mm_kernel<<<grid, block>>>(n, A, B, C);
}