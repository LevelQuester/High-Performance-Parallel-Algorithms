#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define ELEMENTS_PER_BLOCK (2 * BLOCK_SIZE)

__global__ void local_scan_kernel(int n, int64_t* A, int64_t* S, int64_t* G) {
    __shared__ int64_t temp[ELEMENTS_PER_BLOCK];

    int tid = threadIdx.x;
    int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;

    int idx1 = block_offset + 2 * tid;
    int idx2 = block_offset + 2 * tid + 1;

    temp[2 * tid] = (idx1 < n) ? A[idx1] : 0;
    temp[2 * tid + 1] = (idx2 < n) ? A[idx2] : 0;

    __syncthreads();

    for (int d = 1; d <= BLOCK_SIZE; d *= 2) {
        int k = 2 * d * (tid + 1) - 1;
        if (k < ELEMENTS_PER_BLOCK) {
            temp[k] += temp[k - d];
        }
        __syncthreads();
    }

    if (tid == BLOCK_SIZE - 1 && G != nullptr) {
        G[blockIdx.x] = temp[ELEMENTS_PER_BLOCK - 1];
    }

    if (tid == BLOCK_SIZE - 1) {
        temp[ELEMENTS_PER_BLOCK - 1] = 0;
    }
    __syncthreads();

    for (int d = BLOCK_SIZE; d >= 1; d /= 2) {
        int k = 2 * d * (tid + 1) - 1;
        if (k < ELEMENTS_PER_BLOCK) {
            int64_t t = temp[k - d];
            temp[k - d] = temp[k];
            temp[k] += t;
        }
        __syncthreads();
    }

    if (idx1 < n) S[idx1] = temp[2 * tid];
    if (idx2 < n) S[idx2] = temp[2 * tid + 1];
}

__global__ void add_offsets_kernel(int n, int64_t* S, int64_t* G) {
    int64_t offset = G[blockIdx.x];

    int block_offset = blockIdx.x * ELEMENTS_PER_BLOCK;
    int gid1 = block_offset + 2 * threadIdx.x;
    int gid2 = block_offset + 2 * threadIdx.x + 1;

    if (gid1 < n) S[gid1] += offset;
    if (gid2 < n) S[gid2] += offset;
}

void prefixsum(int n, int64_t* A, int64_t* S) {
    if (n <= 0) return;

    int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    if (num_blocks == 1) {
        local_scan_kernel<<<1, BLOCK_SIZE>>>(n, A, S, nullptr);
    } else {
        int64_t* G;
        cudaMalloc(&G, num_blocks * sizeof(int64_t));

        local_scan_kernel<<<num_blocks, BLOCK_SIZE>>>(n, A, S, G);

        prefixsum(num_blocks, G, G);

        add_offsets_kernel<<<num_blocks, BLOCK_SIZE>>>(n, S, G);

        cudaFree(G);
    }
}