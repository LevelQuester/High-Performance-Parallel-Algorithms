#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#define BLOCK_SIZE 256

__global__ void get_flags_kernel(int n, const int* data, int* flags, int bit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        flags[i] = !((data[i] >> bit) & 1);
    }
}

__global__ void scatter_kernel(int n, const int* src, int* dst, const int* scan_res, int total_zeros, int bit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = src[i];
        int bit_val = (val >> bit) & 1;
        int dest_idx;
        if (bit_val == 0) {
            dest_idx = scan_res[i];
        } else {
            dest_idx = total_zeros + (i - scan_res[i]);
        }
        dst[dest_idx] = val;
    }
}

void gpusort(int n, int* a) {
    if (n <= 1) return;

    int *d_buffer, *d_flags;
    cudaMalloc(&d_buffer, n * sizeof(int));
    cudaMalloc(&d_flags, n * sizeof(int));
    int* src = a;
    int* dst = d_buffer;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    thrust::device_ptr<int> dev_flags_ptr(d_flags);

    for (int bit = 0; bit < 30; ++bit) {
        get_flags_kernel<<<blocks, BLOCK_SIZE>>>(n, src, d_flags, bit);

        thrust::exclusive_scan(dev_flags_ptr, dev_flags_ptr + n, dev_flags_ptr);

        int last_val, last_scan;
        cudaMemcpy(&last_val, src + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan, d_flags + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

        int last_bit_is_zero = !((last_val >> bit) & 1);
        int total_zeros = last_scan + last_bit_is_zero;

        scatter_kernel<<<blocks, BLOCK_SIZE>>>(n, src, dst, d_flags, total_zeros, bit);

        int* temp = src;
        src = dst;
        dst = temp;
    }

    if (src != a) {
        cudaMemcpy(a, src, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_buffer);
    cudaFree(d_flags);
}