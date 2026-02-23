#include <iostream>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#define TILE 256

__global__ void dp_kernel(int k, int size1, int size2, const char* d_s1, const char* d_s2, const int* d_prev2, const int* d_prev1, int* d_curr) {
    int min_i = max(0, k - (size2 - 1));
    int max_i = min(k, size1 - 1);
    int count = max_i - min_i + 1;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    int i = min_i + tid;
    int j = k - i;
    int res = 0;

    int p1_min_i = max(0, (k - 1) - (size2 - 1));
    if (i > 0) {
        res = max(res, d_prev1[(i - 1) - p1_min_i]);
    }
    if (j > 0) {
        res = max(res, d_prev1[i - p1_min_i]);
    }

    if (d_s1[i] == d_s2[j]) {
        int val_prev2 = 0;
        if (i > 0 && j > 0) {
            int p2_min_i = max(0, (k - 2) - (size2 - 1));
            val_prev2 = d_prev2[(i - 1) - p2_min_i];
        }
        res = max(res, val_prev2 + 1);
    }
    d_curr[tid] = res;
}

void solve() {
    std::string s1, s2;
    if (!(std::cin >> s1 >> s2)) return;

    int size1 = s1.length();
    int size2 = s2.length();

    char *d_s1, *d_s2;
    cudaMalloc(&d_s1, size1);
    cudaMalloc(&d_s2, size2);
    cudaMemcpy(d_s1, s1.c_str(), size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s2, s2.c_str(), size2, cudaMemcpyHostToDevice);

    int max_diag_len = std::min(size1, size2) + 2;
    int *d_prev2, *d_prev1, *d_curr;
    cudaMalloc(&d_prev2, max_diag_len * sizeof(int));
    cudaMalloc(&d_prev1, max_diag_len * sizeof(int));
    cudaMalloc(&d_curr,  max_diag_len * sizeof(int));
    cudaMemset(d_prev2, 0, max_diag_len * sizeof(int));
    cudaMemset(d_prev1, 0, max_diag_len * sizeof(int));

    int total_diagonals = size1 + size2 - 1;
    for (int k = 0; k < total_diagonals; ++k) {
        int min_i = std::max(0, k - (size2 - 1));
        int max_i = std::min(k, size1 - 1);
        int current_len = max_i - min_i + 1;

        if (current_len > 0) {
            int blocks = (current_len + TILE - 1) / TILE;
            dp_kernel<<<blocks, TILE>>>(k, size1, size2, d_s1, d_s2, d_prev2, d_prev1, d_curr);
        }

        int* temp = d_prev2;
        d_prev2 = d_prev1;
        d_prev1 = d_curr;
        d_curr = temp;
    }

    int ans = 0;
    cudaMemcpy(&ans, d_prev1, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << ans << std::endl;

    cudaFree(d_s1); cudaFree(d_s2);
    cudaFree(d_prev2); cudaFree(d_prev1); cudaFree(d_curr);
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    int z;
    if (std::cin >> z) {
        while (z--) {
            solve();
        }
    }
    return 0;
}