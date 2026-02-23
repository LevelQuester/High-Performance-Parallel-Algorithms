#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

const int MAX_N = 100005;

__global__ void check3SumKernel(const long long* __restrict__ arr, int n, bool* found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    if (*found) return;

    long long first = arr[idx];
    long long target = -first;

    int left = 0;
    int right = n - 1;

    while (left <= right) {
        if (*found) return;

        long long sum = arr[left] + arr[right];

        if (sum == target) {
            *found = true;
            return;
        } else if (sum < target) {
            left++;
        } else {
            right--;
        }
    }
}

void solve() {
    int z;
    std::cin >> z;

    long long* d_arr;
    bool* d_found;

    cudaMalloc(&d_arr, MAX_N * sizeof(long long));
    cudaMalloc(&d_found, sizeof(bool));

    std::vector<long long> h_arr;
    h_arr.reserve(MAX_N);

    while (z--) {
        int n;
        std::cin >> n;

        h_arr.resize(n);
        for (int i = 0; i < n; ++i) {
            std::cin >> h_arr[i];
        }

        cudaMemcpy(d_arr, h_arr.data(), n * sizeof(long long), cudaMemcpyHostToDevice);

        thrust::device_ptr<long long> t_arr(d_arr);
        thrust::sort(t_arr, t_arr + n);

        bool h_found = false;
        cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        check3SumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n, d_found);

        cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

        if (h_found) std::cout << "YES\n";
        else std::cout << "NO\n";
    }

    cudaFree(d_arr);
    cudaFree(d_found);
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    solve();
    return 0;
}