#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include "moderngpu/context.hxx"
#include "moderngpu/kernel_scan.hxx"
#include "moderngpu/kernel_load_balance.hxx"
#include "moderngpu/transform.hxx"
#include <cstdio>
#include <string>
#include "moderngpu/kernel_reduce.hxx"
#include "moderngpu/operators.hxx"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/pair.h>

using namespace mgpu;

struct csr_graph {
    int nodes;
    int edges_count;
    int* rowOffsets;
    int* colIndices;
};

__device__ bool device_binary_search(const int* arr, int len, int val) {
    int l = 0, r = len - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (arr[m] == val) return true;
        if (arr[m] < val) l = m + 1;
        else r = m - 1;
    }
    return false;
}
void load_graph(csr_graph& d_graph) {
    int u, v;
    std::vector<int> h_u, h_v;
    h_u.reserve(10000000);
    h_v.reserve(10000000);

    while (true) {
        int c = std::getc(stdin);
        if (c == EOF) break;
        if (c == '#') {
            while (c != '\n' && c != EOF) c = std::getc(stdin);
            continue;
        }
        std::ungetc(c, stdin);
        if (std::scanf("%d %d", &u, &v) != 2) break;
        if (u == v) continue;
        h_u.push_back(u);
        h_v.push_back(v);
    }

    int n = 0;
    for (size_t i = 0; i < h_u.size(); ++i) {
        if (h_u[i] > n) n = h_u[i];
        if (h_v[i] > n) n = h_v[i];
    }
    n++;

    std::vector<int> h_degrees(n, 0);
    for (size_t i = 0; i < h_u.size(); ++i) {
        h_degrees[h_u[i]]++;
        h_degrees[h_v[i]]++;
    }

    for (size_t i = 0; i < h_u.size(); ++i) {
        int a = h_u[i], b = h_v[i];
        if (std::make_pair(h_degrees[a], a) > std::make_pair(h_degrees[b], b)) {
            std::swap(h_u[i], h_v[i]);
        }
    }

    thrust::device_vector<int> d_u = h_u;
    thrust::device_vector<int> d_v = h_v;

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(d_u.begin(), d_v.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(d_u.end(), d_v.end()));
    thrust::sort(begin, end);

    auto new_end_zip = thrust::unique(begin, end);
    size_t new_size = thrust::distance(begin, new_end_zip);
    d_u.resize(new_size);
    d_v.resize(new_size);

    std::vector<int> final_u(new_size), final_v(new_size);
    thrust::copy(d_u.begin(), d_u.end(), final_u.begin());
    thrust::copy(d_v.begin(), d_v.end(), final_v.begin());

    std::vector<int> h_offsets(n + 1, 0);
    int edge_ptr = 0;
    for (int i = 0; i < n; ++i) {
        h_offsets[i] = edge_ptr;
        while (edge_ptr < new_size && final_u[edge_ptr] == i) {
            edge_ptr++;
        }
    }
    h_offsets[n] = (int)new_size;

    d_graph.nodes = n;
    d_graph.edges_count = (int)new_size;

    cudaMalloc(&d_graph.rowOffsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_graph.colIndices, d_graph.edges_count * sizeof(int));

    cudaMemcpy(d_graph.rowOffsets, h_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph.colIndices, final_v.data(), d_graph.edges_count * sizeof(int), cudaMemcpyHostToDevice);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);

    csr_graph d_graph;
    load_graph(d_graph);
    standard_context_t context(false);

    if (d_graph.edges_count == 0) {
        std::cout << 0 << std::endl;
        return 0;
    }

    mem_t<long long> triangle_counts(d_graph.edges_count, context);
    long long* d_counts = triangle_counts.data();

    int* d_offsets = d_graph.rowOffsets;
    int* d_columns = d_graph.colIndices;
    int n_nodes = d_graph.nodes;



    transform_lbs([=] MGPU_DEVICE(int edge_idx, int u, int local_idx) {
        int v = d_columns[edge_idx];

        int u_start = d_offsets[u];
        int u_len = d_offsets[u + 1] - u_start;

        int v_start = d_offsets[v];
        int v_len = d_offsets[v + 1] - v_start;

        long long local_triangles = 0;
        for (int i = 0; i < v_len; ++i) {
            int w = d_columns[v_start + i];
            if (device_binary_search(d_columns + u_start, u_len, w)) {
                local_triangles++;
            }
        }
        d_counts[edge_idx] = local_triangles;
    }, d_graph.edges_count, d_offsets, n_nodes, context);

    long long *d_total_ptr;
    cudaMalloc(&d_total_ptr, sizeof(long long));

    reduce(d_counts, d_graph.edges_count, d_total_ptr, plus_t<long long>(), context);

    long long total_triangles = 0;
    cudaMemcpy(&total_triangles, d_total_ptr, sizeof(long long), cudaMemcpyDeviceToHost);

    std::cout << total_triangles << std::endl;

    cudaFree(d_graph.rowOffsets);
    cudaFree(d_graph.colIndices);
    cudaFree(d_total_ptr);

    return 0;
}