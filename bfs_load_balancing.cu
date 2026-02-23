#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "moderngpu/context.hxx"
#include "moderngpu/kernel_scan.hxx"
#include "moderngpu/kernel_load_balance.hxx"
#include "moderngpu/transform.hxx"
#include "moderngpu/operators.hxx"

using namespace mgpu;

struct csr_graph {
    int nodes;
    int edges_count;
    int* rowOffsets;
    int* colIndices;
    int* degree;
};

__global__ void fill_kernel(int* arr, int n, int val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

void load_graph(csr_graph& d_graph) {

    std::string line;
    int max_node = 0;
    std::vector<std::pair<int, int>> edges;

    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        int u, v;
        if (sscanf(line.c_str(), "%d %d", &u, &v) == 2) {
            edges.push_back({u, v});
            edges.push_back({v, u});
            max_node = std::max({max_node, u, v});
        }
    }

    int n = max_node + 1;
    std::vector<int> h_offsets(n + 1, 0);
    std::vector<int> h_degree(n, 0);
    for (auto& e : edges) h_degree[e.first]++;
    for (int i = 0; i < n; ++i) h_offsets[i + 1] = h_offsets[i] + h_degree[i];

    std::vector<int> h_indices(edges.size());
    std::vector<int> current_pos(n, 0);
    for (auto& e : edges) {
        h_indices[h_offsets[e.first] + current_pos[e.first]++] = e.second;
    }

    d_graph.nodes = n;
    d_graph.edges_count = (int)edges.size();
    cudaMalloc(&d_graph.rowOffsets, (n + 1) * sizeof(int));
    cudaMalloc(&d_graph.colIndices, d_graph.edges_count * sizeof(int));
    cudaMalloc(&d_graph.degree, n * sizeof(int));

    cudaMemcpy(d_graph.rowOffsets, h_offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph.colIndices, h_indices.data(), d_graph.edges_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph.degree, h_degree.data(), n * sizeof(int), cudaMemcpyHostToDevice);
}

void run_bfs(int start_node, const csr_graph& g, standard_context_t& context,
             int* visited, int* frontier, int* nextFrontier, int* lastWriter,
             int* expandedNbrs, int* isSelected, int* d_frontierDeg, int* d_prefixDeg, int* d_selectPrefix) {
    int n = g.nodes;
    fill_kernel<<<(n + 255) / 256, 256>>>(visited, n, 0);

    cudaMemcpy(frontier, &start_node, sizeof(int), cudaMemcpyHostToDevice);
    int one = 1;
    cudaMemcpy(visited + start_node, &one, sizeof(int), cudaMemcpyHostToDevice);

    int frontierSize = 1;
    int resultLevel = 0;

    for (int level = 0; level < n; ++level) {
        int* d_deg = d_frontierDeg;
        transform([=] MGPU_DEVICE(int i) {
            d_deg[i] = g.degree[frontier[i]];
        }, frontierSize, context);

        scan<scan_type_exc>(d_frontierDeg, frontierSize, d_prefixDeg, plus_t<int>(), d_selectPrefix, context);
        int totalExpanded;
        cudaMemcpy(&totalExpanded, d_selectPrefix, sizeof(int), cudaMemcpyDeviceToHost);

        if (totalExpanded == 0) {
            resultLevel = level;
            break;
        }

        fill_kernel<<<(n + 255) / 256, 256>>>(lastWriter, n, -1);

        transform_lbs([=] MGPU_DEVICE(int e, int i, int local) {
            int v = frontier[i];
            int u = g.colIndices[g.rowOffsets[v] + local];
            expandedNbrs[e] = u;
            lastWriter[u] = e;
        }, totalExpanded, d_prefixDeg, frontierSize, context);

        int* sel_ptr = isSelected;
        transform([=] MGPU_DEVICE(int e) {
            int u = expandedNbrs[e];
            if (lastWriter[u] == e && visited[u] == 0) {
                visited[u] = 1;
                sel_ptr[e] = 1;
            } else {
                sel_ptr[e] = 0;
            }
        }, totalExpanded, context);

        scan<scan_type_exc>(isSelected, totalExpanded, d_selectPrefix, plus_t<int>(), d_frontierDeg, context);

        int nextSize;
        cudaMemcpy(&nextSize, d_frontierDeg, sizeof(int), cudaMemcpyDeviceToHost);

        if (nextSize == 0) {
            resultLevel = level;
            break;
        }

        transform([=] MGPU_DEVICE(int e) {
            if (isSelected[e]) {
                int idx = d_selectPrefix[e];
                nextFrontier[idx] = expandedNbrs[e];
            }
        }, totalExpanded, context);

        std::swap(frontier, nextFrontier);
        frontierSize = nextSize;
    }

    std::cout << resultLevel << std::endl;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(NULL);

    csr_graph d_graph;
    load_graph(d_graph);
    standard_context_t context(false);

    int n = d_graph.nodes;
    int e_count = d_graph.edges_count;

    int *visited, *frontier, *nextFrontier, *lastWriter, *expandedNbrs, *isSelected, *d_frontierDeg, *d_prefixDeg, *d_selectPrefix;
    cudaMalloc(&visited, n * sizeof(int));
    cudaMalloc(&frontier, n * sizeof(int));
    cudaMalloc(&nextFrontier, n * sizeof(int));
    cudaMalloc(&lastWriter, n * sizeof(int));
    cudaMalloc(&expandedNbrs, e_count * sizeof(int));
    cudaMalloc(&isSelected, e_count * sizeof(int));
    cudaMalloc(&d_frontierDeg, n * sizeof(int));
    cudaMalloc(&d_prefixDeg, (n + 1) * sizeof(int));
    cudaMalloc(&d_selectPrefix, (e_count + 1) * sizeof(int));

    for (int i = 0; i * 2019 < n; ++i) {
        run_bfs(i * 2019, d_graph, context, visited, frontier, nextFrontier, lastWriter, expandedNbrs, isSelected, d_frontierDeg, d_prefixDeg, d_selectPrefix);
    }

    cudaFree(visited); cudaFree(frontier); cudaFree(nextFrontier);
    cudaFree(lastWriter); cudaFree(expandedNbrs); cudaFree(isSelected);
    cudaFree(d_frontierDeg); cudaFree(d_prefixDeg); cudaFree(d_selectPrefix);
    cudaFree(d_graph.rowOffsets); cudaFree(d_graph.colIndices); cudaFree(d_graph.degree);

    return 0;
}