#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

#define TILE 128

__global__ void init(int n, int* dev_cc)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n)
    {
        dev_cc[id] = id;
    }
}

__global__ void hook(int m, int n, int* dev_cc, int* dev_edges)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < m)
    {
        int u = dev_edges[2*id];
        int v = dev_edges[2*id+1];

        int p1 = dev_cc[u];
        int p2 = dev_cc[v];

        if(p1 != p2)
        {
            int high = max(p1, p2);
            int low = min(p1, p2);

            atomicMin(&dev_cc[high], low);
        }
    }
}

__global__ void compress(int m, int n, int* dev_cc, int* dev_edges, bool* dev_changed)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n)
    {
        int parent = dev_cc[id];
        int grand_parent = dev_cc[parent];

        if(parent != grand_parent)
        {
            dev_cc[id] = grand_parent;
            *dev_changed = true;
        }
    }
}

void find_cc(int n, int m, int* edges, int* cc){
    int init_blocks = (n + TILE - 1) / TILE;
    init<<<init_blocks, TILE>>>(n, cc);
    cudaDeviceSynchronize();

    bool* dev_changed = nullptr;
    cudaMalloc(&dev_changed, sizeof(bool));

    bool host_changed = false;

    do{
        host_changed = false;
        cudaMemcpy(dev_changed, &host_changed, sizeof(bool), cudaMemcpyHostToDevice);

        int hook_blocks = (m + TILE - 1) / TILE;
        hook<<<hook_blocks, TILE>>>(m, n, cc, edges);
        cudaDeviceSynchronize();

        int compress_blocks = (n + TILE - 1) / TILE;
        compress<<<compress_blocks, TILE>>>(m, n, cc, edges, dev_changed);
        cudaDeviceSynchronize();

        cudaMemcpy(&host_changed, dev_changed, sizeof(bool), cudaMemcpyDeviceToHost);

    } while(host_changed);

    cudaFree(dev_changed);
}

