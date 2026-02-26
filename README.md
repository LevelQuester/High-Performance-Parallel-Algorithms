# High-Performance Parallel Algorithms

A collection of high-performance parallel algorithms implemented in **C++** and **CUDA**. The focus of this repository is on leveraging GPU acceleration to optimize classic computational problems and explore High-Performance Computing (HPC) concepts.

## Implemented Algorithms

### Graph Algorithms
* **Breadth-First Search (BFS)** with Load Balancing
* **Connected Components**
* **Triangle Counting**
* **Floyd-Warshall Algorithm** (All-Pairs Shortest Path)

### Sorting, Searching & Primitives
* **Radix Sort**
* **3SUM Problem**
* **Prefix Sum** (Inclusive & Exclusive Parallel Scan)

### Linear Algebra
* **Matrix Multiplication** (CPU and GPU implementations)

### Dynamic Programming
* **Longest Common Subsequence (LCS)**

### Cryptography
* **Blockchain Simulation** (Proof of Work)

## Prerequisites

* **C++ Compiler** (supporting C++14 or higher)
* **NVIDIA GPU** (Compute Capability 3.0+)
* **CUDA Toolkit** (`nvcc` compiler)
* **CMake** (for project configuration)

## Building the Project

Ensure you have CMake and the CUDA toolkit installed. 

```bash
mkdir build
cd build
cmake ..
make
```
